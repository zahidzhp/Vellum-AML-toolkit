"""Probe engine: orchestrates baseline and shallow probe runs, config-driven."""

import logging
from typing import Any

import numpy as np

from aml_toolkit.artifacts import DatasetManifest, ProbeResult, ProbeResultSet
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import ModalityType
from aml_toolkit.intake.split_builder import SplitResult
from aml_toolkit.probes.baseline_models import MajorityBaseline, StratifiedBaseline
from aml_toolkit.probes.image_embedding_probes import run_embedding_probe
from aml_toolkit.probes.tabular_probes import run_tabular_probe

logger = logging.getLogger("aml_toolkit")

# Probes that support intervention branches
_TABULAR_PROBES = {"logistic", "rf", "xgb"}
_EMBEDDING_PROBES = {"embedding_logistic", "embedding_mlp"}


def run_probes(
    data: Any,
    manifest: DatasetManifest,
    split: SplitResult,
    config: ToolkitConfig,
) -> ProbeResultSet:
    """Run all configured probes and produce a ProbeResultSet.

    Args:
        data: Raw data dict from intake.
        manifest: The DatasetManifest.
        split: The SplitResult with train/val/test indices.
        config: Toolkit configuration.

    Returns:
        ProbeResultSet with baseline, shallow, and intervention branch results.
    """
    metrics = [config.probes.metric]
    # Always include accuracy for reference
    if "accuracy" not in metrics:
        metrics.append("accuracy")

    seed = config.seed
    enabled_probes = set(config.probes.enabled_probes)
    intervention_branches = config.probes.intervention_branches

    # Extract train/val features and labels
    X_train, y_train, X_val, y_val = _extract_train_val(data, manifest, split)

    baseline_results: list[ProbeResult] = []
    shallow_results: list[ProbeResult] = []
    intervention_branch_results: list[ProbeResult] = []

    # 1. Baselines (always run)
    if "majority" in enabled_probes:
        logger.info("Running majority baseline probe")
        maj = MajorityBaseline()
        maj.fit(y_train)
        baseline_results.append(maj.to_probe_result(y_train, y_val, metrics))

    if "stratified" in enabled_probes:
        logger.info("Running stratified baseline probe")
        strat = StratifiedBaseline()
        strat.fit(y_train, seed=seed)
        baseline_results.append(strat.to_probe_result(y_train, y_val, metrics))

    # 2. Shallow probes
    if manifest.modality in (ModalityType.TABULAR, ModalityType.EMBEDDING):
        # Determine which probes to run
        if manifest.modality == ModalityType.TABULAR:
            probe_names = [p for p in enabled_probes if p in _TABULAR_PROBES]
        else:
            probe_names = [p for p in enabled_probes if p in _EMBEDDING_PROBES]
            # Also allow standard embedding probes by default
            if not probe_names:
                probe_names = ["embedding_logistic"]

        # No-intervention run
        for probe_name in probe_names:
            logger.info(f"Running probe: {probe_name} (no intervention)")
            result = _run_single_probe(
                probe_name, X_train, y_train, X_val, y_val,
                metrics, "none", seed, manifest.modality,
            )
            shallow_results.append(result)

        # 3. Intervention branches (tabular probes only)
        if manifest.modality == ModalityType.TABULAR:
            tabular_probes = [p for p in probe_names if p in _TABULAR_PROBES]
            for branch in intervention_branches:
                if branch == "none":
                    continue
                for probe_name in tabular_probes:
                    logger.info(f"Running probe: {probe_name} (branch: {branch})")
                    result = _run_single_probe(
                        probe_name, X_train, y_train, X_val, y_val,
                        metrics, branch, seed, manifest.modality,
                    )
                    intervention_branch_results.append(result)

    elif manifest.modality == ModalityType.IMAGE:
        # For image modality without pre-extracted embeddings, log a note
        logger.info(
            "Image modality without pre-extracted embeddings. "
            "Embedding probe requires pre-computed features."
        )

    # Build intervention sensitivity summary
    sensitivity = _build_intervention_sensitivity(
        shallow_results, intervention_branch_results, config.probes.metric
    )

    # Build shortlist recommendation
    shortlist = _build_shortlist(shallow_results, config.probes.metric)

    return ProbeResultSet(
        baseline_results=baseline_results,
        shallow_results=shallow_results,
        intervention_branch_results=intervention_branch_results,
        selected_metrics=metrics,
        intervention_sensitivity_summary=sensitivity,
        shortlist_recommendation=shortlist,
    )


def _extract_train_val(
    data: Any,
    manifest: DatasetManifest,
    split: SplitResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract train/val feature and label arrays from data dict."""
    if "df" in data and manifest.feature_columns:
        df = data["df"]
        X = df[manifest.feature_columns].values.astype(np.float64)
        y = df[manifest.target_column].values
        # Handle NaN in features for probes
        X = np.nan_to_num(X, nan=0.0)
    elif "embeddings" in data:
        X = data["embeddings"]
        y = data["labels"]
    else:
        # Fallback: empty arrays
        X = np.zeros((0, 1))
        y = np.zeros(0)

    X_train = X[split.train_indices]
    y_train = y[split.train_indices]
    X_val = X[split.val_indices]
    y_val = y[split.val_indices]

    return X_train, y_train, X_val, y_val


def _run_single_probe(
    probe_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metrics: list[str],
    intervention_branch: str,
    seed: int,
    modality: ModalityType,
) -> ProbeResult:
    """Dispatch to the appropriate probe runner."""
    if probe_name in _TABULAR_PROBES:
        return run_tabular_probe(
            probe_name, X_train, y_train, X_val, y_val,
            metrics, intervention_branch, seed,
        )
    elif probe_name in _EMBEDDING_PROBES:
        return run_embedding_probe(
            probe_name, X_train, y_train, X_val, y_val, metrics, seed,
        )
    else:
        return ProbeResult(
            model_name=probe_name,
            intervention_branch=intervention_branch,
            notes=[f"Unsupported probe: {probe_name}"],
        )


def _build_intervention_sensitivity(
    shallow_results: list[ProbeResult],
    intervention_results: list[ProbeResult],
    primary_metric: str,
) -> dict[str, Any]:
    """Compare no-intervention vs intervention results per probe."""
    summary: dict[str, Any] = {}

    baseline_scores: dict[str, float] = {}
    for r in shallow_results:
        if r.intervention_branch == "none" and primary_metric in r.val_metrics:
            baseline_scores[r.model_name] = r.val_metrics[primary_metric]

    for r in intervention_results:
        if primary_metric not in r.val_metrics:
            continue
        base = baseline_scores.get(r.model_name)
        if base is not None:
            delta = r.val_metrics[primary_metric] - base
            key = f"{r.model_name}_{r.intervention_branch}"
            summary[key] = {
                "baseline": base,
                "with_intervention": r.val_metrics[primary_metric],
                "delta": delta,
            }

    return summary


def _build_shortlist(
    shallow_results: list[ProbeResult],
    primary_metric: str,
) -> list[str]:
    """Rank probes by primary metric and return shortlist."""
    scored = []
    for r in shallow_results:
        if primary_metric in r.val_metrics:
            scored.append((r.model_name, r.val_metrics[primary_metric]))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored]
