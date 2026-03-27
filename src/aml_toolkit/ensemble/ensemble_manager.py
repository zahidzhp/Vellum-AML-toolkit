"""Ensemble manager: selective model combination with gain-based acceptance logic."""

import logging
from itertools import combinations

import numpy as np
from sklearn.metrics import f1_score

from aml_toolkit.artifacts import EnsembleReport
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.ensemble.soft_voting import SoftVotingStrategy
from aml_toolkit.ensemble.weighted_averaging import WeightedAveragingStrategy
from aml_toolkit.interfaces.candidate_model import CandidateModel
from aml_toolkit.interfaces.ensemble_strategy import EnsembleStrategy

logger = logging.getLogger("aml_toolkit")

_STRATEGY_REGISTRY: dict[str, type[EnsembleStrategy]] = {
    "soft_voting": SoftVotingStrategy,
    "weighted_averaging": WeightedAveragingStrategy,
}


def run_ensemble(
    trained_models: dict[str, CandidateModel],
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ToolkitConfig,
    metric: str = "macro_f1",
) -> EnsembleReport:
    """Attempt selective ensemble construction.

    For each enabled strategy:
    1. Collect probability predictions from all probabilistic candidates.
    2. Evaluate the ensemble score on the validation set.
    3. Compare marginal gain against the best individual model.
    4. Accept only if the gain meets the configured threshold.

    If no strategy meets the threshold, returns a report with
    ensemble_selected=False and the rejection reason.

    Args:
        trained_models: Dict of candidate_id -> trained CandidateModel.
        X_val: Validation features.
        y_val: Validation labels.
        config: Toolkit configuration.
        metric: Metric to use for gain evaluation.

    Returns:
        EnsembleReport with decision and reasoning.
    """
    ens_config = config.ensemble

    # Collect probabilistic candidates
    prob_models: dict[str, CandidateModel] = {}
    for cid, model in trained_models.items():
        if model.is_probabilistic():
            proba = model.predict_proba(X_val)
            if proba is not None:
                prob_models[cid] = model

    if len(prob_models) < 2:
        return EnsembleReport(
            ensemble_selected=False,
            rejection_reason="Fewer than 2 probabilistic candidates available for ensembling.",
            notes=[f"Available probabilistic models: {list(prob_models.keys())}"],
        )

    # Evaluate individual scores
    individual_scores: dict[str, float] = {}
    individual_proba: dict[str, np.ndarray] = {}
    for cid, model in prob_models.items():
        proba = model.predict_proba(X_val)
        individual_proba[cid] = _extract_positive_class(proba)
        preds = (individual_proba[cid] >= 0.5).astype(int)
        individual_scores[cid] = _compute_metric(y_val, preds, metric)

    # Select top candidates by score (up to max_ensemble_size)
    sorted_ids = sorted(individual_scores, key=lambda k: individual_scores[k], reverse=True)
    member_ids = sorted_ids[: ens_config.max_ensemble_size]

    best_report: EnsembleReport | None = None
    best_ensemble_score: float = -1.0

    for strategy_name in ens_config.enabled_strategies:
        if strategy_name not in _STRATEGY_REGISTRY:
            logger.warning(f"Unknown ensemble strategy '{strategy_name}'; skipping.")
            continue

        strategy = _STRATEGY_REGISTRY[strategy_name]()

        # Build predictions list and weights
        member_proba = [individual_proba[cid] for cid in member_ids]
        weights = [individual_scores[cid] for cid in member_ids]

        combined = strategy.combine(member_proba, weights=weights)
        ensemble_preds = (combined >= 0.5).astype(int)
        ensemble_score = _compute_metric(y_val, ensemble_preds, metric)

        selected = strategy.evaluate_gain(individual_scores, ensemble_score, config)
        report = strategy.to_report(member_ids, individual_scores, ensemble_score, selected)
        report.gain_threshold = ens_config.marginal_gain_threshold

        if selected and ensemble_score > best_ensemble_score:
            best_ensemble_score = ensemble_score
            best_report = report

        # If not selected but no best yet, keep as rejection baseline
        if best_report is None:
            best_report = report

    if best_report is None:
        return EnsembleReport(
            ensemble_selected=False,
            rejection_reason="No ensemble strategies produced a result.",
        )

    return best_report


def _extract_positive_class(proba: np.ndarray) -> np.ndarray:
    """Extract positive-class probabilities."""
    if proba.ndim == 1:
        return proba
    if proba.shape[1] >= 2:
        return proba[:, 1]
    return proba.ravel()


def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    raise ValueError(f"Unknown metric: {metric}")
