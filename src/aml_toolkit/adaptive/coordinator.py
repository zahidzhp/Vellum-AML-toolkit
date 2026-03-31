"""AdaptiveIntelligenceCoordinator — single entry point for all V2 features.

The orchestrator calls this coordinator instead of invoking each V2 module
independently. This enforces correct data flow:
    1. Pre-training: RunHistory → MetaPolicy → candidate ordering + budgets
    2. Post-calibration: CalibrationReport.plot_data → Uncertainty → Ensemble
    3. Post-explainability: ExperimentPlanner → next-run proposals
    4. Finalize: save RunHistoryRecord (non-blocking)

All V2 features degrade gracefully — if a feature fails, V1 behavior
is preserved (no crash, no pipeline abort).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aml_toolkit.artifacts.experiment_plan import ExperimentPlan
from aml_toolkit.artifacts.meta_policy_recommendation import MetaPolicyRecommendation
from aml_toolkit.artifacts.run_history import DatasetSignature, RunHistoryRecord
from aml_toolkit.artifacts.uncertainty_report import UncertaintyReport
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.history.dataset_signature_builder import build_dataset_signature
from aml_toolkit.history.run_history_store import RunHistoryStore
from aml_toolkit.meta_policy.meta_policy_engine import MetaPolicyEngine
from aml_toolkit.planning.experiment_planner import ExperimentPlanner
from aml_toolkit.uncertainty.estimator import UncertaintyEstimator

logger = logging.getLogger("aml_toolkit")


@dataclass
class PreTrainingRecommendation:
    """Output of pre_training_recommendations()."""

    candidate_order: list[str] = field(default_factory=list)
    compute_budget_fractions: dict[str, float] = field(default_factory=dict)
    meta_policy_recommendation: MetaPolicyRecommendation | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class PostCalibrationAnalysis:
    """Output of post_calibration_analysis()."""

    uncertainty_reports: dict[str, UncertaintyReport] = field(default_factory=dict)
    abstention_triggered: bool = False
    abstention_candidate_id: str = ""
    selected_ensemble_members: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class AdaptiveIntelligenceCoordinator:
    """Single entry point for all V2 adaptive intelligence features.

    Usage in orchestrator:
        coordinator = AdaptiveIntelligenceCoordinator(config)

        # Before training:
        pre_rec = coordinator.pre_training_recommendations(manifest, profile)

        # After calibration:
        post_cal = coordinator.post_calibration_analysis(cal_report, models, X_val, y_val)

        # After explainability:
        exp_plan = coordinator.generate_experiment_plan(artifacts, history)

        # In finalize:
        coordinator.save_run_record(artifacts, config)
    """

    def __init__(self, config: ToolkitConfig):
        self.config = config
        self._history_store: RunHistoryStore | None = None
        if config.advanced.run_history.enabled:
            self._history_store = RunHistoryStore(config.advanced.run_history.store_path)

    # ------------------------------------------------------------------
    # Phase hook 1: before training
    # ------------------------------------------------------------------

    def pre_training_recommendations(
        self,
        dataset_manifest: Any,
        data_profile: Any,
    ) -> PreTrainingRecommendation:
        """Recommend candidate ordering and compute budgets before training.

        Uses: RunHistory + MetaPolicy.
        Guaranteed non-raising — returns empty recommendation on any failure.
        """
        try:
            return self._pre_training(dataset_manifest, data_profile)
        except Exception as e:
            logger.warning(f"AdaptiveCoordinator.pre_training_recommendations failed: {e}")
            return PreTrainingRecommendation(notes=[f"pre_training error: {e}"])

    def _pre_training(
        self, dataset_manifest: Any, data_profile: Any
    ) -> PreTrainingRecommendation:
        result = PreTrainingRecommendation()

        if not self.config.advanced.meta_policy.enabled:
            return result

        # Build dataset signature
        sig = build_dataset_signature(dataset_manifest, data_profile)

        # Load run history
        history: list[RunHistoryRecord] = []
        if self._history_store:
            history = self._history_store.find_similar(
                sig,
                top_k=20,
                recency_decay=self.config.advanced.meta_policy.recency_decay,
            )

        # Get candidate IDs from manifest (or return empty for now)
        candidate_ids = self._extract_candidate_ids(dataset_manifest)
        if not candidate_ids:
            return result

        # Run meta-policy
        engine = MetaPolicyEngine(self.config.advanced.meta_policy)
        rec = engine.recommend(candidate_ids, sig, history, self.config)

        result.candidate_order = rec.recommended_order
        result.compute_budget_fractions = rec.compute_budget_fractions
        result.meta_policy_recommendation = rec
        result.notes.append(
            f"MetaPolicy: {len(history)} history records used, "
            f"{len(rec.recommended_order)} candidates reordered."
        )
        return result

    # ------------------------------------------------------------------
    # Phase hook 2: after calibration
    # ------------------------------------------------------------------

    def post_calibration_analysis(
        self,
        calibration_report: Any,
        trained_models: dict,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> PostCalibrationAnalysis:
        """Uncertainty estimation and ensemble member selection after calibration.

        Uses calibrated probabilities from calibration_report.plot_data —
        zero additional inference passes.
        """
        try:
            return self._post_calibration(calibration_report, trained_models, X_val, y_val)
        except Exception as e:
            logger.warning(f"AdaptiveCoordinator.post_calibration_analysis failed: {e}")
            return PostCalibrationAnalysis(notes=[f"post_calibration error: {e}"])

    def _post_calibration(
        self,
        cal_report: Any,
        trained_models: dict,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> PostCalibrationAnalysis:
        result = PostCalibrationAnalysis()

        if not self.config.advanced.uncertainty.enabled:
            return result

        # Get calibrated probabilities (zero extra inference!)
        plot_data: dict = {}
        if hasattr(cal_report, "plot_data"):
            plot_data = cal_report.plot_data or {}

        estimator = UncertaintyEstimator(self.config.advanced.uncertainty)

        for candidate_id, model in trained_models.items():
            # Use calibrated proba if available and configured
            if self.config.advanced.uncertainty.use_calibrated_proba and candidate_id in plot_data:
                proba = plot_data[candidate_id].get("proba_after")
            else:
                # Fall back: skip if no proba available
                proba = plot_data.get(candidate_id, {}).get("proba_before")

            if proba is None:
                continue

            proba_2d = self._ensure_2d(np.asarray(proba))
            report = estimator.estimate(candidate_id, proba_2d, y_val)
            result.uncertainty_reports[candidate_id] = report

            if report.abstention_triggered and not result.abstention_triggered:
                result.abstention_triggered = True
                result.abstention_candidate_id = candidate_id
                result.notes.append(
                    f"Uncertainty abstention triggered for {candidate_id}: {report.abstention_reason}"
                )

        if result.uncertainty_reports:
            result.notes.append(
                f"Uncertainty estimated for {len(result.uncertainty_reports)} candidates."
            )

        return result

    # ------------------------------------------------------------------
    # Phase hook 3: experiment planning
    # ------------------------------------------------------------------

    def generate_experiment_plan(
        self,
        full_artifacts: dict,
        history: list[RunHistoryRecord] | None = None,
    ) -> ExperimentPlan:
        """Generate experiment proposals for the next run.

        Uses: ExperimentPlanner (rule engine + optional LLM).
        """
        try:
            return self._generate_plan(full_artifacts, history or [])
        except Exception as e:
            logger.warning(f"AdaptiveCoordinator.generate_experiment_plan failed: {e}")
            return ExperimentPlan(notes=[f"plan generation error: {e}"])

    def _generate_plan(
        self, artifacts: dict, history: list[RunHistoryRecord]
    ) -> ExperimentPlan:
        if not self.config.advanced.agentic_planner.enabled:
            return ExperimentPlan(notes=["Agentic planner disabled."])

        # Build run state from artifacts
        run_state = self._build_run_state(artifacts)

        planner = ExperimentPlanner(self.config.advanced.agentic_planner)
        return planner.plan(run_state, self.config, history)

    # ------------------------------------------------------------------
    # Phase hook 4: save run record
    # ------------------------------------------------------------------

    def save_run_record(
        self,
        artifacts: dict,
        config: ToolkitConfig,
    ) -> None:
        """Save a RunHistoryRecord to the history store (non-blocking)."""
        if not config.advanced.run_history.enabled or self._history_store is None:
            return
        try:
            self._save_record(artifacts, config)
        except Exception as e:
            logger.warning(f"AdaptiveCoordinator.save_run_record failed (non-fatal): {e}")

    def _save_record(self, artifacts: dict, config: ToolkitConfig) -> None:
        sig = build_dataset_signature(
            artifacts.get("dataset_manifest"),
            artifacts.get("data_profile"),
        )
        best_id = artifacts.get("best_candidate_id", "")
        best_family = best_id.rsplit("_", 1)[0] if "_" in best_id else best_id

        # Extract best F1 from candidate portfolio
        best_f1 = 0.0
        portfolio = artifacts.get("candidate_portfolio")
        if portfolio and hasattr(portfolio, "candidates"):
            for c in portfolio.candidates:
                if hasattr(c, "candidate_id") and c.candidate_id == best_id:
                    best_f1 = float(getattr(c, "val_macro_f1", 0.0) or 0.0)

        record = RunHistoryRecord(
            run_id=artifacts.get("run_id", ""),
            dataset_signature=sig,
            best_candidate_id=best_id,
            best_candidate_family=best_family,
            best_macro_f1=best_f1,
            config_mode=config.mode.value,
            stages_completed=[s.value for s in artifacts.get("stages_completed", [])],
            abstained=artifacts.get("final_status", None) is not None
            and str(artifacts.get("final_status", "")).upper() == "ABSTAINED",
        )
        self._history_store.save(record)
        logger.info(f"Run history record saved: {record.run_id}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_candidate_ids(self, dataset_manifest: Any) -> list[str]:
        """Extract candidate IDs from manifest or return empty list."""
        if dataset_manifest is None:
            return []
        if isinstance(dataset_manifest, dict):
            return dataset_manifest.get("candidate_ids", [])
        if hasattr(dataset_manifest, "candidate_ids"):
            return getattr(dataset_manifest, "candidate_ids", [])
        return []

    def _build_run_state(self, artifacts: dict) -> dict:
        """Build a planner-compatible run state from artifacts."""
        state: dict = {}

        manifest = artifacts.get("dataset_manifest")
        if manifest:
            m = manifest.model_dump(mode="json") if hasattr(manifest, "model_dump") else manifest
            state.update({
                "modality": m.get("modality", "TABULAR"),
                "n_samples": m.get("n_samples", 0),
                "n_features": m.get("n_features", 0),
                "n_classes": m.get("n_classes", 2),
            })

        profile = artifacts.get("data_profile")
        if profile:
            p = profile.model_dump(mode="json") if hasattr(profile, "model_dump") else profile
            state.update({
                "imbalance_severity": p.get("imbalance_severity", "mild"),
                "has_label_noise": p.get("has_label_noise", False),
                "has_ood_shift": p.get("has_ood_shift", False),
            })

        cal_report = artifacts.get("calibration_report")
        if cal_report:
            cr = cal_report.model_dump(mode="json") if hasattr(cal_report, "model_dump") else {}
            state["calibration_results"] = cr.get("results", [])

        state["allowed_interventions"] = [
            t for t in (self.config.interventions.allowed_types or [])
        ]

        return state

    def _ensure_2d(self, proba: np.ndarray) -> np.ndarray:
        if proba.ndim == 1:
            return np.stack([1.0 - proba, proba], axis=1)
        return proba
