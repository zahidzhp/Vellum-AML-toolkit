"""Pipeline orchestrator: wires all stages in the correct order."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aml_toolkit.artifacts import FinalReport
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import AbstentionReason, ModalityType, PipelineStage
from aml_toolkit.core.exceptions import SplitIntegrityError
from aml_toolkit.core.paths import create_run_directory, generate_run_id
from aml_toolkit.orchestration.audit_logger import AuditLogger
from aml_toolkit.orchestration.state_machine import PipelineStateMachine
from aml_toolkit.reporting.report_builder import build_report

logger = logging.getLogger("aml_toolkit")


class PipelineOrchestrator:
    """Orchestrates the full pipeline, enforcing stage order via state machine.

    Each stage method advances the state machine, collects artifacts,
    and logs audit events. On failure, the orchestrator transitions to
    ABSTAINED with the appropriate reason.
    """

    def __init__(self, config: ToolkitConfig) -> None:
        self.config = config
        self.state = PipelineStateMachine()
        self.audit = AuditLogger()
        self.artifacts: dict[str, Any] = {}
        self._run_id: str = ""
        self._run_dir: Path | None = None

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    def run(self, dataset_path: str | Path) -> FinalReport:
        """Execute the full pipeline end-to-end.

        Args:
            dataset_path: Path to the input dataset.

        Returns:
            FinalReport summarizing the run.
        """
        self._run_id = generate_run_id(
            config_repr=self.config.model_dump_json(),
            dataset_path=str(dataset_path),
        )
        output_base = Path(self.config.reporting.output_dir)
        self._run_dir = create_run_directory(self._run_id, output_base)
        self.artifacts["run_id"] = self._run_id

        self.audit.log("INIT", "pipeline_start", {"run_id": self._run_id, "dataset": str(dataset_path)})

        try:
            self._stage_intake(dataset_path)
            self._stage_profiling()
            self._stage_probes()
            self._stage_interventions()
            self._stage_training()
            self._stage_calibration()
            self._stage_ensemble()
            self._stage_explainability()
            self._finalize()
        except Exception as e:
            if not self.state.is_terminal:
                logger.error(f"Pipeline failed at {self.state.current.value}: {e}")
                self.audit.log(self.state.current.value, "critical_failure", {"error": str(e)})
                self.state.abstain(AbstentionReason.CRITICAL_FAILURE)
                self.artifacts["final_status"] = PipelineStage.ABSTAINED
                self.artifacts["abstention_reason"] = AbstentionReason.CRITICAL_FAILURE
                self.artifacts["warnings"] = self.artifacts.get("warnings", []) + [str(e)]

        # Build report
        self.artifacts["stages_completed"] = self.state.history
        report = build_report(self.artifacts, self._run_dir / "reporting", self.config)

        # Save audit log
        self.audit.save(self._run_dir / "logs" / "audit_log.json")

        return report

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage_intake(self, dataset_path: str | Path) -> None:
        from aml_toolkit.intake.dataset_intake_manager import run_intake

        self.audit.log("INIT", "intake_start")

        # Set dataset path in config for intake
        self.config.dataset.path = str(dataset_path)
        intake_result = run_intake(self.config)

        self.artifacts["dataset_manifest"] = intake_result.manifest
        self.artifacts["intake_data"] = intake_result.data
        self.artifacts["split_result"] = intake_result.split_result
        self.artifacts["modality"] = intake_result.manifest.modality
        self.artifacts["task_type"] = intake_result.manifest.task_type

        # Extract numpy arrays from intake data for downstream stages
        data = intake_result.data
        split = intake_result.split_result
        if isinstance(data, dict) and "df" in data:
            df = data["df"]
            features = data["features"]
            target = data["target"]
            X = df[features].values
            y = df[target].values
            self.artifacts["X_train"] = X[split.train_indices]
            self.artifacts["y_train"] = y[split.train_indices]
            self.artifacts["X_val"] = X[split.val_indices]
            self.artifacts["y_val"] = y[split.val_indices]
            self.artifacts["X_test"] = X[split.test_indices]
            self.artifacts["y_test"] = y[split.test_indices]
            self.artifacts["dataframe"] = df
        elif isinstance(data, dict) and "embeddings" in data:
            emb = data["embeddings"]
            y = data["labels"]
            self.artifacts["X_train"] = emb[split.train_indices]
            self.artifacts["y_train"] = y[split.train_indices]
            self.artifacts["X_val"] = emb[split.val_indices]
            self.artifacts["y_val"] = y[split.val_indices]
            self.artifacts["X_test"] = emb[split.test_indices]
            self.artifacts["y_test"] = y[split.test_indices]
        elif isinstance(data, dict) and "image_paths" in data:
            from aml_toolkit.utils.image_feature_extractor import ImageFeatureExtractor

            image_paths = data["image_paths"]
            y = data["labels"]

            # Store raw image paths for CNN/ViT adapters
            self.artifacts["image_paths"] = image_paths
            self.artifacts["image_paths_train"] = [image_paths[i] for i in split.train_indices]
            self.artifacts["image_paths_val"] = [image_paths[i] for i in split.val_indices]
            self.artifacts["image_paths_test"] = [image_paths[i] for i in split.test_indices]

            # Extract embeddings for embedding-based stages (probes, calibration, etc.)
            backbone = self.config.candidates.feature_extractor_backbone
            extractor = ImageFeatureExtractor(
                backbone=backbone,
                gpu_enabled=self.config.compute.gpu_enabled,
            )
            embeddings = extractor.extract(image_paths)

            self.artifacts["X_train"] = embeddings[split.train_indices]
            self.artifacts["y_train"] = y[split.train_indices]
            self.artifacts["X_val"] = embeddings[split.val_indices]
            self.artifacts["y_val"] = y[split.val_indices]
            self.artifacts["X_test"] = embeddings[split.test_indices]
            self.artifacts["y_test"] = y[split.test_indices]

        self.state.transition(PipelineStage.DATA_VALIDATED)
        self.audit.log("DATA_VALIDATED", "intake_complete", {
            "modality": intake_result.manifest.modality.value,
            "task_type": intake_result.manifest.task_type.value,
            "n_train": intake_result.manifest.train_size,
        })

        # Run split audit
        from aml_toolkit.audit.split_auditor import run_split_audit

        audit_report = run_split_audit(
            data=intake_result.data,
            manifest=intake_result.manifest,
            split=intake_result.split_result,
            config=self.config,
        )
        self.artifacts["split_audit_report"] = audit_report

        if not audit_report.passed:
            self.audit.log("DATA_VALIDATED", "audit_failed", {"issues": audit_report.blocking_issues})
            self.state.abstain(AbstentionReason.LEAKAGE_BLOCKED)
            self.artifacts["final_status"] = PipelineStage.ABSTAINED
            self.artifacts["abstention_reason"] = AbstentionReason.LEAKAGE_BLOCKED
            raise SplitIntegrityError("Split audit failed: " + str(audit_report.blocking_issues))

        self.audit.log("DATA_VALIDATED", "audit_passed")

    def _stage_profiling(self) -> None:
        from aml_toolkit.profiling.profiler_engine import run_profiling

        self.audit.log("DATA_VALIDATED", "profiling_start")
        profile = run_profiling(
            data=self.artifacts["intake_data"],
            manifest=self.artifacts["dataset_manifest"],
            split=self.artifacts["split_result"],
            config=self.config,
        )
        self.artifacts["data_profile"] = profile
        self.state.transition(PipelineStage.PROFILED)
        self.audit.log("PROFILED", "profiling_complete")

    def _stage_probes(self) -> None:
        from aml_toolkit.probes.probe_engine import run_probes

        self.audit.log("PROFILED", "probes_start")
        probe_results = run_probes(
            data=self.artifacts["intake_data"],
            manifest=self.artifacts["dataset_manifest"],
            split=self.artifacts["split_result"],
            config=self.config,
        )
        self.artifacts["probe_results"] = probe_results
        self.state.transition(PipelineStage.PROBED)
        self.audit.log("PROBED", "probes_complete")

    def _stage_interventions(self) -> None:
        from aml_toolkit.interventions.intervention_planner import plan_interventions

        self.audit.log("PROBED", "interventions_start")
        plan = plan_interventions(
            profile=self.artifacts["data_profile"],
            audit_report=self.artifacts["split_audit_report"],
            probe_results=self.artifacts.get("probe_results"),
            config=self.config,
        )
        self.artifacts["intervention_plan"] = plan
        self.state.transition(PipelineStage.INTERVENTION_SELECTED)
        self.audit.log("INTERVENTION_SELECTED", "interventions_complete")

    def _stage_training(self) -> None:
        from aml_toolkit.models.registry import build_candidate_portfolio
        from aml_toolkit.runtime.training_executor import run_training

        self.audit.log("INTERVENTION_SELECTED", "training_start")

        portfolio = build_candidate_portfolio(
            modality=self.artifacts["modality"],
            config=self.config,
            intervention_plan=self.artifacts.get("intervention_plan"),
        )
        self.artifacts["candidate_portfolio"] = portfolio

        # Pass raw image paths for CNN/ViT adapters if modality is IMAGE
        raw_data = None
        if self.artifacts.get("modality") == ModalityType.IMAGE and "image_paths_train" in self.artifacts:
            raw_data = {
                "image_paths_train": self.artifacts["image_paths_train"],
                "image_paths_val": self.artifacts["image_paths_val"],
                "y_train": self.artifacts["y_train"],
                "y_val": self.artifacts["y_val"],
            }

        exec_result = run_training(
            self.artifacts["X_train"],
            self.artifacts["y_train"],
            self.artifacts["X_val"],
            self.artifacts["y_val"],
            portfolio=portfolio,
            audit_report=self.artifacts["split_audit_report"],
            config=self.config,
            intervention_plan=self.artifacts.get("intervention_plan"),
            raw_data=raw_data,
        )
        self.artifacts["execution_result"] = exec_result
        self.artifacts["trained_models"] = exec_result.trained_models

        self.state.transition(PipelineStage.TRAINING_ACTIVE)

        # Runtime decisions
        from aml_toolkit.runtime.decision_engine import RuntimeDecisionEngine

        engine = RuntimeDecisionEngine(self.config, portfolio.warmup_rules)
        from aml_toolkit.models.registry import create_default_registry
        registry = create_default_registry()

        for trace in exec_result.traces:
            if trace.status == "completed":
                meta = registry.get_metadata(trace.model_family)
                engine.evaluate_from_trace(
                    trace.candidate_id, trace.model_family, meta.is_neural,
                    trace.training_trace, trace.metrics,
                )

        self.artifacts["runtime_decision_log"] = engine.decision_log

        # Select best candidate
        if exec_result.trained_models:
            best_id = max(
                exec_result.trained_models.keys(),
                key=lambda cid: next(
                    (t.metrics.get("macro_f1", 0) for t in exec_result.traces if t.candidate_id == cid), 0
                ),
            )
            self.artifacts["best_candidate_id"] = best_id
            self._generate_training_plots(exec_result, best_id)
            self.state.transition(PipelineStage.MODEL_SELECTED)
            self.audit.log("MODEL_SELECTED", "training_complete", {"best": best_id})
        else:
            self.audit.log("TRAINING_ACTIVE", "no_models_trained")
            self.state.abstain(AbstentionReason.NO_ROBUST_MODEL)
            self.artifacts["final_status"] = PipelineStage.ABSTAINED
            self.artifacts["abstention_reason"] = AbstentionReason.NO_ROBUST_MODEL
            raise RuntimeError("No models completed training.")

    def _get_model_x_val(self, model: Any) -> Any:
        """Return the appropriate X_val for a model — image paths for CNN/ViT, embeddings otherwise."""
        if (
            hasattr(model, "get_model_family")
            and model.get_model_family() in ("cnn", "vit")
            and "image_paths_val" in self.artifacts
        ):
            return self.artifacts["image_paths_val"]
        return self.artifacts["X_val"]

    def _stage_calibration(self) -> None:
        from aml_toolkit.calibration.calibration_manager import run_calibration

        self.audit.log("MODEL_SELECTED", "calibration_start")

        # Split models by X_val type to avoid passing wrong data
        embedding_models = {}
        image_models = {}
        for cid, model in self.artifacts["trained_models"].items():
            if hasattr(model, "get_model_family") and model.get_model_family() in ("cnn", "vit") and "image_paths_val" in self.artifacts:
                image_models[cid] = model
            else:
                embedding_models[cid] = model

        # Calibrate embedding-based models
        cal_report = run_calibration(
            embedding_models, self.artifacts["X_val"], self.artifacts["y_val"], self.config,
        ) if embedding_models else None

        # Calibrate image-native models
        if image_models and "image_paths_val" in self.artifacts:
            img_cal = run_calibration(
                image_models, self.artifacts["image_paths_val"], self.artifacts["y_val"], self.config,
            )
            if cal_report is not None:
                cal_report.results.extend(img_cal.results)
                cal_report.warnings.extend(img_cal.warnings)
            else:
                cal_report = img_cal

        if cal_report is None:
            from aml_toolkit.artifacts import CalibrationReport
            cal_report = CalibrationReport()

        self.artifacts["calibration_report"] = cal_report
        self._generate_calibration_plots(cal_report)
        self.state.transition(PipelineStage.CALIBRATED)
        self.audit.log("CALIBRATED", "calibration_complete")

    def _stage_ensemble(self) -> None:
        from aml_toolkit.ensemble.ensemble_manager import run_ensemble

        self.audit.log("CALIBRATED", "ensemble_start")

        # Separate embedding-compatible models from image-native models
        embedding_models = {}
        image_native_models = {}
        for cid, m in self.artifacts["trained_models"].items():
            if hasattr(m, "get_model_family") and m.get_model_family() in ("cnn", "vit"):
                image_native_models[cid] = m
            else:
                embedding_models[cid] = m

        if embedding_models:
            # Ensemble embedding-compatible models using embedding X_val
            ens_report = run_ensemble(
                embedding_models,
                self.artifacts["X_val"],
                self.artifacts["y_val"],
                self.config,
            )
        elif image_native_models and "image_paths_val" in self.artifacts:
            # Only image-native models — ensemble using image paths
            ens_report = run_ensemble(
                image_native_models,
                self.artifacts["image_paths_val"],
                self.artifacts["y_val"],
                self.config,
            )
        else:
            # Fallback: run on all models with embedding X_val
            ens_report = run_ensemble(
                self.artifacts["trained_models"],
                self.artifacts["X_val"],
                self.artifacts["y_val"],
                self.config,
            )
        self.artifacts["ensemble_report"] = ens_report
        self.state.transition(PipelineStage.ENSEMBLED)
        self.audit.log("ENSEMBLED", "ensemble_complete", {"selected": ens_report.ensemble_selected})

    def _stage_explainability(self) -> None:
        from aml_toolkit.explainability.explainability_manager import run_explainability

        self.audit.log("ENSEMBLED", "explainability_start")
        exp_report = run_explainability(
            self.artifacts["trained_models"],
            self.artifacts["X_val"],
            self.artifacts["y_val"],
            self.config,
            self._run_dir / "explainability",
            modality=self.artifacts.get("modality", ModalityType.TABULAR),
            image_paths_val=self.artifacts.get("image_paths_val"),
        )
        self.artifacts["explainability_report"] = exp_report
        self.state.transition(PipelineStage.EXPLAINED)
        self.audit.log("EXPLAINED", "explainability_complete")

    def _generate_training_plots(self, exec_result: Any, best_id: str) -> None:
        """Generate learning curves, classification report, ROC, and PR plots after training."""
        if self._run_dir is None:
            return
        from aml_toolkit.reporting.plot_utils import (
            plot_classification_report,
            plot_learning_curves,
            plot_precision_recall_curve,
            plot_roc_curve,
        )

        plots_dir = self._run_dir / "plots"
        plot_paths: dict[str, str] = self.artifacts.setdefault("plot_paths", {})

        for trace in exec_result.traces:
            if trace.status == "completed" and trace.training_trace:
                path = plot_learning_curves(
                    trace.training_trace,
                    plots_dir / f"learning_curves_{trace.candidate_id}.png",
                )
                if path:
                    plot_paths[f"learning_curves_{trace.candidate_id}"] = path

        best_model = self.artifacts["trained_models"].get(best_id)
        if best_model is not None:
            X_val = self._get_model_x_val(best_model)
            y_val = self.artifacts["y_val"]
            try:
                y_pred = best_model.predict(X_val)
                path = plot_classification_report(y_val, y_pred, None, plots_dir / "classification_report.png")
                if path:
                    plot_paths["classification_report"] = path

                if best_model.is_probabilistic():
                    proba = best_model.predict_proba(X_val)
                    if proba is not None and proba.ndim == 2 and proba.shape[1] == 2:
                        y_score = proba[:, 1]
                        path = plot_roc_curve(y_val, y_score, plots_dir / "roc_curve.png")
                        if path:
                            plot_paths["roc_curve"] = path
                        path = plot_precision_recall_curve(y_val, y_score, plots_dir / "precision_recall_curve.png")
                        if path:
                            plot_paths["precision_recall_curve"] = path
            except Exception as e:
                logger.warning(f"Training plot generation failed: {e}")

    def _generate_calibration_plots(self, cal_report: Any) -> None:
        """Generate calibration reliability diagrams and threshold sweep curves."""
        if self._run_dir is None or not cal_report.plot_data:
            return
        from aml_toolkit.reporting.plot_utils import plot_calibration_diagram, plot_threshold_vs_metric

        plots_dir = self._run_dir / "plots"
        plot_paths: dict[str, str] = self.artifacts.setdefault("plot_paths", {})
        y_val = self.artifacts["y_val"]

        for candidate_id, data in cal_report.plot_data.items():
            proba_before = data.get("proba_before")
            proba_after = data.get("proba_after")
            if proba_before is None or proba_after is None:
                continue
            path = plot_calibration_diagram(
                y_val, proba_before, proba_after, 10,
                plots_dir / f"calibration_{candidate_id}.png",
            )
            if path:
                plot_paths[f"calibration_{candidate_id}"] = path
            path = plot_threshold_vs_metric(
                y_val, proba_after, "f1",
                plots_dir / f"threshold_{candidate_id}.png",
            )
            if path:
                plot_paths[f"threshold_{candidate_id}"] = path

    def _finalize(self) -> None:
        self.artifacts["final_status"] = PipelineStage.COMPLETED
        self.state.transition(PipelineStage.COMPLETED)
        self.audit.log("COMPLETED", "pipeline_complete")
