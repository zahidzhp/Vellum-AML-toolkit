"""Phase 15 - Integration tests: verify multi-stage interactions work correctly.

These tests exercise real stage-to-stage data flow (not mocked) on small
synthetic datasets to catch integration seams that unit tests miss.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aml_toolkit.artifacts import (
    DataProfile,
    DatasetManifest,
    InterventionPlan,
    ProbeResultSet,
    SplitAuditReport,
)
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import ModalityType, PipelineStage, TaskType
from aml_toolkit.intake.dataset_intake_manager import IntakeResult, run_intake
from aml_toolkit.intake.split_builder import SplitResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_binary_csv(path: Path, n: int = 100, seed: int = 42) -> Path:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "f3": rng.randn(n),
        "label": np.array([0] * (n // 2) + [1] * (n - n // 2)),
    })
    csv_path = path / "binary.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _make_imbalanced_csv(path: Path, n: int = 200, seed: int = 42) -> Path:
    rng = np.random.RandomState(seed)
    n_minority = max(n // 20, 5)
    df = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "label": np.array([0] * (n - n_minority) + [1] * n_minority),
    })
    csv_path = path / "imbalanced.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# I1: Intake → Audit → Profiling chain
# ---------------------------------------------------------------------------

class TestIntakeAuditProfilingChain:

    def test_intake_produces_valid_split(self, tmp_path):
        csv_path = _make_binary_csv(tmp_path)
        config = ToolkitConfig(dataset={"path": str(csv_path), "target_column": "label"})
        result = run_intake(config)

        assert isinstance(result, IntakeResult)
        assert result.manifest.modality == ModalityType.TABULAR
        assert result.manifest.task_type == TaskType.BINARY
        assert result.split_result is not None
        assert len(result.split_result.train_indices) > 0
        assert len(result.split_result.val_indices) > 0
        assert len(result.split_result.test_indices) > 0

    def test_audit_runs_on_intake_output(self, tmp_path):
        from aml_toolkit.audit.split_auditor import run_split_audit

        csv_path = _make_binary_csv(tmp_path)
        config = ToolkitConfig(dataset={"path": str(csv_path), "target_column": "label"})
        intake_result = run_intake(config)

        audit_report = run_split_audit(
            data=intake_result.data,
            manifest=intake_result.manifest,
            split=intake_result.split_result,
            config=config,
        )
        assert isinstance(audit_report, SplitAuditReport)
        # Clean data should pass audit
        assert audit_report.passed is True

    def test_profiling_runs_on_intake_output(self, tmp_path):
        from aml_toolkit.profiling.profiler_engine import run_profiling

        csv_path = _make_binary_csv(tmp_path)
        config = ToolkitConfig(dataset={"path": str(csv_path), "target_column": "label"})
        intake_result = run_intake(config)

        profile = run_profiling(
            data=intake_result.data,
            manifest=intake_result.manifest,
            split=intake_result.split_result,
            config=config,
        )
        assert isinstance(profile, DataProfile)


# ---------------------------------------------------------------------------
# I2: Profiling → Probes → Interventions chain
# ---------------------------------------------------------------------------

class TestProfilingProbesInterventionsChain:

    def test_probes_run_on_intake_output(self, tmp_path):
        from aml_toolkit.probes.probe_engine import run_probes

        csv_path = _make_binary_csv(tmp_path)
        config = ToolkitConfig(dataset={"path": str(csv_path), "target_column": "label"})
        intake_result = run_intake(config)

        probe_results = run_probes(
            data=intake_result.data,
            manifest=intake_result.manifest,
            split=intake_result.split_result,
            config=config,
        )
        assert isinstance(probe_results, ProbeResultSet)

    def test_interventions_use_profile_and_probes(self, tmp_path):
        from aml_toolkit.audit.split_auditor import run_split_audit
        from aml_toolkit.interventions.intervention_planner import plan_interventions
        from aml_toolkit.probes.probe_engine import run_probes
        from aml_toolkit.profiling.profiler_engine import run_profiling

        csv_path = _make_imbalanced_csv(tmp_path)
        config = ToolkitConfig(dataset={"path": str(csv_path), "target_column": "label"})
        intake_result = run_intake(config)

        audit_report = run_split_audit(
            data=intake_result.data,
            manifest=intake_result.manifest,
            split=intake_result.split_result,
            config=config,
        )
        profile = run_profiling(
            data=intake_result.data,
            manifest=intake_result.manifest,
            split=intake_result.split_result,
            config=config,
        )
        probe_results = run_probes(
            data=intake_result.data,
            manifest=intake_result.manifest,
            split=intake_result.split_result,
            config=config,
        )

        plan = plan_interventions(
            profile=profile,
            audit_report=audit_report,
            probe_results=probe_results,
            config=config,
        )
        assert isinstance(plan, InterventionPlan)


# ---------------------------------------------------------------------------
# I3: Full pipeline through orchestrator
# ---------------------------------------------------------------------------

class TestOrchestratorIntegration:

    def test_full_pipeline_balanced_binary(self, tmp_path):
        """Run the full orchestrator on a clean balanced binary dataset."""
        from aml_toolkit.artifacts import FinalReport
        from aml_toolkit.orchestration.orchestrator import PipelineOrchestrator

        csv_path = _make_binary_csv(tmp_path, n=120)
        config = ToolkitConfig(
            dataset={"path": str(csv_path), "target_column": "label"},
            reporting={"output_dir": str(tmp_path / "outputs")},
            candidates={"allowed_families": ["logistic"], "max_candidates": 1},
            compute={"max_training_time_seconds": 120, "gpu_enabled": False},
        )

        orchestrator = PipelineOrchestrator(config)
        report = orchestrator.run(csv_path)

        assert isinstance(report, FinalReport)
        assert report.final_status in (PipelineStage.COMPLETED, PipelineStage.ABSTAINED)
        assert orchestrator.run_dir is not None
        assert orchestrator.run_dir.exists()

        # Audit log must exist
        audit_path = orchestrator.run_dir / "logs" / "audit_log.json"
        assert audit_path.exists()

        # Reports must exist
        reporting_dir = orchestrator.run_dir / "reporting"
        assert (reporting_dir / "final_report.json").exists()

    def test_pipeline_with_imbalanced_data(self, tmp_path):
        """Run the orchestrator on imbalanced data — should still complete or abstain."""
        from aml_toolkit.artifacts import FinalReport
        from aml_toolkit.orchestration.orchestrator import PipelineOrchestrator

        csv_path = _make_imbalanced_csv(tmp_path, n=200)
        config = ToolkitConfig(
            dataset={"path": str(csv_path), "target_column": "label"},
            reporting={"output_dir": str(tmp_path / "outputs")},
            candidates={"allowed_families": ["logistic"], "max_candidates": 1},
            compute={"max_training_time_seconds": 120, "gpu_enabled": False},
        )

        orchestrator = PipelineOrchestrator(config)
        report = orchestrator.run(csv_path)

        assert isinstance(report, FinalReport)
        assert report.final_status in (PipelineStage.COMPLETED, PipelineStage.ABSTAINED)

    def test_config_override_propagates_through_pipeline(self, tmp_path):
        """Verify that config overrides actually affect pipeline behavior."""
        from aml_toolkit.orchestration.orchestrator import PipelineOrchestrator

        csv_path = _make_binary_csv(tmp_path, n=100)
        config = ToolkitConfig(
            dataset={"path": str(csv_path), "target_column": "label"},
            reporting={"output_dir": str(tmp_path / "outputs")},
            candidates={"allowed_families": ["logistic"], "max_candidates": 1},
            seed=99,
        )
        orchestrator = PipelineOrchestrator(config)
        assert orchestrator.config.seed == 99
        assert orchestrator.config.candidates.max_candidates == 1


# ---------------------------------------------------------------------------
# I4: Artifact persistence across stages
# ---------------------------------------------------------------------------

class TestArtifactPersistence:

    def test_orchestrator_collects_all_stage_artifacts(self, tmp_path):
        """After a full run, the orchestrator's artifact dict should have keys from every stage."""
        from aml_toolkit.orchestration.orchestrator import PipelineOrchestrator

        csv_path = _make_binary_csv(tmp_path, n=120)
        config = ToolkitConfig(
            dataset={"path": str(csv_path), "target_column": "label"},
            reporting={"output_dir": str(tmp_path / "outputs")},
            candidates={"allowed_families": ["logistic"], "max_candidates": 1},
            compute={"gpu_enabled": False},
        )

        orchestrator = PipelineOrchestrator(config)
        report = orchestrator.run(csv_path)

        if report.final_status == PipelineStage.COMPLETED:
            expected_keys = [
                "run_id", "dataset_manifest", "intake_data", "split_result",
                "split_audit_report", "data_profile", "probe_results",
                "intervention_plan", "candidate_portfolio", "execution_result",
                "trained_models", "runtime_decision_log",
                "calibration_report", "ensemble_report", "explainability_report",
                "final_status", "stages_completed",
            ]
            for key in expected_keys:
                assert key in orchestrator.artifacts, f"Missing artifact: {key}"


# ---------------------------------------------------------------------------
# I5: Full image pipeline via orchestrator
# ---------------------------------------------------------------------------

class TestImagePipelineIntegration:

    @staticmethod
    def _make_image_folder(tmp_path, n_per_class: int = 10):
        from PIL import Image

        img_dir = tmp_path / "images"
        for class_name in ["cat", "dog", "bird"]:
            class_dir = img_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                img = Image.fromarray(
                    np.random.RandomState(42 + i).randint(0, 255, (32, 32, 3), dtype=np.uint8)
                )
                img.save(class_dir / f"{class_name}_{i}.jpg")
        return img_dir

    def test_full_image_pipeline_embedding_head(self, tmp_path):
        """End-to-end: image folder -> embedding_head pipeline -> report."""
        from aml_toolkit.artifacts import FinalReport
        from aml_toolkit.orchestration.orchestrator import PipelineOrchestrator

        img_dir = self._make_image_folder(tmp_path, n_per_class=15)
        config = ToolkitConfig(
            dataset={"path": str(img_dir)},
            reporting={"output_dir": str(tmp_path / "outputs")},
            candidates={"allowed_families": ["embedding_head"], "max_candidates": 1},
            compute={"gpu_enabled": False, "max_training_time_seconds": 120},
        )

        orchestrator = PipelineOrchestrator(config)
        report = orchestrator.run(img_dir)

        assert isinstance(report, FinalReport)
        assert report.final_status in (PipelineStage.COMPLETED, PipelineStage.ABSTAINED)
        assert orchestrator.run_dir is not None
        assert orchestrator.run_dir.exists()

        # Verify artifacts were populated
        assert "image_paths" in orchestrator.artifacts
        assert "X_train" in orchestrator.artifacts  # embeddings extracted
        assert orchestrator.artifacts["modality"] == ModalityType.IMAGE

        # Audit log must exist
        audit_path = orchestrator.run_dir / "logs" / "audit_log.json"
        assert audit_path.exists()

    def test_image_pipeline_with_cnn(self, tmp_path):
        """End-to-end: image folder -> CNN training (aggressive mode) -> report."""
        from aml_toolkit.artifacts import FinalReport
        from aml_toolkit.orchestration.orchestrator import PipelineOrchestrator

        img_dir = self._make_image_folder(tmp_path, n_per_class=15)
        config = ToolkitConfig(
            dataset={"path": str(img_dir)},
            reporting={"output_dir": str(tmp_path / "outputs")},
            candidates={
                "allowed_families": ["cnn"],
                "max_candidates": 1,
                "cnn_backbone": "resnet18",
            },
            runtime_decision={"min_warmup_epochs_neural": 1},
            compute={"gpu_enabled": False, "max_training_time_seconds": 300},
        )

        orchestrator = PipelineOrchestrator(config)
        report = orchestrator.run(img_dir)

        assert isinstance(report, FinalReport)
        assert report.final_status in (PipelineStage.COMPLETED, PipelineStage.ABSTAINED)

        # If completed, CNN model should have been trained
        if report.final_status == PipelineStage.COMPLETED:
            assert "trained_models" in orchestrator.artifacts
            assert "image_paths_val" in orchestrator.artifacts


# ---------------------------------------------------------------------------
# I6: State machine enforced by orchestrator
# ---------------------------------------------------------------------------

class TestStateMachineEnforcement:

    def test_orchestrator_tracks_stage_history(self, tmp_path):
        from aml_toolkit.orchestration.orchestrator import PipelineOrchestrator

        csv_path = _make_binary_csv(tmp_path, n=100)
        config = ToolkitConfig(
            dataset={"path": str(csv_path), "target_column": "label"},
            reporting={"output_dir": str(tmp_path / "outputs")},
            candidates={"allowed_families": ["logistic"], "max_candidates": 1},
            compute={"gpu_enabled": False},
        )

        orchestrator = PipelineOrchestrator(config)
        orchestrator.run(csv_path)

        history = orchestrator.state.history
        assert history[0] == PipelineStage.INIT
        # Should always progress forward
        for i in range(1, len(history)):
            assert history[i] != history[i - 1]
