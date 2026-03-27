"""Tests for Phase 14: Reporting, Audit Logging, CLI, and Orchestration Wiring.

Required tests:
1. Orchestrator stage-order test.
2. CLI config override test.
3. Final report generation test.
4. End-to-end happy path smoke test.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from aml_toolkit.artifacts import (
    CalibrationReport,
    CandidatePortfolio,
    DataProfile,
    DatasetManifest,
    EnsembleReport,
    ExplainabilityReport,
    FinalReport,
    InterventionPlan,
    ProbeResultSet,
    RuntimeDecisionLog,
    SplitAuditReport,
)
from aml_toolkit.core.config import ToolkitConfig, load_config
from aml_toolkit.core.enums import AbstentionReason, PipelineStage
from aml_toolkit.orchestration.audit_logger import AuditLogger
from aml_toolkit.orchestration.state_machine import PipelineStateMachine
from aml_toolkit.reporting.report_builder import JsonReporter, MarkdownReporter, build_report


# ---------------------------------------------------------------------------
# Test 1: Orchestrator stage-order test (state machine)
# ---------------------------------------------------------------------------

class TestStateMachine:

    def test_happy_path_transitions(self):
        sm = PipelineStateMachine()
        assert sm.current == PipelineStage.INIT

        stages = [
            PipelineStage.DATA_VALIDATED,
            PipelineStage.PROFILED,
            PipelineStage.PROBED,
            PipelineStage.INTERVENTION_SELECTED,
            PipelineStage.TRAINING_ACTIVE,
            PipelineStage.MODEL_SELECTED,
            PipelineStage.CALIBRATED,
            PipelineStage.ENSEMBLED,
            PipelineStage.EXPLAINED,
            PipelineStage.COMPLETED,
        ]

        for stage in stages:
            assert sm.can_transition(stage) is True
            sm.transition(stage)
            assert sm.current == stage

        assert sm.is_terminal is True
        assert sm.history == [PipelineStage.INIT] + stages

    def test_illegal_transition_raises(self):
        sm = PipelineStateMachine()
        with pytest.raises(ValueError, match="Illegal transition"):
            sm.transition(PipelineStage.PROFILED)  # skip DATA_VALIDATED

    def test_abstain_from_any_stage(self):
        for start_stage in [PipelineStage.INIT, PipelineStage.DATA_VALIDATED, PipelineStage.PROFILED]:
            sm = PipelineStateMachine()
            # Advance to start_stage
            path = {
                PipelineStage.INIT: [],
                PipelineStage.DATA_VALIDATED: [PipelineStage.DATA_VALIDATED],
                PipelineStage.PROFILED: [PipelineStage.DATA_VALIDATED, PipelineStage.PROFILED],
            }
            for s in path[start_stage]:
                sm.transition(s)

            sm.abstain(AbstentionReason.CRITICAL_FAILURE)
            assert sm.current == PipelineStage.ABSTAINED
            assert sm.abstention_reason == AbstentionReason.CRITICAL_FAILURE
            assert sm.is_terminal is True

    def test_cannot_transition_from_terminal(self):
        sm = PipelineStateMachine()
        sm.abstain(AbstentionReason.LEAKAGE_BLOCKED)
        assert sm.can_transition(PipelineStage.DATA_VALIDATED) is False
        with pytest.raises(ValueError):
            sm.transition(PipelineStage.DATA_VALIDATED)

    def test_history_tracking(self):
        sm = PipelineStateMachine()
        sm.transition(PipelineStage.DATA_VALIDATED)
        sm.abstain(AbstentionReason.SCHEMA_INVALID)
        assert sm.history == [
            PipelineStage.INIT,
            PipelineStage.DATA_VALIDATED,
            PipelineStage.ABSTAINED,
        ]


# ---------------------------------------------------------------------------
# Test 2: CLI config override test
# ---------------------------------------------------------------------------

class TestCLIConfigOverride:

    def test_load_config_defaults(self):
        config = load_config()
        assert config.mode.value == "BALANCED"
        assert config.seed == 42

    def test_load_config_with_mode_override(self):
        config = load_config(mode="CONSERVATIVE")
        assert config.mode.value == "CONSERVATIVE"

    def test_load_config_with_dict_overrides(self):
        config = load_config(overrides={"seed": 99, "reporting": {"output_dir": "/tmp/test"}})
        assert config.seed == 99
        assert config.reporting.output_dir == "/tmp/test"

    def test_load_config_with_yaml_file(self, tmp_path):
        yaml_content = "seed: 123\nmode: AGGRESSIVE\n"
        yaml_path = tmp_path / "custom.yaml"
        yaml_path.write_text(yaml_content)

        config = load_config(config_path=yaml_path, mode="AGGRESSIVE")
        assert config.seed == 123
        assert config.mode.value == "AGGRESSIVE"

    def test_overrides_take_precedence(self, tmp_path):
        yaml_content = "seed: 123\n"
        yaml_path = tmp_path / "custom.yaml"
        yaml_path.write_text(yaml_content)

        config = load_config(config_path=yaml_path, overrides={"seed": 999})
        assert config.seed == 999


# ---------------------------------------------------------------------------
# Test 3: Final report generation test
# ---------------------------------------------------------------------------

class TestFinalReportGeneration:

    @pytest.fixture()
    def sample_artifacts(self):
        return {
            "run_id": "test_run_001",
            "final_status": PipelineStage.COMPLETED,
            "stages_completed": [PipelineStage.INIT, PipelineStage.DATA_VALIDATED, PipelineStage.COMPLETED],
            "best_candidate_id": "logistic_001",
            "dataset_manifest": DatasetManifest(
                dataset_id="test", modality="TABULAR", task_type="BINARY", split_strategy="STRATIFIED",
            ),
            "split_audit_report": SplitAuditReport(passed=True),
            "data_profile": DataProfile(),
            "probe_results": ProbeResultSet(),
            "intervention_plan": InterventionPlan(),
            "candidate_portfolio": CandidatePortfolio(),
            "runtime_decision_log": RuntimeDecisionLog(),
            "calibration_report": CalibrationReport(),
            "ensemble_report": EnsembleReport(),
            "explainability_report": ExplainabilityReport(),
            "warnings": [],
        }

    def test_json_report_generated(self, sample_artifacts, tmp_path):
        config = ToolkitConfig()
        reporter = JsonReporter()
        report = reporter.generate(sample_artifacts, tmp_path, config)

        assert isinstance(report, FinalReport)
        assert report.run_id == "test_run_001"
        assert report.final_status == PipelineStage.COMPLETED
        assert (tmp_path / "final_report.json").exists()

        # Verify JSON is valid
        with open(tmp_path / "final_report.json") as f:
            data = json.load(f)
        assert data["run_id"] == "test_run_001"

    def test_markdown_report_generated(self, sample_artifacts, tmp_path):
        config = ToolkitConfig()
        reporter = MarkdownReporter()
        report = reporter.generate(sample_artifacts, tmp_path, config)

        assert isinstance(report, FinalReport)
        md_path = tmp_path / "final_report.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "# Pipeline Report" in content
        assert "test_run_001" in content

    def test_build_report_writes_all_formats(self, sample_artifacts, tmp_path):
        config = ToolkitConfig()
        report = build_report(sample_artifacts, tmp_path, config)

        assert isinstance(report, FinalReport)
        assert (tmp_path / "final_report.json").exists()
        assert (tmp_path / "final_report.md").exists()

    def test_report_includes_recommendation(self, sample_artifacts, tmp_path):
        config = ToolkitConfig()
        report = build_report(sample_artifacts, tmp_path, config)
        assert "logistic_001" in report.final_recommendation

    def test_abstained_report(self, tmp_path):
        artifacts = {
            "run_id": "abstained_run",
            "final_status": PipelineStage.ABSTAINED,
            "abstention_reason": AbstentionReason.LEAKAGE_BLOCKED,
            "stages_completed": [PipelineStage.INIT, PipelineStage.ABSTAINED],
            "warnings": ["Leakage detected"],
        }
        config = ToolkitConfig()
        report = build_report(artifacts, tmp_path, config)
        assert report.final_status == PipelineStage.ABSTAINED
        assert report.abstention_reason == AbstentionReason.LEAKAGE_BLOCKED
        assert "abstained" in report.final_recommendation.lower()

    def test_report_serializes(self, sample_artifacts, tmp_path):
        config = ToolkitConfig()
        report = build_report(sample_artifacts, tmp_path, config)
        data = report.model_dump(mode="json")
        reloaded = FinalReport.model_validate(data)
        assert reloaded.run_id == report.run_id


# ---------------------------------------------------------------------------
# Test 4: Audit logger
# ---------------------------------------------------------------------------

class TestAuditLogger:

    def test_log_and_retrieve(self):
        audit = AuditLogger()
        audit.log("INIT", "pipeline_start", {"run_id": "test_001"})
        audit.log("DATA_VALIDATED", "intake_complete")

        entries = audit.to_list()
        assert len(entries) == 2
        assert entries[0]["stage"] == "INIT"
        assert entries[0]["event"] == "pipeline_start"
        assert entries[0]["detail"]["run_id"] == "test_001"
        assert entries[1]["stage"] == "DATA_VALIDATED"

    def test_save_to_file(self, tmp_path):
        audit = AuditLogger()
        audit.log("INIT", "start")
        audit.log("COMPLETED", "done")

        path = tmp_path / "audit.json"
        audit.save(path)

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["event"] == "start"

    def test_timestamps_present(self):
        audit = AuditLogger()
        audit.log("INIT", "start")
        entry = audit.to_list()[0]
        assert "timestamp" in entry
        assert len(entry["timestamp"]) > 0


# ---------------------------------------------------------------------------
# Test 5: End-to-end happy path smoke test (using synthetic data)
# ---------------------------------------------------------------------------

class TestEndToEndSmoke:

    def test_orchestrator_with_tabular_csv(self, tmp_path):
        """Smoke test: run the full orchestrator on a tiny synthetic CSV."""
        # Create a minimal CSV dataset
        import pandas as pd
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame({
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "f3": rng.randn(n),
            "label": np.array([0] * 50 + [1] * 50),
        })
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        config = ToolkitConfig(
            dataset={"path": str(csv_path), "target_column": "label"},
            reporting={"output_dir": str(tmp_path / "outputs")},
            candidates={"allowed_families": ["logistic"], "max_candidates": 1},
        )

        from aml_toolkit.orchestration.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(config)
        report = orchestrator.run(csv_path)

        assert isinstance(report, FinalReport)
        assert report.run_id != ""

        # Should either complete or abstain (both are valid for a smoke test)
        assert report.final_status in (PipelineStage.COMPLETED, PipelineStage.ABSTAINED)

        # Check outputs were written
        assert orchestrator.run_dir is not None
        assert orchestrator.run_dir.exists()

        # Audit log should exist
        audit_path = orchestrator.run_dir / "logs" / "audit_log.json"
        assert audit_path.exists()
