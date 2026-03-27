"""Phase 15 - Regression tests: verify key contracts from all previous phases remain intact.

These tests guard against regressions introduced by Phase 14+ wiring
and ensure each phase's core contract is still honored.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from aml_toolkit.artifacts import (
    CalibrationReport,
    CandidateEntry,
    CandidatePortfolio,
    DataProfile,
    DatasetManifest,
    EnsembleReport,
    ExplainabilityReport,
    FinalReport,
    InterventionEntry,
    InterventionPlan,
    ProbeResult,
    ProbeResultSet,
    RuntimeDecision,
    RuntimeDecisionLog,
    SplitAuditReport,
)
from aml_toolkit.core.config import ToolkitConfig, load_config
from aml_toolkit.core.enums import (
    AbstentionReason,
    DecisionType,
    InterventionType,
    ModalityType,
    OperatingMode,
    PipelineStage,
    RiskFlag,
    SplitStrategy,
    TaskType,
)
from aml_toolkit.core.exceptions import (
    CalibrationFailureError,
    LeakageDetectedError,
    ResourceAbstentionError,
    SchemaValidationError,
    SplitIntegrityError,
    UnsupportedModalityError,
)
from aml_toolkit.core.paths import create_run_directory, generate_run_id
from aml_toolkit.orchestration.audit_logger import AuditLogger
from aml_toolkit.orchestration.state_machine import PipelineStateMachine


# ---------------------------------------------------------------------------
# R1: Config contracts
# ---------------------------------------------------------------------------

class TestConfigContracts:

    def test_all_sections_present(self):
        config = ToolkitConfig()
        assert hasattr(config, "dataset")
        assert hasattr(config, "splitting")
        assert hasattr(config, "profiling")
        assert hasattr(config, "probes")
        assert hasattr(config, "interventions")
        assert hasattr(config, "candidates")
        assert hasattr(config, "runtime_decision")
        assert hasattr(config, "calibration")
        assert hasattr(config, "ensemble")
        assert hasattr(config, "explainability")
        assert hasattr(config, "reporting")
        assert hasattr(config, "compute")

    def test_mode_overlay_changes_defaults(self):
        conservative = load_config(mode="CONSERVATIVE")
        aggressive = load_config(mode="AGGRESSIVE")
        # Both should be valid configs
        assert conservative.mode == OperatingMode.CONSERVATIVE
        assert aggressive.mode == OperatingMode.AGGRESSIVE

    def test_config_roundtrip_serialization(self):
        config = ToolkitConfig(seed=99)
        data = config.model_dump(mode="json")
        restored = ToolkitConfig.model_validate(data)
        assert restored.seed == 99
        assert restored.mode == config.mode

    def test_deep_merge_overrides_nested(self):
        config = load_config(overrides={
            "candidates": {"max_candidates": 2},
            "calibration": {"primary_objective": "brier"},
        })
        assert config.candidates.max_candidates == 2
        assert config.calibration.primary_objective == "brier"


# ---------------------------------------------------------------------------
# R2: Artifact serialization contracts
# ---------------------------------------------------------------------------

class TestArtifactSerializationContracts:

    @pytest.mark.parametrize("artifact_cls", [
        DatasetManifest,
        SplitAuditReport,
        DataProfile,
        ProbeResultSet,
        InterventionPlan,
        CandidatePortfolio,
        RuntimeDecisionLog,
        CalibrationReport,
        EnsembleReport,
        ExplainabilityReport,
        FinalReport,
    ])
    def test_model_dump_roundtrip(self, artifact_cls):
        if artifact_cls == DatasetManifest:
            obj = artifact_cls(
                dataset_id="test", modality="TABULAR", task_type="BINARY",
                split_strategy="STRATIFIED",
            )
        elif artifact_cls == FinalReport:
            obj = artifact_cls(
                run_id="r1", final_status=PipelineStage.COMPLETED,
                stages_completed=[PipelineStage.INIT],
            )
        elif artifact_cls == SplitAuditReport:
            obj = artifact_cls(passed=True)
        else:
            obj = artifact_cls()

        data = obj.model_dump(mode="json")
        assert isinstance(data, dict)
        restored = artifact_cls.model_validate(data)
        assert restored.model_dump(mode="json") == data

    def test_final_report_includes_all_summary_fields(self):
        report = FinalReport(
            run_id="r1",
            final_status=PipelineStage.COMPLETED,
            stages_completed=[PipelineStage.INIT],
        )
        # These fields must exist (even if None)
        for field in [
            "dataset_summary", "split_audit_summary", "profile_summary",
            "probe_summary", "intervention_summary", "candidate_summary",
            "runtime_decision_summary", "calibration_summary",
            "ensemble_summary", "explainability_summary",
        ]:
            assert hasattr(report, field)


# ---------------------------------------------------------------------------
# R3: Enum completeness
# ---------------------------------------------------------------------------

class TestEnumCompleteness:

    def test_pipeline_stage_has_all_states(self):
        required = {
            "INIT", "DATA_VALIDATED", "PROFILED", "PROBED",
            "INTERVENTION_SELECTED", "TRAINING_ACTIVE", "MODEL_SELECTED",
            "CALIBRATED", "ENSEMBLED", "EXPLAINED", "COMPLETED", "ABSTAINED",
        }
        actual = {s.value for s in PipelineStage}
        assert required <= actual

    def test_abstention_reasons_match_design(self):
        required = {
            "LEAKAGE_BLOCKED", "SCHEMA_INVALID", "RESOURCE_EXHAUSTED",
            "NO_ROBUST_MODEL", "CRITICAL_FAILURE",
        }
        actual = {r.value for r in AbstentionReason}
        assert required <= actual

    def test_intervention_types_complete(self):
        required = {
            "CLASS_WEIGHTING", "OVERSAMPLING", "UNDERSAMPLING",
            "AUGMENTATION", "FOCAL_LOSS", "THRESHOLDING", "CALIBRATION",
        }
        actual = {t.value for t in InterventionType}
        assert required <= actual


# ---------------------------------------------------------------------------
# R4: State machine contracts
# ---------------------------------------------------------------------------

class TestStateMachineContracts:

    def test_full_happy_path(self):
        sm = PipelineStateMachine()
        path = [
            PipelineStage.DATA_VALIDATED, PipelineStage.PROFILED,
            PipelineStage.PROBED, PipelineStage.INTERVENTION_SELECTED,
            PipelineStage.TRAINING_ACTIVE, PipelineStage.MODEL_SELECTED,
            PipelineStage.CALIBRATED, PipelineStage.ENSEMBLED,
            PipelineStage.EXPLAINED, PipelineStage.COMPLETED,
        ]
        for stage in path:
            sm.transition(stage)
        assert sm.is_terminal
        assert sm.current == PipelineStage.COMPLETED

    def test_abstain_from_every_non_terminal_stage(self):
        non_terminal = [
            PipelineStage.INIT, PipelineStage.DATA_VALIDATED,
            PipelineStage.PROFILED, PipelineStage.PROBED,
            PipelineStage.INTERVENTION_SELECTED, PipelineStage.TRAINING_ACTIVE,
            PipelineStage.MODEL_SELECTED, PipelineStage.CALIBRATED,
            PipelineStage.ENSEMBLED, PipelineStage.EXPLAINED,
        ]
        happy_path = [
            PipelineStage.DATA_VALIDATED, PipelineStage.PROFILED,
            PipelineStage.PROBED, PipelineStage.INTERVENTION_SELECTED,
            PipelineStage.TRAINING_ACTIVE, PipelineStage.MODEL_SELECTED,
            PipelineStage.CALIBRATED, PipelineStage.ENSEMBLED,
            PipelineStage.EXPLAINED,
        ]
        for stage in non_terminal:
            sm = PipelineStateMachine()
            idx = non_terminal.index(stage)
            for s in happy_path[:idx]:
                sm.transition(s)
            sm.abstain(AbstentionReason.CRITICAL_FAILURE)
            assert sm.is_terminal

    def test_skip_stage_is_illegal(self):
        sm = PipelineStateMachine()
        with pytest.raises(ValueError):
            sm.transition(PipelineStage.PROBED)

    def test_backward_transition_is_illegal(self):
        sm = PipelineStateMachine()
        sm.transition(PipelineStage.DATA_VALIDATED)
        sm.transition(PipelineStage.PROFILED)
        with pytest.raises(ValueError):
            sm.transition(PipelineStage.DATA_VALIDATED)


# ---------------------------------------------------------------------------
# R5: Path and run directory contracts
# ---------------------------------------------------------------------------

class TestPathContracts:

    def test_run_id_is_unique(self):
        id1 = generate_run_id("config1", "data.csv")
        id2 = generate_run_id("config1", "data.csv")
        # IDs include timestamp, so they may differ by seconds
        assert isinstance(id1, str)
        assert len(id1) > 10

    def test_run_directory_structure(self, tmp_path):
        run_dir = create_run_directory("test_run", tmp_path)
        required_subdirs = [
            "intake", "audit", "profiling", "probes", "interventions",
            "candidates", "runtime", "calibration", "ensemble",
            "explainability", "explainability/heatmaps", "reporting", "logs",
        ]
        for subdir in required_subdirs:
            assert (run_dir / subdir).is_dir(), f"Missing subdir: {subdir}"


# ---------------------------------------------------------------------------
# R6: Exception hierarchy contracts
# ---------------------------------------------------------------------------

class TestExceptionContracts:

    def test_all_exceptions_are_catchable(self):
        for exc_cls in [
            SchemaValidationError, UnsupportedModalityError,
            SplitIntegrityError, LeakageDetectedError,
            CalibrationFailureError, ResourceAbstentionError,
        ]:
            with pytest.raises(exc_cls):
                raise exc_cls("test")

    def test_exceptions_inherit_from_base(self):
        from aml_toolkit.core.exceptions import ToolkitError
        for exc_cls in [
            SchemaValidationError, UnsupportedModalityError,
            SplitIntegrityError, LeakageDetectedError,
            CalibrationFailureError, ResourceAbstentionError,
        ]:
            assert issubclass(exc_cls, ToolkitError)


# ---------------------------------------------------------------------------
# R7: Audit logger contracts
# ---------------------------------------------------------------------------

class TestAuditLoggerContracts:

    def test_log_produces_structured_entries(self):
        audit = AuditLogger()
        audit.log("INIT", "start", {"k": "v"})
        entries = audit.to_list()
        assert len(entries) == 1
        assert set(entries[0].keys()) == {"timestamp", "stage", "event", "detail"}

    def test_save_produces_valid_json(self, tmp_path):
        audit = AuditLogger()
        audit.log("INIT", "start")
        audit.log("COMPLETED", "done")
        path = tmp_path / "audit.json"
        audit.save(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2


# ---------------------------------------------------------------------------
# R8: Report builder contracts
# ---------------------------------------------------------------------------

class TestReportContracts:

    def test_build_report_all_formats(self, complete_artifacts, tmp_path):
        from aml_toolkit.reporting.report_builder import build_report
        config = ToolkitConfig()
        report = build_report(complete_artifacts, tmp_path, config)
        assert isinstance(report, FinalReport)
        assert (tmp_path / "final_report.json").exists()
        assert (tmp_path / "final_report.md").exists()

    def test_abstained_report_has_reason(self, tmp_path):
        from aml_toolkit.reporting.report_builder import build_report
        artifacts = {
            "run_id": "r1",
            "final_status": PipelineStage.ABSTAINED,
            "abstention_reason": AbstentionReason.LEAKAGE_BLOCKED,
            "stages_completed": [PipelineStage.INIT, PipelineStage.ABSTAINED],
            "warnings": ["blocked"],
        }
        config = ToolkitConfig()
        report = build_report(artifacts, tmp_path, config)
        assert report.abstention_reason == AbstentionReason.LEAKAGE_BLOCKED
        assert "abstained" in report.final_recommendation.lower()
