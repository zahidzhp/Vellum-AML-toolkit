"""Tests for artifact serialization and deserialization."""

import json
from pathlib import Path

import pytest

from aml_toolkit.artifacts import (
    CalibrationReport,
    CalibrationResult,
    CandidateEntry,
    CandidatePortfolio,
    DataProfile,
    DatasetManifest,
    EnsembleReport,
    ExplainabilityOutput,
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
from aml_toolkit.core.enums import (
    DecisionType,
    InterventionType,
    ModalityType,
    PipelineStage,
    RiskFlag,
    SplitStrategy,
    TaskType,
)
from aml_toolkit.utils.serialization import load_artifact_json, save_artifact_json


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


def _roundtrip_json(artifact, artifact_class, tmp_dir: Path):
    """Save to JSON, load back, and verify equality."""
    path = tmp_dir / "test_artifact.json"
    save_artifact_json(artifact, path)
    loaded = load_artifact_json(artifact_class, path)
    assert artifact.model_dump() == loaded.model_dump()
    # Verify it's valid JSON
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    return loaded


class TestDatasetManifest:
    def test_defaults(self):
        m = DatasetManifest(
            dataset_id="ds_001",
            modality=ModalityType.TABULAR,
            task_type=TaskType.BINARY,
            split_strategy=SplitStrategy.STRATIFIED,
        )
        assert m.dataset_id == "ds_001"
        assert m.class_labels == []
        assert m.warnings == []

    def test_roundtrip(self, tmp_dir):
        m = DatasetManifest(
            dataset_id="ds_002",
            modality=ModalityType.IMAGE,
            task_type=TaskType.MULTICLASS,
            split_strategy=SplitStrategy.GROUPED,
            class_labels=["cat", "dog", "bird"],
            train_size=800,
            val_size=100,
            test_size=100,
        )
        _roundtrip_json(m, DatasetManifest, tmp_dir)


class TestSplitAuditReport:
    def test_defaults(self):
        r = SplitAuditReport(passed=True)
        assert r.passed is True
        assert r.blocking_issues == []

    def test_roundtrip(self, tmp_dir):
        r = SplitAuditReport(
            passed=False,
            leakage_flags=["duplicate_overlap"],
            blocking_issues=["50 duplicate rows across train/test"],
        )
        _roundtrip_json(r, SplitAuditReport, tmp_dir)


class TestDataProfile:
    def test_risk_flags(self):
        p = DataProfile(
            total_samples=1000,
            class_counts={"positive": 50, "negative": 950},
            risk_flags=[RiskFlag.CLASS_IMBALANCE, RiskFlag.LABEL_NOISE],
        )
        assert RiskFlag.CLASS_IMBALANCE in p.risk_flags
        assert p.total_samples == 1000

    def test_roundtrip(self, tmp_dir):
        p = DataProfile(
            total_samples=500,
            class_counts={"a": 250, "b": 250},
            imbalance_severity="none",
        )
        _roundtrip_json(p, DataProfile, tmp_dir)


class TestProbeResultSet:
    def test_with_probe_results(self):
        pr = ProbeResult(
            model_name="logistic",
            intervention_branch="none",
            val_metrics={"macro_f1": 0.72},
            fit_time_seconds=0.5,
        )
        prs = ProbeResultSet(
            baseline_results=[pr],
            selected_metrics=["macro_f1"],
        )
        assert len(prs.baseline_results) == 1
        assert prs.baseline_results[0].val_metrics["macro_f1"] == 0.72

    def test_roundtrip(self, tmp_dir):
        prs = ProbeResultSet(
            shallow_results=[
                ProbeResult(model_name="rf", val_metrics={"macro_f1": 0.80}),
            ],
        )
        _roundtrip_json(prs, ProbeResultSet, tmp_dir)


class TestInterventionPlan:
    def test_selected_and_rejected(self):
        plan = InterventionPlan(
            selected_interventions=[
                InterventionEntry(
                    intervention_type=InterventionType.CLASS_WEIGHTING,
                    selected=True,
                    rationale="Moderate imbalance detected",
                )
            ],
            rejected_interventions=[
                InterventionEntry(
                    intervention_type=InterventionType.OVERSAMPLING,
                    selected=False,
                    rationale="Label noise risk too high",
                )
            ],
        )
        assert len(plan.selected_interventions) == 1
        assert len(plan.rejected_interventions) == 1

    def test_roundtrip(self, tmp_dir):
        plan = InterventionPlan(rationale="test plan")
        _roundtrip_json(plan, InterventionPlan, tmp_dir)


class TestCandidatePortfolio:
    def test_with_entries(self):
        portfolio = CandidatePortfolio(
            candidate_models=[
                CandidateEntry(
                    candidate_id="xgb_01",
                    model_family="xgb",
                    model_name="XGBClassifier",
                )
            ],
            selected_families=["xgb"],
        )
        assert portfolio.candidate_models[0].candidate_id == "xgb_01"

    def test_roundtrip(self, tmp_dir):
        portfolio = CandidatePortfolio()
        _roundtrip_json(portfolio, CandidatePortfolio, tmp_dir)


class TestRuntimeDecisionLog:
    def test_with_decisions(self):
        log = RuntimeDecisionLog(
            decisions=[
                RuntimeDecision(
                    candidate_id="xgb_01",
                    epochs_seen=10,
                    decision=DecisionType.CONTINUE,
                    reasons=["improving trend"],
                )
            ]
        )
        assert log.decisions[0].decision == DecisionType.CONTINUE

    def test_roundtrip(self, tmp_dir):
        log = RuntimeDecisionLog()
        _roundtrip_json(log, RuntimeDecisionLog, tmp_dir)


class TestCalibrationReport:
    def test_with_results(self):
        report = CalibrationReport(
            results=[
                CalibrationResult(
                    candidate_id="xgb_01",
                    method="temperature_scaling",
                    ece_before=0.12,
                    ece_after=0.04,
                )
            ]
        )
        assert report.results[0].ece_after == 0.04

    def test_roundtrip(self, tmp_dir):
        report = CalibrationReport(primary_objective="brier")
        _roundtrip_json(report, CalibrationReport, tmp_dir)


class TestEnsembleReport:
    def test_rejected_ensemble(self):
        report = EnsembleReport(
            ensemble_selected=False,
            rejection_reason="marginal gain below threshold",
        )
        assert report.ensemble_selected is False
        assert report.rejection_reason is not None

    def test_roundtrip(self, tmp_dir):
        report = EnsembleReport(ensemble_selected=True, strategy="soft_voting")
        _roundtrip_json(report, EnsembleReport, tmp_dir)


class TestExplainabilityReport:
    def test_with_outputs(self):
        report = ExplainabilityReport(
            outputs=[
                ExplainabilityOutput(
                    method="gradcam",
                    candidate_id="cnn_01",
                    faithfulness_score=0.85,
                ),
                ExplainabilityOutput(
                    method="shap",
                    candidate_id="cnn_01",
                    supported=False,
                    fallback_reason="SHAP not supported for this architecture",
                ),
            ],
            methods_attempted=["gradcam", "shap"],
            methods_succeeded=["gradcam"],
            methods_failed=["shap"],
        )
        assert len(report.methods_failed) == 1

    def test_roundtrip(self, tmp_dir):
        report = ExplainabilityReport()
        _roundtrip_json(report, ExplainabilityReport, tmp_dir)


class TestFinalReport:
    def test_completed_status(self):
        report = FinalReport(
            run_id="20260327_143052_a1b2c3",
            final_status=PipelineStage.COMPLETED,
        )
        assert report.final_status == PipelineStage.COMPLETED
        assert report.abstention_reason is None

    def test_abstained_status(self):
        from aml_toolkit.core.enums import AbstentionReason

        report = FinalReport(
            run_id="test",
            final_status=PipelineStage.ABSTAINED,
            abstention_reason=AbstentionReason.NO_ROBUST_MODEL,
        )
        assert report.final_status == PipelineStage.ABSTAINED

    def test_roundtrip(self, tmp_dir):
        report = FinalReport(run_id="test_run")
        _roundtrip_json(report, FinalReport, tmp_dir)
