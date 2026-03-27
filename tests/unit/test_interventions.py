"""Tests for intervention planner and rule evaluators."""

import pytest

from aml_toolkit.artifacts import (
    DataProfile,
    InterventionPlan,
    ProbeResultSet,
    SplitAuditReport,
)
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import InterventionType, RiskFlag
from aml_toolkit.interventions.augmentation import evaluate_augmentation
from aml_toolkit.interventions.intervention_planner import plan_interventions
from aml_toolkit.interventions.resampling import (
    evaluate_oversampling,
    evaluate_undersampling,
)
from aml_toolkit.interventions.thresholding import (
    evaluate_calibration_required,
    evaluate_thresholding,
)
from aml_toolkit.interventions.weighting import evaluate_weighting


def _imbalanced_profile(**overrides) -> DataProfile:
    defaults = dict(
        total_samples=1000,
        class_counts={"0": 900, "1": 100},
        imbalance_severity="moderate",
        risk_flags=[RiskFlag.CLASS_IMBALANCE],
    )
    defaults.update(overrides)
    return DataProfile(**defaults)


def _clean_profile() -> DataProfile:
    return DataProfile(
        total_samples=1000,
        class_counts={"0": 500, "1": 500},
        imbalance_severity="none",
        risk_flags=[],
    )


def _passed_audit() -> SplitAuditReport:
    return SplitAuditReport(passed=True, augmentation_leakage_safe=True)


def _failed_audit() -> SplitAuditReport:
    return SplitAuditReport(
        passed=False,
        blocking_issues=["duplicate leakage"],
        augmentation_leakage_safe=False,
    )


# --- Weighting Rule Tests ---


class TestWeighting:
    def test_selected_when_imbalanced(self):
        ok, reason = evaluate_weighting(_imbalanced_profile(), ToolkitConfig())
        assert ok is True
        assert "Class weighting selected" in reason

    def test_rejected_when_balanced(self):
        ok, reason = evaluate_weighting(_clean_profile(), ToolkitConfig())
        assert ok is False
        assert "not needed" in reason

    def test_rejected_when_not_allowed(self):
        config = ToolkitConfig(interventions={"allowed_types": ["OVERSAMPLING"]})
        ok, reason = evaluate_weighting(_imbalanced_profile(), config)
        assert ok is False
        assert "not in allowed" in reason


# --- Oversampling Rule Tests ---


class TestOversampling:
    def test_selected_when_imbalanced_no_noise(self):
        ok, reason = evaluate_oversampling(_imbalanced_profile(), None, ToolkitConfig())
        assert ok is True

    def test_rejected_due_to_label_conflict(self):
        profile = _imbalanced_profile(
            risk_flags=[RiskFlag.CLASS_IMBALANCE, RiskFlag.LABEL_CONFLICT],
            label_conflict_summary={"conflict_fraction": 0.2},
        )
        ok, reason = evaluate_oversampling(profile, None, ToolkitConfig())
        assert ok is False
        assert "noise" in reason.lower() or "conflict" in reason.lower()

    def test_rejected_due_to_high_conflict_fraction(self):
        profile = _imbalanced_profile(
            label_conflict_summary={"conflict_fraction": 0.25},
        )
        config = ToolkitConfig(interventions={"oversampling_noise_risk_threshold": 0.15})
        ok, reason = evaluate_oversampling(profile, None, config)
        assert ok is False

    def test_rejected_due_to_label_noise_flag(self):
        profile = _imbalanced_profile(
            risk_flags=[RiskFlag.CLASS_IMBALANCE, RiskFlag.LABEL_NOISE],
        )
        ok, reason = evaluate_oversampling(profile, None, ToolkitConfig())
        assert ok is False
        assert "LABEL_NOISE" in reason


# --- Undersampling Rule Tests ---


class TestUndersampling:
    def test_selected_when_imbalanced(self):
        ok, reason = evaluate_undersampling(_imbalanced_profile(), ToolkitConfig())
        assert ok is True

    def test_rejected_when_balanced(self):
        ok, reason = evaluate_undersampling(_clean_profile(), ToolkitConfig())
        assert ok is False


# --- Augmentation Rule Tests ---


class TestAugmentation:
    def test_selected_when_audit_passed(self):
        ok, reason = evaluate_augmentation(_clean_profile(), _passed_audit(), ToolkitConfig())
        assert ok is True

    def test_rejected_when_audit_failed(self):
        ok, reason = evaluate_augmentation(_clean_profile(), _failed_audit(), ToolkitConfig())
        assert ok is False
        assert "FR-128" in reason or "blocking" in reason

    def test_rejected_when_not_allowed(self):
        config = ToolkitConfig(interventions={"allowed_types": ["CLASS_WEIGHTING"]})
        ok, reason = evaluate_augmentation(_clean_profile(), _passed_audit(), config)
        assert ok is False


# --- Calibration / Thresholding Rule Tests ---


class TestCalibrationThresholding:
    def test_calibration_required_when_imbalanced(self):
        config = ToolkitConfig(interventions={"require_calibration_when_imbalanced": True})
        ok, reason = evaluate_calibration_required(_imbalanced_profile(), config)
        assert ok is True
        assert "required" in reason.lower()

    def test_calibration_not_required_when_balanced(self):
        ok, reason = evaluate_calibration_required(_clean_profile(), ToolkitConfig())
        assert ok is False

    def test_thresholding_selected_when_imbalanced(self):
        ok, reason = evaluate_thresholding(_imbalanced_profile(), ToolkitConfig())
        assert ok is True

    def test_thresholding_not_needed_when_balanced(self):
        ok, reason = evaluate_thresholding(_clean_profile(), ToolkitConfig())
        assert ok is False


# --- Intervention Planner Integration Tests ---


class TestInterventionPlanner:
    def test_imbalanced_plan_selects_weighting(self):
        plan = plan_interventions(
            _imbalanced_profile(), _passed_audit(), None, ToolkitConfig()
        )
        assert isinstance(plan, InterventionPlan)
        selected_types = [e.intervention_type for e in plan.selected_interventions]
        assert InterventionType.CLASS_WEIGHTING in selected_types

    def test_prefers_weighting_over_oversampling_when_noise(self):
        """When label noise blocks oversampling, weighting should still be selected."""
        profile = _imbalanced_profile(
            risk_flags=[RiskFlag.CLASS_IMBALANCE, RiskFlag.LABEL_CONFLICT],
            label_conflict_summary={"conflict_fraction": 0.25},
        )
        plan = plan_interventions(profile, _passed_audit(), None, ToolkitConfig())

        selected_types = [e.intervention_type for e in plan.selected_interventions]
        rejected_types = [e.intervention_type for e in plan.rejected_interventions]

        assert InterventionType.CLASS_WEIGHTING in selected_types
        assert InterventionType.OVERSAMPLING in rejected_types

    def test_oversampling_rejected_logged_with_reason(self):
        profile = _imbalanced_profile(
            risk_flags=[RiskFlag.CLASS_IMBALANCE, RiskFlag.LABEL_CONFLICT],
            label_conflict_summary={"conflict_fraction": 0.3},
        )
        plan = plan_interventions(profile, _passed_audit(), None, ToolkitConfig())

        os_rejected = [
            e for e in plan.rejected_interventions
            if e.intervention_type == InterventionType.OVERSAMPLING
        ]
        assert len(os_rejected) == 1
        assert "noise" in os_rejected[0].rationale.lower() or "conflict" in os_rejected[0].rationale.lower()

    def test_calibration_required_in_plan(self):
        config = ToolkitConfig(interventions={"require_calibration_when_imbalanced": True})
        plan = plan_interventions(_imbalanced_profile(), _passed_audit(), None, config)

        selected_types = [e.intervention_type for e in plan.selected_interventions]
        assert InterventionType.CALIBRATION in selected_types

    def test_abstention_suggested_when_audit_failed(self):
        plan = plan_interventions(_imbalanced_profile(), _failed_audit(), None, ToolkitConfig())
        assert any("abstention" in c.lower() or "WARNING" in c for c in plan.safety_constraints)

    def test_clean_data_minimal_plan(self):
        plan = plan_interventions(_clean_profile(), _passed_audit(), None, ToolkitConfig())
        # No imbalance → no weighting, no resampling, no thresholding
        selected_types = [e.intervention_type for e in plan.selected_interventions]
        assert InterventionType.CLASS_WEIGHTING not in selected_types
        assert InterventionType.OVERSAMPLING not in selected_types
        # Augmentation should be eligible (audit passed, allowed)
        assert InterventionType.AUGMENTATION in selected_types

    def test_plan_serializes(self, tmp_path):
        plan = plan_interventions(_imbalanced_profile(), _passed_audit(), None, ToolkitConfig())

        from aml_toolkit.utils.serialization import save_artifact_json, load_artifact_json

        path = tmp_path / "plan.json"
        save_artifact_json(plan, path)
        loaded = load_artifact_json(InterventionPlan, path)
        assert len(loaded.selected_interventions) == len(plan.selected_interventions)

    def test_execution_order_follows_selection(self):
        plan = plan_interventions(_imbalanced_profile(), _passed_audit(), None, ToolkitConfig())
        # Execution order should only contain selected types
        selected_types = {e.intervention_type for e in plan.selected_interventions}
        for t in plan.execution_order:
            assert t in selected_types
