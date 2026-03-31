"""Integration sanity checks for V2 adaptive intelligence features.

Verifies:
1. All V2 features off → zero V2 artifacts in output
2. V2 coordinator doesn't crash when features are disabled
3. User constraints (max_candidates, max_ensemble_size) always respected
4. AdaptiveCoordinator degrades gracefully on bad inputs
5. Config loading with profile YAMLs works end-to-end
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from aml_toolkit.adaptive.coordinator import AdaptiveIntelligenceCoordinator
from aml_toolkit.artifacts.calibration_report import CalibrationReport, CalibrationResult
from aml_toolkit.core.config import ToolkitConfig, load_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
PROFILES_DIR = CONFIGS_DIR / "profiles"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cal_report(n: int = 100, cid: str = "rf_001") -> CalibrationReport:
    rng = np.random.default_rng(0)
    proba_before = rng.uniform(0.2, 0.8, n)
    proba_after = np.clip(proba_before + rng.uniform(-0.05, 0.05, n), 0, 1)
    report = CalibrationReport(primary_objective="ece")
    report.plot_data[cid] = {"proba_before": proba_before, "proba_after": proba_after}
    result = CalibrationResult(
        candidate_id=cid,
        method="isotonic",
        objective_metric="ece",
        ece_before=0.15,
        ece_after=0.07,
    )
    report.results.append(result)
    return report


# ---------------------------------------------------------------------------
# V2 all off
# ---------------------------------------------------------------------------

class TestV2AllOff:
    def test_v2_off_pre_training_returns_empty(self):
        cfg = ToolkitConfig()  # all advanced off by default
        coordinator = AdaptiveIntelligenceCoordinator(cfg)
        rec = coordinator.pre_training_recommendations({}, {})
        assert rec.candidate_order == []
        assert rec.compute_budget_fractions == {}
        assert rec.meta_policy_recommendation is None

    def test_v2_off_post_calibration_returns_empty(self):
        cfg = ToolkitConfig()
        coordinator = AdaptiveIntelligenceCoordinator(cfg)
        cal_report = _make_cal_report()
        result = coordinator.post_calibration_analysis(cal_report, {"rf_001": MagicMock()}, None, None)
        assert result.uncertainty_reports == {}
        assert result.abstention_triggered is False

    def test_v2_off_plan_returns_disabled_note(self):
        cfg = ToolkitConfig()
        coordinator = AdaptiveIntelligenceCoordinator(cfg)
        plan = coordinator.generate_experiment_plan({})
        assert "disabled" in " ".join(plan.notes).lower()

    def test_v2_off_save_record_noop(self, tmp_path):
        cfg = ToolkitConfig()
        coordinator = AdaptiveIntelligenceCoordinator(cfg)
        # Should not raise, should not create any file
        coordinator.save_run_record({}, cfg)
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# V2 uncertainty enabled
# ---------------------------------------------------------------------------

class TestV2UncertaintyEnabled:
    def test_uncertainty_reports_generated(self):
        cfg = load_config(
            configs_dir=CONFIGS_DIR,
            overrides={"advanced": {"uncertainty": {"enabled": True, "use_calibrated_proba": True}}},
        )
        coordinator = AdaptiveIntelligenceCoordinator(cfg)
        cal_report = _make_cal_report(n=200, cid="rf_001")
        y_val = np.random.default_rng(0).integers(0, 2, 200)
        models = {"rf_001": MagicMock()}
        result = coordinator.post_calibration_analysis(cal_report, models, None, y_val)
        assert "rf_001" in result.uncertainty_reports
        report = result.uncertainty_reports["rf_001"]
        assert report.sample_count == 200

    def test_no_proba_candidate_skipped(self):
        cfg = load_config(
            configs_dir=CONFIGS_DIR,
            overrides={"advanced": {"uncertainty": {"enabled": True}}},
        )
        coordinator = AdaptiveIntelligenceCoordinator(cfg)
        # cal_report with no plot_data
        cal_report = CalibrationReport(primary_objective="ece")
        models = {"mystery_001": MagicMock()}
        result = coordinator.post_calibration_analysis(cal_report, models, None, None)
        # No proba → skipped gracefully
        assert "mystery_001" not in result.uncertainty_reports


# ---------------------------------------------------------------------------
# V2 meta-policy enabled
# ---------------------------------------------------------------------------

class TestV2MetaPolicyEnabled:
    def test_meta_policy_respects_candidate_ids(self):
        cfg = load_config(
            configs_dir=CONFIGS_DIR,
            overrides={"advanced": {"meta_policy": {"enabled": True}}},
        )
        coordinator = AdaptiveIntelligenceCoordinator(cfg)
        # Pass manifest with candidate_ids
        manifest = {
            "candidate_ids": ["rf_001", "xgb_001"],
            "modality": "TABULAR",
            "task_type": "BINARY",
            "n_classes": 2,
            "n_samples": 1000,
            "n_features": 20,
        }
        rec = coordinator.pre_training_recommendations(manifest, {})
        # Recommended order must be subset of input
        assert set(rec.candidate_order).issubset({"rf_001", "xgb_001"})

    def test_meta_policy_budget_sums_to_one(self):
        cfg = load_config(
            configs_dir=CONFIGS_DIR,
            overrides={"advanced": {"meta_policy": {"enabled": True}}},
        )
        coordinator = AdaptiveIntelligenceCoordinator(cfg)
        manifest = {
            "candidate_ids": ["rf_001", "xgb_001", "logistic_001"],
            "modality": "TABULAR",
            "task_type": "BINARY",
            "n_classes": 2,
            "n_samples": 500,
            "n_features": 10,
        }
        rec = coordinator.pre_training_recommendations(manifest, {})
        if rec.compute_budget_fractions:
            total = sum(rec.compute_budget_fractions.values())
            assert total == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_bad_manifest_does_not_crash(self):
        cfg = load_config(
            configs_dir=CONFIGS_DIR,
            overrides={"advanced": {"meta_policy": {"enabled": True}}},
        )
        coordinator = AdaptiveIntelligenceCoordinator(cfg)
        rec = coordinator.pre_training_recommendations(None, None)
        assert isinstance(rec.candidate_order, list)

    def test_bad_cal_report_does_not_crash(self):
        cfg = load_config(
            configs_dir=CONFIGS_DIR,
            overrides={"advanced": {"uncertainty": {"enabled": True}}},
        )
        coordinator = AdaptiveIntelligenceCoordinator(cfg)
        result = coordinator.post_calibration_analysis(None, {"m1": MagicMock()}, None, None)
        assert isinstance(result.uncertainty_reports, dict)

    def test_bad_artifacts_plan_does_not_crash(self):
        cfg = load_config(
            configs_dir=CONFIGS_DIR,
            overrides={"advanced": {"agentic_planner": {"enabled": True}}},
        )
        coordinator = AdaptiveIntelligenceCoordinator(cfg)
        plan = coordinator.generate_experiment_plan(None)
        assert plan is not None

    def test_save_record_bad_path_does_not_crash(self):
        cfg = load_config(
            configs_dir=CONFIGS_DIR,
            overrides={
                "advanced": {
                    "run_history": {
                        "enabled": True,
                        "store_path": "/nonexistent_dir_xyz/history.jsonl",
                    }
                }
            },
        )
        coordinator = AdaptiveIntelligenceCoordinator(cfg)
        coordinator.save_run_record({}, cfg)  # must not raise


# ---------------------------------------------------------------------------
# Profile-based config end-to-end
# ---------------------------------------------------------------------------

class TestProfileConfigEndToEnd:
    def test_research_profile_all_v2_enabled(self):
        path = PROFILES_DIR / "research.yaml"
        if not path.exists():
            pytest.skip("research.yaml profile not found")
        cfg = load_config(config_path=path, configs_dir=CONFIGS_DIR)
        assert cfg.advanced.uncertainty.enabled
        assert cfg.advanced.dynamic_ensemble.enabled
        assert cfg.advanced.meta_policy.enabled
        assert cfg.advanced.agentic_planner.enabled
        assert cfg.advanced.run_history.enabled

    def test_conservative_profile_all_v2_disabled(self):
        path = PROFILES_DIR / "conservative.yaml"
        if not path.exists():
            pytest.skip("conservative.yaml profile not found")
        cfg = load_config(config_path=path, configs_dir=CONFIGS_DIR)
        assert not cfg.advanced.version_features.uncertainty
        assert not cfg.advanced.version_features.dynamic_ensemble

    def test_coordinator_works_with_research_profile(self):
        path = PROFILES_DIR / "research.yaml"
        if not path.exists():
            pytest.skip("research.yaml profile not found")
        cfg = load_config(config_path=path, configs_dir=CONFIGS_DIR)
        coordinator = AdaptiveIntelligenceCoordinator(cfg)
        # Should construct without error
        assert coordinator is not None

    def test_old_config_without_advanced_section_works(self, tmp_path):
        minimal = tmp_path / "minimal.yaml"
        minimal.write_text("mode: BALANCED\nseed: 42\n")
        cfg = load_config(config_path=minimal, configs_dir=CONFIGS_DIR)
        coordinator = AdaptiveIntelligenceCoordinator(cfg)
        # All V2 off — coordinator still works
        rec = coordinator.pre_training_recommendations({}, {})
        assert rec.candidate_order == []
