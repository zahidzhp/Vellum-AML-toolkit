"""Tests for Phase 2.1 — Advanced config expansion."""

from pathlib import Path

import pytest

from aml_toolkit.core.config import (
    AdvancedConfig,
    AgenticPlannerConfig,
    DynamicEnsembleConfig,
    MetaPolicyConfig,
    RunHistoryConfig,
    ToolkitConfig,
    UncertaintyConfig,
    VersionFeaturesConfig,
    load_config,
)
from aml_toolkit.core.enums import AbstentionReason

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
PROFILES_DIR = CONFIGS_DIR / "profiles"


# ---------------------------------------------------------------------------
# Default field values
# ---------------------------------------------------------------------------

class TestRunHistoryConfig:
    def test_defaults(self):
        cfg = RunHistoryConfig()
        assert cfg.enabled is False
        assert "run_history.jsonl" in cfg.store_path
        assert cfg.max_records == 1000


class TestUncertaintyConfig:
    def test_defaults(self):
        cfg = UncertaintyConfig()
        assert cfg.enabled is False
        assert "entropy" in cfg.methods
        assert "margin" in cfg.methods
        assert cfg.aggregation == "mean"
        assert cfg.abstain_if_above == 0.8
        assert cfg.use_calibrated_proba is True
        assert cfg.conformal_enabled is False
        assert cfg.conformal_coverage == 0.9
        assert cfg.use_cross_val is False
        assert cfg.cross_val_folds == 5


class TestDynamicEnsembleConfig:
    def test_defaults(self):
        cfg = DynamicEnsembleConfig()
        assert cfg.enabled is False
        assert "greedy_diverse" in cfg.allowed_modes
        assert cfg.max_members == 4
        assert cfg.diversity_threshold == 0.05
        assert cfg.use_uncertainty_weights is False


class TestMetaPolicyConfig:
    def test_defaults(self):
        cfg = MetaPolicyConfig()
        assert cfg.enabled is False
        assert cfg.exploration_weight == 0.3
        assert cfg.never_override_user_constraints is True
        assert cfg.compute_budget_aware is True
        assert cfg.similarity_method == "cosine"
        assert cfg.recency_decay == 0.9


class TestAgenticPlannerConfig:
    def test_defaults(self):
        cfg = AgenticPlannerConfig()
        assert cfg.enabled is False
        assert cfg.mode == "propose_only"
        assert cfg.max_suggestions == 3
        assert cfg.llm_enhanced is False
        assert cfg.track_proposal_outcomes is True


class TestVersionFeaturesConfig:
    def test_defaults_all_false(self):
        cfg = VersionFeaturesConfig()
        assert cfg.uncertainty is False
        assert cfg.dynamic_ensemble is False
        assert cfg.meta_policy is False
        assert cfg.agentic_planner is False
        assert cfg.run_history is False


class TestAdvancedConfig:
    def test_nests_all_sub_configs(self):
        cfg = AdvancedConfig()
        assert isinstance(cfg.run_history, RunHistoryConfig)
        assert isinstance(cfg.uncertainty, UncertaintyConfig)
        assert isinstance(cfg.dynamic_ensemble, DynamicEnsembleConfig)
        assert isinstance(cfg.meta_policy, MetaPolicyConfig)
        assert isinstance(cfg.agentic_planner, AgenticPlannerConfig)
        assert isinstance(cfg.version_features, VersionFeaturesConfig)


# ---------------------------------------------------------------------------
# ToolkitConfig integration
# ---------------------------------------------------------------------------

class TestToolkitConfigAdvanced:
    def test_toolkit_config_has_advanced(self):
        cfg = ToolkitConfig()
        assert isinstance(cfg.advanced, AdvancedConfig)

    def test_all_v2_off_by_default(self):
        cfg = ToolkitConfig()
        assert cfg.advanced.uncertainty.enabled is False
        assert cfg.advanced.dynamic_ensemble.enabled is False
        assert cfg.advanced.meta_policy.enabled is False
        assert cfg.advanced.agentic_planner.enabled is False
        assert cfg.advanced.run_history.enabled is False

    def test_load_config_includes_advanced(self):
        cfg = load_config(configs_dir=CONFIGS_DIR)
        assert hasattr(cfg, "advanced")
        assert isinstance(cfg.advanced, AdvancedConfig)

    def test_programmatic_override_enables_uncertainty(self):
        cfg = load_config(
            configs_dir=CONFIGS_DIR,
            overrides={"advanced": {"uncertainty": {"enabled": True}}},
        )
        assert cfg.advanced.uncertainty.enabled is True

    def test_programmatic_override_preserves_other_fields(self):
        cfg = load_config(
            configs_dir=CONFIGS_DIR,
            overrides={"advanced": {"uncertainty": {"enabled": True, "conformal_enabled": True}}},
        )
        # Other advanced fields should keep defaults
        assert cfg.advanced.meta_policy.enabled is False
        assert cfg.advanced.uncertainty.conformal_enabled is True


# ---------------------------------------------------------------------------
# Profile YAMLs
# ---------------------------------------------------------------------------

class TestProfileYAMLs:
    def test_conservative_profile_loads(self):
        path = PROFILES_DIR / "conservative.yaml"
        assert path.exists(), f"Profile not found: {path}"
        cfg = load_config(config_path=path, configs_dir=CONFIGS_DIR)
        assert cfg.advanced.version_features.uncertainty is False

    def test_balanced_profile_enables_uncertainty_and_history(self):
        path = PROFILES_DIR / "balanced.yaml"
        assert path.exists()
        cfg = load_config(config_path=path, configs_dir=CONFIGS_DIR)
        assert cfg.advanced.uncertainty.enabled is True
        assert cfg.advanced.run_history.enabled is True

    def test_advanced_profile_enables_ensemble_and_meta(self):
        path = PROFILES_DIR / "advanced.yaml"
        assert path.exists()
        cfg = load_config(config_path=path, configs_dir=CONFIGS_DIR)
        assert cfg.advanced.dynamic_ensemble.enabled is True
        assert cfg.advanced.meta_policy.enabled is True
        assert cfg.advanced.uncertainty.conformal_enabled is True

    def test_research_profile_enables_all_features(self):
        path = PROFILES_DIR / "research.yaml"
        assert path.exists()
        cfg = load_config(config_path=path, configs_dir=CONFIGS_DIR)
        assert cfg.advanced.agentic_planner.enabled is True
        assert cfg.advanced.uncertainty.use_cross_val is True
        assert cfg.advanced.dynamic_ensemble.use_uncertainty_weights is True

    def test_old_config_loads_without_advanced_section(self, tmp_path):
        """YAML without advanced: section must still load cleanly."""
        minimal = tmp_path / "minimal.yaml"
        minimal.write_text("mode: BALANCED\nseed: 42\n")
        cfg = load_config(config_path=minimal, configs_dir=CONFIGS_DIR)
        # All V2 features should be off by default
        assert cfg.advanced.uncertainty.enabled is False


# ---------------------------------------------------------------------------
# AbstentionReason enum
# ---------------------------------------------------------------------------

class TestAbstentionReasonEnum:
    def test_high_uncertainty_enum_exists(self):
        assert AbstentionReason.HIGH_UNCERTAINTY == "HIGH_UNCERTAINTY"

    def test_existing_reasons_unchanged(self):
        assert AbstentionReason.LEAKAGE_BLOCKED == "LEAKAGE_BLOCKED"
        assert AbstentionReason.NO_ROBUST_MODEL == "NO_ROBUST_MODEL"
        assert AbstentionReason.CRITICAL_FAILURE == "CRITICAL_FAILURE"
