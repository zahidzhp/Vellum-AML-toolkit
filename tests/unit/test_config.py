"""Tests for config loading with hierarchical merging and mode overlays."""

from pathlib import Path

import pytest
import yaml

from aml_toolkit.core.config import ToolkitConfig, load_config, load_yaml, _deep_merge
from aml_toolkit.core.enums import OperatingMode


@pytest.fixture
def configs_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory with default + mode files."""
    cdir = tmp_path / "configs"
    cdir.mkdir()
    modes_dir = cdir / "modes"
    modes_dir.mkdir()

    default = {
        "mode": "BALANCED",
        "seed": 42,
        "profiling": {
            "imbalance_ratio_warning": 5.0,
            "imbalance_ratio_severe": 20.0,
        },
        "candidates": {
            "allowed_families": ["logistic", "rf", "xgb", "mlp"],
            "max_candidates": 5,
        },
        "compute": {
            "max_training_time_seconds": 3600,
        },
    }
    with open(cdir / "default.yaml", "w") as f:
        yaml.dump(default, f)

    conservative = {
        "profiling": {
            "imbalance_ratio_warning": 3.0,
        },
        "candidates": {
            "max_candidates": 3,
        },
        "compute": {
            "max_training_time_seconds": 1800,
        },
    }
    with open(modes_dir / "conservative.yaml", "w") as f:
        yaml.dump(conservative, f)

    return cdir


def test_load_default_config(configs_dir: Path):
    cfg = load_config(configs_dir=configs_dir)
    assert cfg.mode == OperatingMode.BALANCED
    assert cfg.seed == 42
    assert cfg.profiling.imbalance_ratio_warning == 5.0


def test_load_with_mode_overlay(configs_dir: Path):
    cfg = load_config(mode="CONSERVATIVE", configs_dir=configs_dir)
    assert cfg.mode == OperatingMode.CONSERVATIVE
    # Overridden by conservative.yaml
    assert cfg.profiling.imbalance_ratio_warning == 3.0
    assert cfg.candidates.max_candidates == 3
    assert cfg.compute.max_training_time_seconds == 1800
    # Not overridden — stays at default
    assert cfg.profiling.imbalance_ratio_severe == 20.0


def test_load_with_user_config(configs_dir: Path, tmp_path: Path):
    user_config = tmp_path / "user.yaml"
    with open(user_config, "w") as f:
        yaml.dump({"seed": 99, "candidates": {"max_candidates": 10}}, f)

    cfg = load_config(config_path=user_config, configs_dir=configs_dir)
    assert cfg.seed == 99
    assert cfg.candidates.max_candidates == 10


def test_load_with_overrides(configs_dir: Path):
    cfg = load_config(
        overrides={"seed": 123, "compute": {"gpu_enabled": False}},
        configs_dir=configs_dir,
    )
    assert cfg.seed == 123
    assert cfg.compute.gpu_enabled is False


def test_deep_merge():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99, "e": 5}, "f": 6}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": {"c": 99, "d": 3, "e": 5}, "f": 6}


def test_default_config_has_all_sections():
    """A fresh ToolkitConfig must have all config sections."""
    cfg = ToolkitConfig()
    assert cfg.dataset is not None
    assert cfg.splitting is not None
    assert cfg.profiling is not None
    assert cfg.probes is not None
    assert cfg.interventions is not None
    assert cfg.candidates is not None
    assert cfg.runtime_decision is not None
    assert cfg.calibration is not None
    assert cfg.ensemble is not None
    assert cfg.explainability is not None
    assert cfg.reporting is not None
    assert cfg.compute is not None


def test_config_serializes_to_json():
    cfg = ToolkitConfig()
    json_str = cfg.model_dump_json()
    assert "BALANCED" in json_str


def test_load_yaml_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    data = load_yaml(empty)
    assert data == {}


def test_missing_mode_file_graceful(configs_dir: Path):
    """Loading a mode that has no file should work (just skip overlay)."""
    cfg = load_config(mode="AGGRESSIVE", configs_dir=configs_dir)
    # No aggressive.yaml in our test fixture, so defaults apply
    assert cfg.mode == OperatingMode.AGGRESSIVE
    assert cfg.profiling.imbalance_ratio_warning == 5.0
