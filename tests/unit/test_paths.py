"""Tests for path management and run ID generation."""

from pathlib import Path

from aml_toolkit.core.paths import create_run_directory, generate_run_id


def test_generate_run_id_format():
    run_id = generate_run_id()
    parts = run_id.split("_")
    assert len(parts) == 3
    assert len(parts[0]) == 8  # YYYYMMDD
    assert len(parts[1]) == 6  # HHMMSS
    assert len(parts[2]) == 6  # short hash


def test_generate_run_id_uniqueness():
    id1 = generate_run_id(config_repr="a", dataset_path="b")
    id2 = generate_run_id(config_repr="a", dataset_path="b")
    # Timestamps differ (or hash includes time), so IDs should differ
    assert id1 != id2


def test_create_run_directory(tmp_path: Path):
    run_dir = create_run_directory("test_run_001", base_dir=tmp_path)
    assert run_dir.exists()
    assert (run_dir / "intake").is_dir()
    assert (run_dir / "audit").is_dir()
    assert (run_dir / "profiling").is_dir()
    assert (run_dir / "probes").is_dir()
    assert (run_dir / "interventions").is_dir()
    assert (run_dir / "candidates").is_dir()
    assert (run_dir / "runtime").is_dir()
    assert (run_dir / "calibration").is_dir()
    assert (run_dir / "ensemble").is_dir()
    assert (run_dir / "explainability").is_dir()
    assert (run_dir / "explainability" / "heatmaps").is_dir()
    assert (run_dir / "reporting").is_dir()
    assert (run_dir / "logs").is_dir()
