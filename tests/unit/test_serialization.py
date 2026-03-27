"""Tests for serialization utilities (JSON and YAML save/load)."""

from pathlib import Path

from aml_toolkit.artifacts import DatasetManifest
from aml_toolkit.core.enums import ModalityType, SplitStrategy, TaskType
from aml_toolkit.utils.serialization import (
    load_artifact_json,
    load_artifact_yaml,
    save_artifact_json,
    save_artifact_yaml,
)


def _make_manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_id="test_ds",
        modality=ModalityType.TABULAR,
        task_type=TaskType.BINARY,
        split_strategy=SplitStrategy.STRATIFIED,
        class_labels=["pos", "neg"],
        train_size=800,
        val_size=100,
        test_size=100,
    )


def test_json_roundtrip(tmp_path: Path):
    manifest = _make_manifest()
    path = tmp_path / "manifest.json"
    save_artifact_json(manifest, path)
    loaded = load_artifact_json(DatasetManifest, path)
    assert manifest.model_dump() == loaded.model_dump()


def test_yaml_roundtrip(tmp_path: Path):
    manifest = _make_manifest()
    path = tmp_path / "manifest.yaml"
    save_artifact_yaml(manifest, path)
    loaded = load_artifact_yaml(DatasetManifest, path)
    assert manifest.model_dump() == loaded.model_dump()


def test_json_creates_parent_dirs(tmp_path: Path):
    manifest = _make_manifest()
    path = tmp_path / "nested" / "dir" / "manifest.json"
    save_artifact_json(manifest, path)
    assert path.exists()


def test_yaml_creates_parent_dirs(tmp_path: Path):
    manifest = _make_manifest()
    path = tmp_path / "nested" / "dir" / "manifest.yaml"
    save_artifact_yaml(manifest, path)
    assert path.exists()
