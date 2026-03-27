"""Tests for dataset intake: schema parsing, modality/task detection, splitting, and the intake manager."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import ModalityType, SplitStrategy, TaskType
from aml_toolkit.core.exceptions import SchemaValidationError, UnsupportedModalityError
from aml_toolkit.intake.dataset_intake_manager import IntakeResult, run_intake
from aml_toolkit.intake.modality_detector import detect_modality
from aml_toolkit.intake.schema_parser import (
    validate_embedding_schema,
    validate_image_folder_schema,
    validate_tabular_schema,
)
from aml_toolkit.intake.split_builder import build_splits
from aml_toolkit.intake.task_detector import (
    detect_task_type_from_array,
    detect_task_type_from_series,
)


# --- Fixtures ---


@pytest.fixture
def binary_csv(tmp_path: Path) -> Path:
    """Create a simple binary classification CSV."""
    df = pd.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
        "label": np.array([0] * 50 + [1] * 50),
    })
    path = tmp_path / "binary.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def multiclass_csv(tmp_path: Path) -> Path:
    """Create a multiclass classification CSV."""
    df = pd.DataFrame({
        "feat1": np.random.randn(150),
        "feat2": np.random.randn(150),
        "label": np.array(["cat"] * 50 + ["dog"] * 50 + ["bird"] * 50),
    })
    path = tmp_path / "multiclass.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def grouped_csv(tmp_path: Path) -> Path:
    """Create a CSV with a group column."""
    n = 200
    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "label": np.random.choice([0, 1], size=n),
        "patient_id": np.repeat(np.arange(20), 10),
    })
    path = tmp_path / "grouped.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def temporal_csv(tmp_path: Path) -> Path:
    """Create a CSV with a time column."""
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "label": np.random.choice([0, 1], size=n),
        "date": dates,
    })
    path = tmp_path / "temporal.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def image_folder(tmp_path: Path) -> Path:
    """Create a minimal image folder structure."""
    for cls in ["cats", "dogs"]:
        cls_dir = tmp_path / "images" / cls
        cls_dir.mkdir(parents=True)
        for i in range(10):
            (cls_dir / f"img_{i}.jpg").write_bytes(b"\xff\xd8\xff")
    return tmp_path / "images"


@pytest.fixture
def embedding_npz(tmp_path: Path) -> Path:
    """Create a .npz file with embeddings and labels."""
    embeddings = np.random.randn(100, 64)
    labels = np.array([0] * 50 + [1] * 50)
    path = tmp_path / "embeddings.npz"
    np.savez(path, embeddings=embeddings, labels=labels)
    return path


# --- Schema Parser Tests ---


class TestTabularSchemaValidation:
    def test_valid_csv(self, binary_csv: Path):
        df = pd.read_csv(binary_csv)
        config = ToolkitConfig(dataset={"target_column": "label"})
        schema = validate_tabular_schema(df, config)
        assert schema["target_column"] == "label"
        assert "feat1" in schema["feature_columns"]
        assert "feat2" in schema["feature_columns"]
        assert len(schema["class_labels"]) == 2

    def test_missing_target_column(self, binary_csv: Path):
        df = pd.read_csv(binary_csv)
        config = ToolkitConfig(dataset={"target_column": "nonexistent"})
        with pytest.raises(SchemaValidationError, match="not found"):
            validate_tabular_schema(df, config)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"label": pd.Series(dtype=int)})
        config = ToolkitConfig(dataset={"target_column": "label"})
        with pytest.raises(SchemaValidationError, match="empty"):
            validate_tabular_schema(df, config)

    def test_all_na_target(self):
        df = pd.DataFrame({"feat": [1, 2, 3], "label": [None, None, None]})
        config = ToolkitConfig(dataset={"target_column": "label"})
        with pytest.raises(SchemaValidationError, match="missing values"):
            validate_tabular_schema(df, config)

    def test_excludes_group_and_time_columns(self):
        df = pd.DataFrame({
            "feat": [1, 2],
            "label": [0, 1],
            "group": ["a", "b"],
            "ts": ["2024-01-01", "2024-01-02"],
        })
        config = ToolkitConfig(
            dataset={
                "target_column": "label",
                "group_column": "group",
                "time_column": "ts",
            }
        )
        schema = validate_tabular_schema(df, config)
        assert "group" not in schema["feature_columns"]
        assert "ts" not in schema["feature_columns"]
        assert "feat" in schema["feature_columns"]


class TestImageFolderSchemaValidation:
    def test_valid_image_folder(self, image_folder: Path):
        schema = validate_image_folder_schema(image_folder)
        assert "cats" in schema["class_labels"]
        assert "dogs" in schema["class_labels"]
        assert schema["total_samples"] == 20

    def test_nonexistent_path(self, tmp_path: Path):
        with pytest.raises(SchemaValidationError, match="not a directory"):
            validate_image_folder_schema(tmp_path / "nonexistent")

    def test_single_class_folder(self, tmp_path: Path):
        cls_dir = tmp_path / "only_class"
        cls_dir.mkdir()
        (cls_dir / "img.jpg").write_bytes(b"\xff")
        with pytest.raises(SchemaValidationError, match="at least 2"):
            validate_image_folder_schema(tmp_path)

    def test_empty_class_folder(self, tmp_path: Path):
        (tmp_path / "cats").mkdir()
        (tmp_path / "dogs").mkdir()
        (tmp_path / "cats" / "img.jpg").write_bytes(b"\xff")
        # dogs is empty
        with pytest.raises(SchemaValidationError, match="no recognized image"):
            validate_image_folder_schema(tmp_path)


class TestEmbeddingSchemaValidation:
    def test_valid_embedding(self):
        embeddings = np.random.randn(50, 32)
        labels = np.array([0] * 25 + [1] * 25)
        schema = validate_embedding_schema(embeddings, labels)
        assert schema["total_samples"] == 50
        assert schema["embedding_dim"] == 32
        assert len(schema["class_labels"]) == 2

    def test_shape_mismatch(self):
        embeddings = np.random.randn(50, 32)
        labels = np.array([0] * 40)
        with pytest.raises(SchemaValidationError, match="does not match"):
            validate_embedding_schema(embeddings, labels)

    def test_1d_embeddings_rejected(self):
        with pytest.raises(SchemaValidationError, match="must be 2D"):
            validate_embedding_schema(np.array([1, 2, 3]), np.array([0, 1, 0]))

    def test_multilabel_embedding(self):
        embeddings = np.random.randn(30, 16)
        labels = np.random.randint(0, 2, size=(30, 4))
        schema = validate_embedding_schema(embeddings, labels)
        assert len(schema["class_labels"]) == 4


# --- Modality Detection Tests ---


class TestModalityDetector:
    def test_csv_auto_detect(self, binary_csv: Path):
        config = ToolkitConfig(dataset={"path": str(binary_csv)})
        assert detect_modality(config) == ModalityType.TABULAR

    def test_directory_auto_detect(self, image_folder: Path):
        config = ToolkitConfig(dataset={"path": str(image_folder)})
        assert detect_modality(config) == ModalityType.IMAGE

    def test_npz_auto_detect(self, embedding_npz: Path):
        config = ToolkitConfig(dataset={"path": str(embedding_npz)})
        assert detect_modality(config) == ModalityType.EMBEDDING

    def test_explicit_override(self, binary_csv: Path):
        config = ToolkitConfig(
            dataset={"path": str(binary_csv), "modality_override": "EMBEDDING"}
        )
        assert detect_modality(config) == ModalityType.EMBEDDING

    def test_invalid_override(self, binary_csv: Path):
        config = ToolkitConfig(
            dataset={"path": str(binary_csv), "modality_override": "AUDIO"}
        )
        with pytest.raises(UnsupportedModalityError, match="AUDIO"):
            detect_modality(config)

    def test_unknown_extension(self, tmp_path: Path):
        weird_file = tmp_path / "data.parquet"
        weird_file.touch()
        config = ToolkitConfig(dataset={"path": str(weird_file)})
        with pytest.raises(UnsupportedModalityError, match="Cannot detect"):
            detect_modality(config)


# --- Task Detection Tests ---


class TestTaskDetector:
    def test_binary_series(self):
        s = pd.Series([0, 1, 0, 1, 1])
        assert detect_task_type_from_series(s) == TaskType.BINARY

    def test_multiclass_series(self):
        s = pd.Series(["a", "b", "c", "a", "b"])
        assert detect_task_type_from_series(s) == TaskType.MULTICLASS

    def test_binary_array(self):
        arr = np.array([0, 1, 0, 1])
        assert detect_task_type_from_array(arr) == TaskType.BINARY

    def test_multiclass_array(self):
        arr = np.array([0, 1, 2, 0, 1])
        assert detect_task_type_from_array(arr) == TaskType.MULTICLASS

    def test_multilabel_array(self):
        arr = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
        assert detect_task_type_from_array(arr) == TaskType.MULTILABEL

    def test_single_class_is_binary(self):
        s = pd.Series([1, 1, 1])
        assert detect_task_type_from_series(s) == TaskType.BINARY


# --- Split Builder Tests ---


class TestSplitBuilder:
    def test_stratified_split(self):
        labels = np.array([0] * 80 + [1] * 20)
        config = ToolkitConfig(splitting={"strategy": "STRATIFIED"})
        result = build_splits(100, labels, config)
        assert result.strategy == SplitStrategy.STRATIFIED
        total = len(result.train_indices) + len(result.val_indices) + len(result.test_indices)
        assert total == 100
        # No index overlap
        all_idx = set(result.train_indices) | set(result.val_indices) | set(result.test_indices)
        assert len(all_idx) == 100

    def test_grouped_split(self):
        n = 200
        labels = np.random.choice([0, 1], size=n)
        groups = np.repeat(np.arange(20), 10)
        config = ToolkitConfig(splitting={"strategy": "GROUPED"})
        result = build_splits(n, labels, config, groups=groups)
        assert result.strategy == SplitStrategy.GROUPED
        # Verify group isolation
        train_groups = set(groups[result.train_indices])
        test_groups = set(groups[result.test_indices])
        assert train_groups.isdisjoint(test_groups)

    def test_temporal_split(self):
        n = 100
        timestamps = np.arange(n)
        labels = np.random.choice([0, 1], size=n)
        config = ToolkitConfig(splitting={"strategy": "TEMPORAL"})
        result = build_splits(n, labels, config, timestamps=timestamps)
        assert result.strategy == SplitStrategy.TEMPORAL
        # Train should have earliest timestamps
        assert timestamps[result.train_indices].max() <= timestamps[result.val_indices].min()
        assert timestamps[result.val_indices].max() <= timestamps[result.test_indices].min()

    def test_grouped_without_groups_raises(self):
        config = ToolkitConfig(splitting={"strategy": "GROUPED"})
        with pytest.raises(SchemaValidationError, match="group column"):
            build_splits(100, np.zeros(100), config)

    def test_temporal_without_timestamps_raises(self):
        config = ToolkitConfig(splitting={"strategy": "TEMPORAL"})
        with pytest.raises(SchemaValidationError, match="time column"):
            build_splits(100, np.zeros(100), config)


# --- Intake Manager Integration Tests ---


class TestIntakeManager:
    def test_tabular_intake(self, binary_csv: Path):
        config = ToolkitConfig(
            dataset={"path": str(binary_csv), "target_column": "label"}
        )
        result = run_intake(config)
        assert isinstance(result, IntakeResult)
        assert result.manifest.modality == ModalityType.TABULAR
        assert result.manifest.task_type == TaskType.BINARY
        assert result.manifest.train_size > 0
        assert result.manifest.val_size > 0
        assert result.manifest.test_size > 0

    def test_multiclass_intake(self, multiclass_csv: Path):
        config = ToolkitConfig(
            dataset={"path": str(multiclass_csv), "target_column": "label"}
        )
        result = run_intake(config)
        assert result.manifest.task_type == TaskType.MULTICLASS
        assert len(result.manifest.class_labels) == 3

    def test_image_intake(self, image_folder: Path):
        config = ToolkitConfig(dataset={"path": str(image_folder)})
        result = run_intake(config)
        assert result.manifest.modality == ModalityType.IMAGE
        assert result.manifest.task_type == TaskType.BINARY
        assert len(result.manifest.class_labels) == 2

    def test_embedding_intake(self, embedding_npz: Path):
        config = ToolkitConfig(dataset={"path": str(embedding_npz)})
        result = run_intake(config)
        assert result.manifest.modality == ModalityType.EMBEDDING
        assert result.manifest.task_type == TaskType.BINARY

    def test_grouped_intake(self, grouped_csv: Path):
        config = ToolkitConfig(
            dataset={
                "path": str(grouped_csv),
                "target_column": "label",
                "group_column": "patient_id",
            },
            splitting={"strategy": "GROUPED"},
        )
        result = run_intake(config)
        assert result.manifest.split_strategy == SplitStrategy.GROUPED
        assert result.manifest.group_column == "patient_id"

    def test_temporal_intake(self, temporal_csv: Path):
        config = ToolkitConfig(
            dataset={
                "path": str(temporal_csv),
                "target_column": "label",
                "time_column": "date",
            },
            splitting={"strategy": "TEMPORAL"},
        )
        result = run_intake(config)
        assert result.manifest.split_strategy == SplitStrategy.TEMPORAL

    def test_invalid_path_raises(self):
        config = ToolkitConfig(dataset={"path": "/nonexistent/data.csv"})
        with pytest.raises(SchemaValidationError, match="not found"):
            run_intake(config)

    def test_manifest_serializes(self, binary_csv: Path, tmp_path: Path):
        config = ToolkitConfig(
            dataset={"path": str(binary_csv), "target_column": "label"}
        )
        result = run_intake(config)
        from aml_toolkit.utils.serialization import load_artifact_json, save_artifact_json
        from aml_toolkit.artifacts import DatasetManifest

        path = tmp_path / "manifest.json"
        save_artifact_json(result.manifest, path)
        loaded = load_artifact_json(DatasetManifest, path)
        assert loaded.dataset_id == result.manifest.dataset_id
