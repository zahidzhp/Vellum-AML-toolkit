"""Dataset intake manager: orchestrates loading, detection, validation, and splitting."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aml_toolkit.artifacts import DatasetManifest
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import ModalityType, SplitStrategy, TaskType
from aml_toolkit.core.exceptions import SchemaValidationError, UnsupportedModalityError
from aml_toolkit.intake.modality_detector import detect_modality
from aml_toolkit.intake.schema_parser import (
    validate_embedding_schema,
    validate_image_folder_schema,
    validate_tabular_schema,
)
from aml_toolkit.intake.split_builder import SplitResult, build_splits
from aml_toolkit.intake.task_detector import (
    detect_task_type_from_array,
    detect_task_type_from_series,
)


class IntakeResult:
    """Container for the full intake output."""

    def __init__(
        self,
        manifest: DatasetManifest,
        data: Any,
        split_result: SplitResult | None = None,
    ) -> None:
        self.manifest = manifest
        self.data = data
        self.split_result = split_result


def run_intake(config: ToolkitConfig) -> IntakeResult:
    """Run the full dataset intake pipeline.

    Steps:
    1. Detect modality
    2. Load and validate data
    3. Detect task type
    4. Build splits
    5. Emit DatasetManifest

    Args:
        config: Toolkit configuration.

    Returns:
        IntakeResult with manifest, raw data, and split indices.

    Raises:
        SchemaValidationError: On invalid input data.
        UnsupportedModalityError: On unrecognized modality.
    """
    modality = detect_modality(config)

    if modality == ModalityType.TABULAR:
        return _intake_tabular(config, modality)
    elif modality == ModalityType.IMAGE:
        return _intake_image(config, modality)
    elif modality == ModalityType.EMBEDDING:
        return _intake_embedding(config, modality)
    else:
        raise UnsupportedModalityError(f"Modality {modality} is not yet supported.")


def _intake_tabular(config: ToolkitConfig, modality: ModalityType) -> IntakeResult:
    """Intake pipeline for tabular CSV data."""
    dataset_path = Path(config.dataset.path)
    if not dataset_path.exists():
        raise SchemaValidationError(f"Dataset file not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    schema = validate_tabular_schema(df, config)

    target_col = schema["target_column"]
    task_type = detect_task_type_from_series(df[target_col])

    labels = df[target_col].values
    n_samples = len(df)

    groups = None
    if config.dataset.group_column and config.dataset.group_column in df.columns:
        groups = df[config.dataset.group_column].values

    timestamps = None
    if config.dataset.time_column and config.dataset.time_column in df.columns:
        timestamps = pd.to_datetime(df[config.dataset.time_column]).values

    split_result = build_splits(
        n_samples=n_samples,
        labels=labels,
        config=config,
        groups=groups,
        timestamps=timestamps,
    )

    dataset_id = dataset_path.stem

    manifest = DatasetManifest(
        dataset_id=dataset_id,
        modality=modality,
        task_type=task_type,
        target_column=target_col,
        feature_columns=schema["feature_columns"],
        split_strategy=split_result.strategy,
        train_size=len(split_result.train_indices),
        val_size=len(split_result.val_indices),
        test_size=len(split_result.test_indices),
        class_labels=schema["class_labels"],
        group_column=config.dataset.group_column,
        time_column=config.dataset.time_column,
        metadata_columns=config.dataset.metadata_columns,
        warnings=split_result.warnings,
    )

    data = {
        "df": df,
        "features": schema["feature_columns"],
        "target": target_col,
        "split": split_result,
    }

    return IntakeResult(manifest=manifest, data=data, split_result=split_result)


def _intake_image(config: ToolkitConfig, modality: ModalityType) -> IntakeResult:
    """Intake pipeline for image folder classification data."""
    dataset_path = Path(config.dataset.path)
    schema = validate_image_folder_schema(dataset_path)

    task_type = (
        TaskType.BINARY if len(schema["class_labels"]) == 2 else TaskType.MULTICLASS
    )

    # Build a flat list of (image_path, label) for splitting
    image_paths: list[Path] = []
    labels_list: list[str] = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        for img in sorted(class_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in image_extensions:
                image_paths.append(img)
                labels_list.append(class_dir.name)

    labels = np.array(labels_list)
    n_samples = len(labels)

    split_result = build_splits(
        n_samples=n_samples,
        labels=labels,
        config=config,
    )

    manifest = DatasetManifest(
        dataset_id=dataset_path.name,
        modality=modality,
        task_type=task_type,
        split_strategy=split_result.strategy,
        train_size=len(split_result.train_indices),
        val_size=len(split_result.val_indices),
        test_size=len(split_result.test_indices),
        class_labels=schema["class_labels"],
        warnings=split_result.warnings,
    )

    data = {
        "image_paths": image_paths,
        "labels": labels,
        "split": split_result,
    }

    return IntakeResult(manifest=manifest, data=data, split_result=split_result)


def _intake_embedding(config: ToolkitConfig, modality: ModalityType) -> IntakeResult:
    """Intake pipeline for embedding matrix + labels."""
    dataset_path = Path(config.dataset.path)
    if not dataset_path.exists():
        raise SchemaValidationError(f"Embedding file not found: {dataset_path}")

    if dataset_path.suffix == ".npz":
        loaded = np.load(dataset_path)
        if "embeddings" not in loaded or "labels" not in loaded:
            raise SchemaValidationError(
                "NPZ file must contain 'embeddings' and 'labels' arrays. "
                f"Found keys: {list(loaded.keys())}"
            )
        embeddings = loaded["embeddings"]
        labels = loaded["labels"]
    elif dataset_path.suffix == ".npy":
        raise SchemaValidationError(
            "For embedding input, use .npz format with 'embeddings' and 'labels' keys. "
            "Single .npy files are not supported."
        )
    else:
        raise SchemaValidationError(
            f"Unsupported embedding file format: {dataset_path.suffix}"
        )

    schema = validate_embedding_schema(embeddings, labels)
    task_type = detect_task_type_from_array(labels)

    split_result = build_splits(
        n_samples=schema["total_samples"],
        labels=labels,
        config=config,
    )

    manifest = DatasetManifest(
        dataset_id=dataset_path.stem,
        modality=modality,
        task_type=task_type,
        split_strategy=split_result.strategy,
        train_size=len(split_result.train_indices),
        val_size=len(split_result.val_indices),
        test_size=len(split_result.test_indices),
        class_labels=schema["class_labels"],
        warnings=split_result.warnings,
    )

    data = {
        "embeddings": embeddings,
        "labels": labels,
        "split": split_result,
    }

    return IntakeResult(manifest=manifest, data=data, split_result=split_result)
