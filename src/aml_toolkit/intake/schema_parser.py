"""Schema validation for different input formats."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.exceptions import SchemaValidationError


def validate_tabular_schema(df: pd.DataFrame, config: ToolkitConfig) -> dict[str, Any]:
    """Validate a tabular CSV DataFrame and extract schema info.

    Args:
        df: The loaded DataFrame.
        config: Toolkit configuration.

    Returns:
        Dict with keys: target_column, feature_columns, class_labels.

    Raises:
        SchemaValidationError: If the schema is invalid.
    """
    target_col = config.dataset.target_column

    if target_col not in df.columns:
        raise SchemaValidationError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    if df.empty:
        raise SchemaValidationError("Dataset is empty (0 rows).")

    if df[target_col].isna().all():
        raise SchemaValidationError(
            f"Target column '{target_col}' contains only missing values."
        )

    exclude_cols = {target_col}
    if config.dataset.group_column:
        exclude_cols.add(config.dataset.group_column)
    if config.dataset.time_column:
        exclude_cols.add(config.dataset.time_column)
    for col in config.dataset.metadata_columns:
        exclude_cols.add(col)

    feature_columns = [c for c in df.columns if c not in exclude_cols]
    if not feature_columns:
        raise SchemaValidationError(
            "No feature columns remaining after excluding target, group, time, "
            "and metadata columns."
        )

    class_labels = sorted(df[target_col].dropna().unique().astype(str).tolist())

    return {
        "target_column": target_col,
        "feature_columns": feature_columns,
        "class_labels": class_labels,
    }


def validate_image_folder_schema(dataset_path: Path) -> dict[str, Any]:
    """Validate an image folder classification structure.

    Expected layout:
        dataset_path/
            class_a/
                img1.jpg
                img2.png
            class_b/
                img3.jpg

    Args:
        dataset_path: Root directory of the image dataset.

    Returns:
        Dict with keys: class_labels, samples_per_class, total_samples.

    Raises:
        SchemaValidationError: If the folder structure is invalid.
    """
    if not dataset_path.is_dir():
        raise SchemaValidationError(
            f"Image dataset path does not exist or is not a directory: {dataset_path}"
        )

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    class_dirs = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if len(class_dirs) < 2:
        raise SchemaValidationError(
            f"Image folder must contain at least 2 class subdirectories, "
            f"found {len(class_dirs)} in {dataset_path}"
        )

    class_labels = []
    samples_per_class: dict[str, int] = {}
    total = 0

    for class_dir in class_dirs:
        label = class_dir.name
        images = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        if not images:
            raise SchemaValidationError(
                f"Class directory '{label}' contains no recognized image files."
            )
        class_labels.append(label)
        samples_per_class[label] = len(images)
        total += len(images)

    return {
        "class_labels": class_labels,
        "samples_per_class": samples_per_class,
        "total_samples": total,
    }


def validate_embedding_schema(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Validate an embedding matrix + labels input.

    Args:
        embeddings: 2D numpy array of shape (n_samples, n_features).
        labels: 1D or 2D numpy array of labels.

    Returns:
        Dict with keys: class_labels, total_samples, embedding_dim.

    Raises:
        SchemaValidationError: If shapes are inconsistent.
    """
    if embeddings.ndim != 2:
        raise SchemaValidationError(
            f"Embeddings must be 2D, got shape {embeddings.shape}"
        )

    if labels.ndim == 1:
        if len(labels) != embeddings.shape[0]:
            raise SchemaValidationError(
                f"Label count ({len(labels)}) does not match "
                f"embedding count ({embeddings.shape[0]})."
            )
        class_labels = sorted(np.unique(labels).astype(str).tolist())
    elif labels.ndim == 2:
        # Multilabel: each row is a binary vector
        if labels.shape[0] != embeddings.shape[0]:
            raise SchemaValidationError(
                f"Label rows ({labels.shape[0]}) do not match "
                f"embedding count ({embeddings.shape[0]})."
            )
        class_labels = [str(i) for i in range(labels.shape[1])]
    else:
        raise SchemaValidationError(
            f"Labels must be 1D or 2D, got shape {labels.shape}"
        )

    return {
        "class_labels": class_labels,
        "total_samples": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
    }
