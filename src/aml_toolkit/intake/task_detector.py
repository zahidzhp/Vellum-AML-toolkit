"""Task type detection: binary, multiclass, or multilabel."""

import numpy as np
import pandas as pd

from aml_toolkit.core.enums import TaskType


def detect_task_type_from_series(labels: pd.Series) -> TaskType:
    """Detect task type from a pandas Series of labels.

    Args:
        labels: Target column values.

    Returns:
        Detected TaskType.
    """
    unique_labels = labels.dropna().unique()
    n_classes = len(unique_labels)

    if n_classes <= 1:
        # Degenerate case — treat as binary with a single class
        return TaskType.BINARY
    elif n_classes == 2:
        return TaskType.BINARY
    else:
        return TaskType.MULTICLASS


def detect_task_type_from_array(labels: np.ndarray) -> TaskType:
    """Detect task type from a numpy array of labels.

    Args:
        labels: 1D array (single-label) or 2D binary matrix (multilabel).

    Returns:
        Detected TaskType.
    """
    if labels.ndim == 2:
        # 2D binary matrix → multilabel
        # Verify it looks like a binary indicator matrix
        unique_values = np.unique(labels)
        if set(unique_values.tolist()).issubset({0, 1, 0.0, 1.0}):
            return TaskType.MULTILABEL
        # If not binary indicators, treat each row as a multiclass encoding
        return TaskType.MULTILABEL

    # 1D array
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    if n_classes <= 2:
        return TaskType.BINARY
    else:
        return TaskType.MULTICLASS
