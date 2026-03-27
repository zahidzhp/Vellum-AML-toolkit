"""Leakage detection: duplicate overlap, grouped/entity leakage, temporal leakage, class absence."""

from typing import Any

import numpy as np
import pandas as pd

from aml_toolkit.intake.split_builder import SplitResult


def check_duplicate_overlap(
    data: Any,
    split: SplitResult,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Check for exact duplicate rows appearing across splits.

    For tabular data, compares feature rows between train/test and train/val.
    For embeddings, compares embedding vectors directly.

    Args:
        data: Dict with 'df' key (tabular) or 'embeddings' key.
        split: The SplitResult with index arrays.
        feature_columns: Feature columns to compare (tabular only).

    Returns:
        Dict with 'train_test_overlap', 'train_val_overlap' counts and 'details'.
    """
    result: dict[str, Any] = {
        "train_test_overlap": 0,
        "train_val_overlap": 0,
        "details": [],
    }

    if "df" in data and feature_columns:
        df = data["df"]
        train_features = df.iloc[split.train_indices][feature_columns]
        val_features = df.iloc[split.val_indices][feature_columns]
        test_features = df.iloc[split.test_indices][feature_columns]

        # Convert to tuples of values for set-based comparison
        train_set = set(map(tuple, train_features.values.tolist()))
        val_set = set(map(tuple, val_features.values.tolist()))
        test_set = set(map(tuple, test_features.values.tolist()))

        train_test = len(train_set & test_set)
        train_val = len(train_set & val_set)

        result["train_test_overlap"] = train_test
        result["train_val_overlap"] = train_val
        if train_test > 0:
            result["details"].append(
                f"{train_test} duplicate feature rows found in both train and test."
            )
        if train_val > 0:
            result["details"].append(
                f"{train_val} duplicate feature rows found in both train and val."
            )

    elif "embeddings" in data:
        emb = data["embeddings"]
        train_emb = emb[split.train_indices]
        val_emb = emb[split.val_indices]
        test_emb = emb[split.test_indices]

        # Use hashing for efficient comparison
        train_hashes = set(map(lambda r: r.tobytes(), train_emb))
        val_hashes = set(map(lambda r: r.tobytes(), val_emb))
        test_hashes = set(map(lambda r: r.tobytes(), test_emb))

        train_test = len(train_hashes & test_hashes)
        train_val = len(train_hashes & val_hashes)

        result["train_test_overlap"] = train_test
        result["train_val_overlap"] = train_val
        if train_test > 0:
            result["details"].append(
                f"{train_test} duplicate embeddings found in both train and test."
            )
        if train_val > 0:
            result["details"].append(
                f"{train_val} duplicate embeddings found in both train and val."
            )

    return result


def check_grouped_leakage(
    groups: np.ndarray,
    split: SplitResult,
) -> dict[str, Any]:
    """Check if any group ID appears in more than one split.

    Args:
        groups: Array of group IDs (e.g., patient IDs).
        split: The SplitResult with index arrays.

    Returns:
        Dict with 'leaked_groups_train_test', 'leaked_groups_train_val', and 'leaked_groups'.
    """
    train_groups = set(groups[split.train_indices].tolist())
    val_groups = set(groups[split.val_indices].tolist())
    test_groups = set(groups[split.test_indices].tolist())

    train_test_leak = train_groups & test_groups
    train_val_leak = train_groups & val_groups
    val_test_leak = val_groups & test_groups

    all_leaked = train_test_leak | train_val_leak | val_test_leak

    return {
        "leaked_groups_train_test": sorted(str(g) for g in train_test_leak),
        "leaked_groups_train_val": sorted(str(g) for g in train_val_leak),
        "leaked_groups_val_test": sorted(str(g) for g in val_test_leak),
        "leaked_groups": sorted(str(g) for g in all_leaked),
        "total_leaked_groups": len(all_leaked),
    }


def check_temporal_leakage(
    timestamps: np.ndarray,
    split: SplitResult,
) -> dict[str, Any]:
    """Check if temporal ordering is violated across splits.

    Temporal integrity means: max(train timestamps) <= min(val timestamps)
    and max(val timestamps) <= min(test timestamps).

    Args:
        timestamps: Array of timestamps (sortable).
        split: The SplitResult with index arrays.

    Returns:
        Dict with 'train_val_leakage', 'val_test_leakage', and details.
    """
    train_ts = timestamps[split.train_indices]
    val_ts = timestamps[split.val_indices]
    test_ts = timestamps[split.test_indices]

    result: dict[str, Any] = {
        "train_val_leakage": False,
        "val_test_leakage": False,
        "details": [],
    }

    if len(val_ts) > 0 and len(train_ts) > 0:
        if np.max(train_ts) > np.min(val_ts):
            result["train_val_leakage"] = True
            result["details"].append(
                f"Temporal leakage: max train timestamp ({np.max(train_ts)}) > "
                f"min val timestamp ({np.min(val_ts)})."
            )

    if len(test_ts) > 0 and len(val_ts) > 0:
        if np.max(val_ts) > np.min(test_ts):
            result["val_test_leakage"] = True
            result["details"].append(
                f"Temporal leakage: max val timestamp ({np.max(val_ts)}) > "
                f"min test timestamp ({np.min(test_ts)})."
            )

    return result


def check_class_absence(
    labels: np.ndarray,
    split: SplitResult,
    class_labels: list[str],
) -> dict[str, Any]:
    """Check if any class is completely absent from a split.

    Args:
        labels: Full label array.
        split: The SplitResult with index arrays.
        class_labels: List of expected class labels.

    Returns:
        Dict with 'absent_in_train', 'absent_in_val', 'absent_in_test'.
    """
    label_set = set(str(c) for c in class_labels)

    train_labels = set(str(l) for l in labels[split.train_indices])
    val_labels = set(str(l) for l in labels[split.val_indices])
    test_labels = set(str(l) for l in labels[split.test_indices])

    return {
        "absent_in_train": sorted(label_set - train_labels),
        "absent_in_val": sorted(label_set - val_labels),
        "absent_in_test": sorted(label_set - test_labels),
    }
