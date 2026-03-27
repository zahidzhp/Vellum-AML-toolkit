"""Duplicate row detection for profiling."""

from typing import Any

import pandas as pd


def profile_duplicates(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    """Summarize exact duplicate rows in the dataset.

    Args:
        df: The tabular DataFrame.
        feature_columns: Feature columns to check for duplicates.

    Returns:
        Dict with total_duplicates, duplicate_fraction, and duplicate_groups count.
    """
    if not feature_columns or len(df) == 0:
        return {
            "total_duplicates": 0,
            "duplicate_fraction": 0.0,
            "duplicate_groups": 0,
        }

    subset = df[feature_columns]
    duplicated_mask = subset.duplicated(keep=False)
    total_duplicates = int(duplicated_mask.sum())
    duplicate_fraction = total_duplicates / len(df) if len(df) > 0 else 0.0

    # Count distinct duplicate groups (sets of identical rows)
    if total_duplicates > 0:
        duplicate_groups = int(subset[duplicated_mask].drop_duplicates().shape[0])
    else:
        duplicate_groups = 0

    return {
        "total_duplicates": total_duplicates,
        "duplicate_fraction": duplicate_fraction,
        "duplicate_groups": duplicate_groups,
    }
