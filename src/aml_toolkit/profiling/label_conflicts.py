"""Label conflict detection: identical inputs with different labels."""

from typing import Any

import pandas as pd


def detect_label_conflicts(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> dict[str, Any]:
    """Detect rows with identical feature values but conflicting labels.

    Args:
        df: The tabular DataFrame.
        feature_columns: Feature columns to group by.
        target_column: The label column.

    Returns:
        Dict with conflict_count, conflict_fraction, and sample conflicts.
    """
    if not feature_columns or len(df) == 0 or target_column not in df.columns:
        return {
            "conflict_count": 0,
            "conflict_fraction": 0.0,
            "conflict_groups": 0,
            "sample_conflicts": [],
        }

    # Group by features and count unique labels per group
    grouped = df.groupby(feature_columns, dropna=False)[target_column].nunique()
    conflict_groups = grouped[grouped > 1]

    conflict_group_count = len(conflict_groups)

    # Count total rows involved in conflicts
    if conflict_group_count > 0:
        # Get the feature value combinations that have conflicts
        conflict_keys = conflict_groups.index
        if isinstance(conflict_keys, pd.MultiIndex):
            conflict_mask = df.set_index(feature_columns).index.isin(conflict_keys)
        else:
            conflict_mask = df[feature_columns[0]].isin(conflict_keys)
        conflict_row_count = int(conflict_mask.sum())
    else:
        conflict_row_count = 0

    conflict_fraction = conflict_row_count / len(df) if len(df) > 0 else 0.0

    # Sample a few conflicts for reporting
    sample_conflicts: list[dict[str, Any]] = []
    if conflict_group_count > 0:
        sample_keys = list(conflict_groups.index[:5])
        for key in sample_keys:
            if isinstance(key, tuple):
                mask = pd.Series(True, index=df.index)
                for col, val in zip(feature_columns, key):
                    mask &= df[col] == val
            else:
                mask = df[feature_columns[0]] == key
            labels_found = df.loc[mask, target_column].unique().tolist()
            sample_conflicts.append({
                "features": key if isinstance(key, tuple) else (key,),
                "labels": labels_found,
                "count": int(mask.sum()),
            })

    return {
        "conflict_count": conflict_row_count,
        "conflict_fraction": conflict_fraction,
        "conflict_groups": conflict_group_count,
        "sample_conflicts": sample_conflicts,
    }
