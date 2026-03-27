"""Outlier summary for numeric features using IQR method."""

from typing import Any

import numpy as np
import pandas as pd


def profile_outliers(
    df: pd.DataFrame,
    feature_columns: list[str],
    iqr_multiplier: float = 1.5,
) -> dict[str, Any]:
    """Summarize outlier counts per numeric feature using IQR method.

    Args:
        df: The tabular DataFrame.
        feature_columns: Feature columns to check.
        iqr_multiplier: Multiplier for IQR bounds (default 1.5).

    Returns:
        Dict with per-column outlier counts and total outlier fraction.
    """
    numeric_cols = [
        c for c in feature_columns
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not numeric_cols or len(df) == 0:
        return {
            "per_column": {},
            "total_outlier_rows": 0,
            "total_outlier_fraction": 0.0,
        }

    per_column: dict[str, int] = {}
    outlier_row_mask = pd.Series(False, index=df.index)

    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) < 4:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        col_outliers = (df[col] < lower) | (df[col] > upper)
        count = int(col_outliers.sum())
        if count > 0:
            per_column[col] = count
            outlier_row_mask |= col_outliers

    total_outlier_rows = int(outlier_row_mask.sum())
    total_outlier_fraction = total_outlier_rows / len(df) if len(df) > 0 else 0.0

    return {
        "per_column": per_column,
        "total_outlier_rows": total_outlier_rows,
        "total_outlier_fraction": total_outlier_fraction,
    }
