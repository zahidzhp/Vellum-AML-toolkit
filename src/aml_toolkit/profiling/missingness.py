"""Missingness profiling for tabular data."""

import pandas as pd


def profile_missingness(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float]:
    """Compute per-column missing value ratios.

    Args:
        df: The tabular DataFrame.
        feature_columns: Feature columns to check.

    Returns:
        Dict mapping column name to fraction of missing values (0.0 to 1.0).
        Only columns with any missingness are included.
    """
    n_rows = len(df)
    if n_rows == 0:
        return {}

    summary: dict[str, float] = {}
    for col in feature_columns:
        if col in df.columns:
            missing_count = int(df[col].isna().sum())
            if missing_count > 0:
                summary[col] = missing_count / n_rows

    return summary
