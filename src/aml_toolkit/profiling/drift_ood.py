"""Basic train-vs-test distribution shift / OOD detection (FR-131)."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def profile_drift(
    df: pd.DataFrame,
    feature_columns: list[str],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    p_value_threshold: float = 0.01,
) -> dict[str, Any]:
    """Compare feature distributions between train and test splits.

    Uses Kolmogorov-Smirnov test for numeric features and chi-squared test
    for categorical features. Flags features where the distributions
    differ significantly.

    Args:
        df: The tabular DataFrame.
        feature_columns: Feature columns to compare.
        train_indices: Training set indices.
        test_indices: Test set indices.
        p_value_threshold: P-value threshold for flagging drift.

    Returns:
        Dict with shifted_features, shift_details, and summary.
    """
    if len(train_indices) == 0 or len(test_indices) == 0:
        return {
            "shifted_features": [],
            "shift_details": {},
            "total_shifted": 0,
        }

    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    shifted_features: list[str] = []
    shift_details: dict[str, dict[str, Any]] = {}

    for col in feature_columns:
        if col not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            train_vals = train_df[col].dropna().values
            test_vals = test_df[col].dropna().values
            if len(train_vals) < 5 or len(test_vals) < 5:
                continue
            stat, p_value = stats.ks_2samp(train_vals, test_vals)
            if p_value < p_value_threshold:
                shifted_features.append(col)
                shift_details[col] = {
                    "test": "ks_2samp",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                }
        else:
            # Categorical: chi-squared test
            train_counts = train_df[col].value_counts()
            test_counts = test_df[col].value_counts()
            all_cats = set(train_counts.index) | set(test_counts.index)
            if len(all_cats) < 2:
                continue
            train_freq = np.array([train_counts.get(c, 0) for c in all_cats])
            test_freq = np.array([test_counts.get(c, 0) for c in all_cats])
            # Normalize to expected proportions
            total_train = train_freq.sum()
            total_test = test_freq.sum()
            if total_train == 0 or total_test == 0:
                continue
            expected = train_freq * (total_test / total_train)
            expected = np.where(expected == 0, 1e-10, expected)
            try:
                stat, p_value = stats.chisquare(test_freq, f_exp=expected)
                if p_value < p_value_threshold:
                    shifted_features.append(col)
                    shift_details[col] = {
                        "test": "chi_squared",
                        "statistic": float(stat),
                        "p_value": float(p_value),
                    }
            except ValueError:
                continue

    return {
        "shifted_features": shifted_features,
        "shift_details": shift_details,
        "total_shifted": len(shifted_features),
    }
