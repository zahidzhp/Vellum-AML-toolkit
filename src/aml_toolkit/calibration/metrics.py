"""Calibration quality metrics: Expected Calibration Error (ECE) and Brier Score."""

import numpy as np


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions by confidence, then measures the weighted average gap
    between predicted confidence and observed accuracy per bin.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of equal-width bins.

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if lo == 0.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += (n_in_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier Score (mean squared error of probabilities).

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Brier score (lower is better, 0 = perfect).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        return 0.0

    return float(np.mean((y_prob - y_true) ** 2))
