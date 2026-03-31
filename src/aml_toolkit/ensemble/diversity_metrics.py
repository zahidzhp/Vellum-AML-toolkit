"""Ensemble diversity metrics — pairwise disagreement and ambiguity decomposition."""

from __future__ import annotations

import numpy as np


def ensemble_diversity_score(predictions: list[np.ndarray]) -> float:
    """Mean pairwise disagreement rate across all model pairs.

    Range: 0.0 (all models identical) to 1.0 (all models completely different).

    Args:
        predictions: List of hard-prediction arrays, each shape (n,).

    Returns:
        Mean pairwise disagreement in [0, 1].
    """
    k = len(predictions)
    if k < 2:
        return 0.0

    disagreements = []
    for i in range(k):
        for j in range(i + 1, k):
            preds_i = np.asarray(predictions[i])
            preds_j = np.asarray(predictions[j])
            disagreements.append(float(np.mean(preds_i != preds_j)))

    return float(np.mean(disagreements))


def pairwise_disagreement(preds_a: np.ndarray, preds_b: np.ndarray) -> float:
    """Fraction of samples where two models disagree on hard prediction.

    Args:
        preds_a: Hard predictions from model A, shape (n,).
        preds_b: Hard predictions from model B, shape (n,).

    Returns:
        Disagreement rate in [0, 1].
    """
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    return float(np.mean(preds_a != preds_b))


def ambiguity_decomposition(
    probas: list[np.ndarray],
    y_true: np.ndarray,
) -> dict[str, float]:
    """Bias-variance-diversity decomposition of ensemble error (Krogh & Vedelsby 1995).

    For regression, Ambiguity Decomposition gives:
        ensemble_error = bias + variance - diversity

    For classification, we use the probabilistic generalization:
        ensemble_loss ≈ mean_individual_loss - ambiguity (diversity)

    Args:
        probas: List of probability arrays (n, K), one per model.
        y_true: True integer class labels, shape (n,).

    Returns:
        Dict with keys: "bias", "variance", "diversity", "error".
        All values are in [0, 1].
    """
    if not probas:
        return {"bias": 0.0, "variance": 0.0, "diversity": 0.0, "error": 0.0}

    probas_arr = [np.asarray(p, dtype=np.float64) for p in probas]
    y_true = np.asarray(y_true, dtype=np.int64)
    n = len(y_true)
    K = probas_arr[0].shape[1] if probas_arr[0].ndim == 2 else 2

    # Ensemble average probabilities
    ensemble_proba = np.mean(probas_arr, axis=0)  # (n, K)

    # Ensemble error (using 0/1 loss on argmax)
    ensemble_preds = np.argmax(ensemble_proba, axis=1)
    error = float(np.mean(ensemble_preds != y_true))

    # Mean individual error
    individual_errors = []
    for p in probas_arr:
        preds = np.argmax(p, axis=1)
        individual_errors.append(float(np.mean(preds != y_true)))
    mean_individual_error = float(np.mean(individual_errors))

    # Ambiguity (diversity contribution): mean_individual_error - ensemble_error
    # This is the core insight: diversity reduces ensemble error
    diversity = max(0.0, mean_individual_error - error)

    # Bias: best individual error as proxy (lower bound on achievable error)
    bias = float(np.min(individual_errors))

    # Variance: additional error beyond best individual
    variance = max(0.0, error - bias)

    return {
        "bias": bias,
        "variance": variance,
        "diversity": diversity,
        "error": error,
        "mean_individual_error": mean_individual_error,
    }
