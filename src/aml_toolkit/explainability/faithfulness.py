"""Faithfulness metric helper: measures how well explanations reflect true model behavior."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("aml_toolkit")


def feature_removal_faithfulness(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    importances: np.ndarray,
    top_k: int = 5,
    metric_fn: Any = None,
) -> float:
    """Compute faithfulness by removing top-K important features and measuring degradation.

    A faithful explanation should cause significant performance drop when the
    most important features are removed (zeroed out).

    Args:
        model: Trained model with a predict method.
        X: Input features.
        y: True labels.
        importances: Feature importance array (higher = more important).
        top_k: Number of top features to remove.
        metric_fn: Callable(y_true, y_pred) -> float. Defaults to accuracy.

    Returns:
        Faithfulness score: (original_score - degraded_score). Higher = more faithful.
        Returns 0.0 if computation fails.
    """
    if metric_fn is None:
        from sklearn.metrics import accuracy_score
        metric_fn = accuracy_score

    try:
        # Original score
        original_preds = model.predict(X)
        original_score = float(metric_fn(y, original_preds))

        # Remove top-K features (zero them out)
        top_indices = np.argsort(np.abs(importances))[::-1][:top_k]
        X_degraded = X.copy()
        X_degraded[:, top_indices] = 0.0

        degraded_preds = model.predict(X_degraded)
        degraded_score = float(metric_fn(y, degraded_preds))

        faithfulness = original_score - degraded_score
        logger.info(
            f"Faithfulness check: original={original_score:.4f}, "
            f"degraded={degraded_score:.4f}, faithfulness={faithfulness:.4f}"
        )
        return faithfulness

    except Exception as e:
        logger.warning(f"Faithfulness computation failed: {e}")
        return 0.0
