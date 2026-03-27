"""Threshold optimizer: finds the optimal decision threshold for binary classification."""

import logging

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger("aml_toolkit")


class ThresholdOptimizer:
    """Finds the threshold that maximizes F1 (or a specified metric) on validation data.

    Searches a grid of thresholds between 0 and 1 to find the point that
    maximizes the chosen metric.
    """

    def __init__(self, metric: str = "f1", n_steps: int = 100) -> None:
        self._metric = metric
        self._n_steps = n_steps
        self._best_threshold: float = 0.5
        self._best_score: float = 0.0

    @property
    def best_threshold(self) -> float:
        return self._best_threshold

    @property
    def best_score(self) -> float:
        return self._best_score

    def optimize(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Find the optimal threshold.

        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities for the positive class.

        Returns:
            The optimal threshold value.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        thresholds = np.linspace(0.01, 0.99, self._n_steps)
        best_score = -1.0
        best_thresh = 0.5

        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            score = self._compute_metric(y_true, preds)
            if score > best_score:
                best_score = score
                best_thresh = t

        self._best_threshold = float(best_thresh)
        self._best_score = float(best_score)

        logger.info(
            f"Threshold optimization: best={self._best_threshold:.4f}, "
            f"{self._metric}={self._best_score:.4f}"
        )
        return self._best_threshold

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self._metric == "f1":
            return float(f1_score(y_true, y_pred, average="binary", zero_division=0))
        if self._metric == "macro_f1":
            return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        raise ValueError(f"Unknown threshold metric: {self._metric}")
