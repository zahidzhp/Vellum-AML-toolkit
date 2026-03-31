"""Split conformal prediction for classification (LAC algorithm, numpy-only).

Provides prediction sets C(x) with a mathematical coverage guarantee:
    P(y ∈ C(x)) ≥ 1 - α

Reference: Angelopoulos & Bates (2022) "A Gentle Introduction to Conformal Prediction"
Algorithm: LAC (Least Ambiguous set-valued Classifier), a simplified form of RAPS.
No new dependencies — uses numpy only.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("aml_toolkit")


class SplitConformalPredictor:
    """Split conformal prediction for classification.

    Usage:
        predictor = SplitConformalPredictor(coverage=0.9)
        predictor.fit(proba_cal, y_cal)   # calibrate threshold on held-out set
        sets = predictor.predict_sets(proba_test)  # get prediction sets
        eff = predictor.efficiency(proba_test)     # mean set size

    The predictor works for both binary (proba shape (n,2)) and multiclass
    (proba shape (n, K)) inputs. For binary, you can pass either shape.
    """

    def __init__(self, coverage: float = 0.9):
        """
        Args:
            coverage: Desired coverage level (1 - α). E.g. 0.9 for 90% coverage.
        """
        if not (0.0 < coverage < 1.0):
            raise ValueError(f"coverage must be in (0, 1), got {coverage}")
        self.coverage = coverage
        self._threshold: float | None = None
        self._n_classes: int | None = None

    def fit(self, proba_cal: np.ndarray, y_cal: np.ndarray) -> "SplitConformalPredictor":
        """Compute conformal threshold on a held-out calibration set.

        Args:
            proba_cal: Predicted probabilities, shape (n, K) for K classes.
                       For binary, shape (n, 2) or (n,) for positive-class proba.
            y_cal: True integer labels, shape (n,).

        Returns:
            self (for chaining)
        """
        proba_cal = np.asarray(proba_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal, dtype=np.int64)

        proba_cal = self._ensure_2d(proba_cal)
        self._n_classes = proba_cal.shape[1]
        n = len(y_cal)

        if n == 0:
            raise ValueError("Calibration set is empty.")

        # Nonconformity score: 1 - P(true class)
        true_class_proba = proba_cal[np.arange(n), y_cal]
        scores = 1.0 - true_class_proba

        # Adjusted quantile for finite-sample valid coverage
        # q_hat = ceil((n+1)(1-α)) / n  (Theorem 1 in Angelopoulos & Bates 2022)
        alpha = 1.0 - self.coverage
        q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
        q_level = min(q_level, 1.0)
        self._threshold = float(np.quantile(scores, q_level))
        return self

    def predict_sets(self, proba: np.ndarray) -> list[list[int]]:
        """Return prediction sets for each sample.

        A prediction set C(x) contains all classes k where
        (1 - P(k|x)) ≤ threshold. Smaller sets = more confident.

        Args:
            proba: Predicted probabilities, shape (n, K) or (n,).

        Returns:
            List of lists; each inner list contains the class indices included.
        """
        if self._threshold is None:
            raise RuntimeError("Call fit() before predict_sets().")

        proba = np.asarray(proba, dtype=np.float64)
        proba = self._ensure_2d(proba)
        sets = []
        for row in proba:
            included = [int(i) for i, p in enumerate(row) if (1.0 - p) <= self._threshold]
            if not included:
                # Always include the argmax to avoid empty sets
                included = [int(np.argmax(row))]
            sets.append(included)
        return sets

    def efficiency(self, proba: np.ndarray) -> float:
        """Mean prediction set size across samples.

        Lower = more efficient/confident model. A value close to 1.0 means
        the model is making mostly singleton predictions (high confidence).
        """
        sets = self.predict_sets(proba)
        return float(np.mean([len(s) for s in sets]))

    def empirical_coverage(
        self, proba: np.ndarray, y_true: np.ndarray
    ) -> float:
        """Fraction of samples where true label is in the prediction set."""
        sets = self.predict_sets(proba)
        y_true = np.asarray(y_true, dtype=np.int64)
        covered = sum(1 for s, y in zip(sets, y_true) if int(y) in s)
        return covered / len(y_true) if len(y_true) > 0 else 0.0

    def _ensure_2d(self, proba: np.ndarray) -> np.ndarray:
        """Convert (n,) binary proba to (n, 2) shape."""
        if proba.ndim == 1:
            # Assume positive-class probabilities for binary case
            return np.stack([1.0 - proba, proba], axis=1)
        return proba
