"""Temperature scaling calibrator for neural and probabilistic models."""

import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from aml_toolkit.artifacts import CalibrationResult
from aml_toolkit.calibration.metrics import brier_score, expected_calibration_error
from aml_toolkit.interfaces.calibrator import Calibrator

logger = logging.getLogger("aml_toolkit")

# Clamp bounds to avoid log(0) or log(inf)
_EPS = 1e-7


class TemperatureScalingCalibrator(Calibrator):
    """Platt-style temperature scaling on logits.

    Learns a single scalar T that rescales logits: calibrated = sigmoid(logit(p) / T).
    Optimizes for negative log-likelihood on the validation set.
    """

    def __init__(self) -> None:
        self._temperature: float = 1.0

    @property
    def temperature(self) -> float:
        return self._temperature

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> None:
        """Fit the temperature parameter on validation data.

        Args:
            probabilities: Raw model probability outputs for positive class.
            y_true: True binary labels.
        """
        probabilities = np.clip(probabilities, _EPS, 1.0 - _EPS)
        logits = logit(probabilities)
        y = np.asarray(y_true, dtype=float)

        def nll(t: float) -> float:
            scaled = expit(logits / t)
            scaled = np.clip(scaled, _EPS, 1.0 - _EPS)
            return float(-np.mean(y * np.log(scaled) + (1 - y) * np.log(1 - scaled)))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self._temperature = float(result.x)
        logger.info(f"Temperature scaling fitted: T={self._temperature:.4f}")

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.clip(probabilities, _EPS, 1.0 - _EPS)
        logits = logit(probabilities)
        return expit(logits / self._temperature)

    def evaluate(
        self,
        probabilities_before: np.ndarray,
        probabilities_after: np.ndarray,
        y_true: np.ndarray,
    ) -> CalibrationResult:
        return CalibrationResult(
            candidate_id="",  # filled by caller
            method=self.method_name(),
            ece_before=expected_calibration_error(y_true, probabilities_before),
            ece_after=expected_calibration_error(y_true, probabilities_after),
            brier_before=brier_score(y_true, probabilities_before),
            brier_after=brier_score(y_true, probabilities_after),
        )

    def method_name(self) -> str:
        return "temperature_scaling"
