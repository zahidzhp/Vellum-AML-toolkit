"""Isotonic regression calibrator."""

import logging

import numpy as np
from sklearn.isotonic import IsotonicRegression

from aml_toolkit.artifacts import CalibrationResult
from aml_toolkit.calibration.metrics import brier_score, expected_calibration_error
from aml_toolkit.interfaces.calibrator import Calibrator

logger = logging.getLogger("aml_toolkit")


class IsotonicCalibrator(Calibrator):
    """Non-parametric calibration using isotonic regression.

    Fits a monotonic step function that maps raw probabilities to
    calibrated probabilities.
    """

    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> None:
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._iso.fit(probabilities, y_true)
        logger.info("Isotonic calibrator fitted.")

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("IsotonicCalibrator has not been fitted.")
        return self._iso.predict(probabilities)

    def evaluate(
        self,
        probabilities_before: np.ndarray,
        probabilities_after: np.ndarray,
        y_true: np.ndarray,
    ) -> CalibrationResult:
        return CalibrationResult(
            candidate_id="",
            method=self.method_name(),
            ece_before=expected_calibration_error(y_true, probabilities_before),
            ece_after=expected_calibration_error(y_true, probabilities_after),
            brier_before=brier_score(y_true, probabilities_before),
            brier_after=brier_score(y_true, probabilities_after),
        )

    def method_name(self) -> str:
        return "isotonic"
