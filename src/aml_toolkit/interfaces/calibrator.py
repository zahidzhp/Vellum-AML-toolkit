"""Interface for probability calibration methods."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from aml_toolkit.artifacts import CalibrationResult
from aml_toolkit.core.config import ToolkitConfig


class Calibrator(ABC):
    """Abstract contract for probability calibration methods.

    Calibrators adjust raw model probability outputs to better reflect
    true class probabilities.
    """

    @abstractmethod
    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> None:
        """Fit the calibrator on validation predictions and true labels.

        Args:
            probabilities: Raw model probability outputs.
            y_true: True labels.
        """

    @abstractmethod
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration to probability outputs.

        Args:
            probabilities: Raw model probability outputs.

        Returns:
            Calibrated probability outputs.
        """

    @abstractmethod
    def evaluate(
        self,
        probabilities_before: np.ndarray,
        probabilities_after: np.ndarray,
        y_true: np.ndarray,
    ) -> CalibrationResult:
        """Evaluate calibration quality before and after.

        Args:
            probabilities_before: Uncalibrated probabilities.
            probabilities_after: Calibrated probabilities.
            y_true: True labels.

        Returns:
            CalibrationResult with before/after metrics.
        """

    @abstractmethod
    def method_name(self) -> str:
        """Return the calibration method identifier."""
