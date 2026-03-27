"""Interface for diagnostic probe models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from aml_toolkit.artifacts import ProbeResult
from aml_toolkit.core.config import ToolkitConfig


class ProbeModel(ABC):
    """Abstract contract for lightweight diagnostic probe models.

    Probes are cheap models used to estimate learnability and intervention
    sensitivity. They are NOT the final training pipeline.
    """

    @abstractmethod
    def fit(self, X_train: Any, y_train: Any, config: ToolkitConfig) -> None:
        """Fit the probe on training data.

        Args:
            X_train: Training features.
            y_train: Training labels.
            config: Toolkit configuration.
        """

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Input features.

        Returns:
            Array of predictions.
        """

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray | None:
        """Generate probability estimates if supported.

        Args:
            X: Input features.

        Returns:
            Array of probability estimates, or None if not supported.
        """

    @abstractmethod
    def evaluate(self, X: Any, y: Any, metrics: list[str]) -> dict[str, float]:
        """Evaluate the probe on given data.

        Args:
            X: Input features.
            y: True labels.
            metrics: List of metric names to compute.

        Returns:
            Dict mapping metric name to value.
        """

    @abstractmethod
    def name(self) -> str:
        """Return the probe model's identifier."""

    @abstractmethod
    def to_probe_result(self, intervention_branch: str = "none") -> ProbeResult:
        """Package the probe's results into a ProbeResult artifact.

        Args:
            intervention_branch: Name of the intervention branch used.

        Returns:
            A ProbeResult with metrics and metadata.
        """
