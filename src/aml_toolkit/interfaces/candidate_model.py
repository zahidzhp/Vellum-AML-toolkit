"""Interface for candidate model adapters."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from aml_toolkit.core.config import ToolkitConfig


class CandidateModel(ABC):
    """Abstract contract for candidate model adapters.

    Every model adapter (logistic, RF, XGB, CNN, ViT, etc.) must expose
    this common interface. The adapter wraps library-specific code behind
    a uniform surface.
    """

    @abstractmethod
    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        config: ToolkitConfig,
    ) -> None:
        """Fit the model on training data with validation monitoring.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
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
            Array of probability estimates, or None if not probabilistic.
        """

    @abstractmethod
    def evaluate(self, X: Any, y: Any, metrics: list[str]) -> dict[str, float]:
        """Evaluate the model on given data.

        Args:
            X: Input features.
            y: True labels.
            metrics: Metric names to compute.

        Returns:
            Dict mapping metric name to value.
        """

    @abstractmethod
    def get_training_trace(self) -> dict[str, list[float]]:
        """Return training metric traces for runtime decision monitoring.

        Returns:
            Dict mapping metric name to list of per-epoch values.
        """

    @abstractmethod
    def get_model_family(self) -> str:
        """Return the model family identifier (e.g., 'xgb', 'cnn')."""

    @abstractmethod
    def is_probabilistic(self) -> bool:
        """Return True if predict_proba is supported."""

    @abstractmethod
    def serialize(self, path: Any) -> None:
        """Persist the trained model to disk.

        Args:
            path: Destination path.
        """
