"""Interface for data intervention strategies."""

from abc import ABC, abstractmethod
from typing import Any

from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import InterventionType


class Intervention(ABC):
    """Abstract contract for data intervention strategies.

    Interventions transform training data or model behavior to address
    issues like class imbalance (weighting, resampling, augmentation, etc.).
    """

    @abstractmethod
    def apply(
        self,
        X_train: Any,
        y_train: Any,
        config: ToolkitConfig,
    ) -> tuple[Any, Any]:
        """Apply the intervention to training data.

        Args:
            X_train: Training features.
            y_train: Training labels.
            config: Toolkit configuration.

        Returns:
            Tuple of (transformed X_train, transformed y_train).
        """

    @abstractmethod
    def intervention_type(self) -> InterventionType:
        """Return the intervention type enum value."""

    @abstractmethod
    def is_applicable(self, config: ToolkitConfig) -> bool:
        """Check whether this intervention is allowed by the current config.

        Args:
            config: Toolkit configuration.

        Returns:
            True if the intervention is permitted.
        """
