"""Interface for explainability strategies."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from aml_toolkit.artifacts.explainability_report import ExplainabilityOutput
from aml_toolkit.core.config import ToolkitConfig


class ExplainabilityStrategy(ABC):
    """Abstract contract for explainability methods.

    Implementations produce interpretable outputs (feature importance,
    SHAP values, Grad-CAM heatmaps, etc.) and assess faithfulness
    where supported.
    """

    @abstractmethod
    def explain(
        self,
        model: Any,
        X: Any,
        y: Any,
        output_dir: Path,
        config: ToolkitConfig,
    ) -> ExplainabilityOutput:
        """Generate explainability artifacts for the given model and data.

        Args:
            model: The trained model (adapter or raw model object).
            X: Input data for explanation.
            y: True labels.
            output_dir: Directory to save artifact files (heatmaps, plots).
            config: Toolkit configuration.

        Returns:
            ExplainabilityOutput with paths to generated artifacts and scores.
        """

    @abstractmethod
    def supports_model(self, model: Any) -> bool:
        """Check whether this explainability method supports the given model.

        Args:
            model: The model to check.

        Returns:
            True if the method can produce explanations for this model.
        """

    @abstractmethod
    def method_name(self) -> str:
        """Return the explainability method identifier."""
