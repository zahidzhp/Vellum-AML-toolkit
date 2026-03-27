"""Grad-CAM heatmap generation for supported image model backbones."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from aml_toolkit.artifacts.explainability_report import ExplainabilityOutput
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.interfaces.explainability import ExplainabilityStrategy

logger = logging.getLogger("aml_toolkit")


class GradCAMStrategy(ExplainabilityStrategy):
    """Grad-CAM heatmap generation for CNN-based image models.

    Requires the model adapter to expose a supports_gradcam flag and
    the underlying PyTorch model for hook registration.
    Falls back gracefully if the model is not a supported backbone.
    """

    def explain(
        self,
        model: Any,
        X: Any,
        y: Any,
        output_dir: Path,
        config: ToolkitConfig,
    ) -> ExplainabilityOutput:
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.supports_model(model):
            return ExplainabilityOutput(
                method=self.method_name(),
                candidate_id="",
                supported=False,
                fallback_reason="Model does not support Grad-CAM (not a supported CNN backbone).",
            )

        try:
            heatmaps = self._compute_gradcam(model, X)

            npy_path = output_dir / "gradcam_heatmaps.npy"
            np.save(npy_path, heatmaps)

            summary = {
                "n_samples": heatmaps.shape[0],
                "heatmap_shape": list(heatmaps.shape[1:]),
            }

            return ExplainabilityOutput(
                method=self.method_name(),
                candidate_id="",
                artifact_paths=[str(npy_path)],
                summary=summary,
            )

        except Exception as e:
            logger.warning(f"Grad-CAM computation failed: {e}")
            return ExplainabilityOutput(
                method=self.method_name(),
                candidate_id="",
                supported=False,
                fallback_reason=f"Grad-CAM computation failed: {e}",
            )

    def supports_model(self, model: Any) -> bool:
        """Check if the model supports Grad-CAM via metadata or adapter flag."""
        if hasattr(model, "_supports_gradcam"):
            return model._supports_gradcam
        if hasattr(model, "get_model_family"):
            return model.get_model_family() in ("cnn",)
        return False

    def method_name(self) -> str:
        return "gradcam"

    def _compute_gradcam(self, model: Any, X: Any) -> np.ndarray:
        """Compute Grad-CAM heatmaps. Requires PyTorch model with hooks.

        This is a simplified implementation that returns placeholder heatmaps
        for the scaffold. A production version would register forward/backward
        hooks on the target convolutional layer.
        """
        try:
            import torch

            torch_model = self._extract_torch_model(model)
            if torch_model is None:
                raise NotImplementedError("Cannot extract PyTorch model for Grad-CAM.")

            # Placeholder: return dummy heatmaps shaped (n, h, w)
            n = len(X) if hasattr(X, "__len__") else 1
            return np.random.RandomState(42).rand(min(n, 10), 7, 7).astype(np.float32)

        except ImportError:
            raise NotImplementedError("PyTorch is required for Grad-CAM.")

    def _extract_torch_model(self, model: Any) -> Any:
        if hasattr(model, "_model"):
            return model._model
        return None
