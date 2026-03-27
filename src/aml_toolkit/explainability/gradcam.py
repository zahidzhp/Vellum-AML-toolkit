"""Grad-CAM heatmap generation for CNN-based image model backbones."""

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

    Registers forward and backward hooks on the last convolutional layer
    to compute class-discriminative heatmaps.
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
        if hasattr(model, "_supports_gradcam") and model._supports_gradcam:
            return True
        if hasattr(model, "get_model_family"):
            return model.get_model_family() in ("cnn",)
        return False

    def method_name(self) -> str:
        return "gradcam"

    def _compute_gradcam(self, model: Any, X: Any) -> np.ndarray:
        """Compute Grad-CAM heatmaps using PyTorch hooks.

        Args:
            model: A CandidateModel adapter with a ._model attribute (PyTorch Module).
            X: Image paths (list of Path) or image tensors.

        Returns:
            Numpy array of shape (n_samples, H, W) with heatmap values in [0, 1].
        """
        import torch
        from aml_toolkit.utils.image_feature_extractor import (
            ImagePathDataset,
            get_eval_transform,
        )

        torch_model = self._extract_torch_model(model)
        if torch_model is None:
            raise NotImplementedError("Cannot extract PyTorch model for Grad-CAM.")

        # Guard against empty input
        if X is None or (hasattr(X, "__len__") and len(X) == 0):
            return np.zeros((0, 7, 7), dtype=np.float32)

        device = next(torch_model.parameters()).device
        torch_model.eval()

        # Find the last Conv2d layer
        target_layer = self._find_last_conv_layer(torch_model)
        if target_layer is None:
            raise NotImplementedError("No Conv2d layer found in model for Grad-CAM.")

        # Register hooks
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def forward_hook(module: Any, input: Any, output: torch.Tensor) -> None:
            activations.append(output.detach())

        def backward_hook(module: Any, grad_input: Any, grad_output: tuple[torch.Tensor, ...]) -> None:
            gradients.append(grad_output[0].detach())

        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        # Prepare data — take up to 10 samples
        if isinstance(X, (list, tuple)) and len(X) > 0 and isinstance(X[0], Path):
            sample_paths = list(X[:10])
            dataset = ImagePathDataset(sample_paths, transform=get_eval_transform())
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=len(sample_paths), shuffle=False, num_workers=0)
            images, _ = next(iter(loader))
        elif isinstance(X, torch.Tensor):
            images = X[:10]
        elif isinstance(X, np.ndarray):
            images = torch.from_numpy(X[:10]).float()
        else:
            raise ValueError(f"Unsupported input type for Grad-CAM: {type(X)}")

        images = images.to(device)

        # Forward pass
        logits = torch_model(images)
        pred_classes = logits.argmax(dim=1)

        # Backward pass from predicted class
        heatmap_list: list[np.ndarray] = []
        for i in range(images.shape[0]):
            torch_model.zero_grad()
            activations.clear()
            gradients.clear()

            # Single image forward
            single_img = images[i : i + 1]
            output = torch_model(single_img)
            target_class = output.argmax(dim=1).item()
            output[0, target_class].backward()

            if not activations or not gradients:
                continue

            act = activations[0]  # (1, C, H, W)
            grad = gradients[0]  # (1, C, H, W)

            # Compute alpha weights: global average pool over spatial dims
            alpha = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

            # Weighted combination of activation maps
            cam = (alpha * act).sum(dim=1, keepdim=True)  # (1, 1, H, W)
            cam = torch.relu(cam)

            # Normalize to [0, 1]
            cam = cam.squeeze().cpu().numpy()
            if cam.max() > 0:
                cam = cam / cam.max()
            heatmap_list.append(cam)

        # Remove hooks
        fwd_handle.remove()
        bwd_handle.remove()

        if not heatmap_list:
            return np.zeros((0, 7, 7), dtype=np.float32)

        return np.stack(heatmap_list, axis=0).astype(np.float32)

    def _extract_torch_model(self, model: Any) -> Any:
        if hasattr(model, "_model"):
            return model._model
        return None

    def _find_last_conv_layer(self, model: Any) -> Any:
        """Walk modules in reverse order and return the last Conv2d layer."""
        import torch.nn as nn

        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv
