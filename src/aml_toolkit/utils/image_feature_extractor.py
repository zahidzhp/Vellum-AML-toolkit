"""Shared image utilities: dataset wrapper, transforms, and feature extraction."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger("aml_toolkit")

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Standard training transform with augmentation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Standard evaluation transform (deterministic)."""
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImagePathDataset(Dataset):
    """PyTorch Dataset that loads images from file paths."""

    def __init__(
        self,
        image_paths: list[Path],
        labels: np.ndarray | None = None,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or get_eval_transform()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        tensor = self.transform(img)
        label = int(self.labels[idx]) if self.labels is not None else 0
        return tensor, label


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

# Module-level cache: backbone_name -> (model, feature_dim)
_EXTRACTOR_CACHE: dict[str, tuple[nn.Module, int]] = {}


def _load_backbone(backbone: str, device: torch.device) -> tuple[nn.Module, int]:
    """Load a pretrained backbone with the classification head removed."""
    if backbone in _EXTRACTOR_CACHE:
        model, dim = _EXTRACTOR_CACHE[backbone]
        return model.to(device), dim

    from torchvision.models import get_model, get_model_weights

    weights = get_model_weights(backbone).DEFAULT
    model = get_model(backbone, weights=weights)

    # Remove classification head — works for all ResNet variants
    if hasattr(model, "fc"):
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        # Some models (e.g., EfficientNet, DenseNet) use 'classifier'
        if isinstance(model.classifier, nn.Linear):
            feature_dim = model.classifier.in_features
            model.classifier = nn.Identity()
        elif isinstance(model.classifier, nn.Sequential):
            feature_dim = model.classifier[-1].in_features
            model.classifier[-1] = nn.Identity()
        else:
            raise ValueError(f"Cannot determine feature dim for backbone: {backbone}")
    elif hasattr(model, "head"):
        # timm-style models
        feature_dim = model.head.in_features if hasattr(model.head, "in_features") else model.num_features
        model.head = nn.Identity()
    else:
        # Fallback: run a dummy forward pass to detect output shape
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        feature_dim = out.shape[1]
        logger.warning(
            f"Could not identify classification head for {backbone}. "
            f"Inferred feature_dim={feature_dim} from forward pass."
        )

    model.eval()
    model = model.to(device)
    _EXTRACTOR_CACHE[backbone] = (model, feature_dim)
    return model, feature_dim


class ImageFeatureExtractor:
    """Extracts feature embeddings from images using a pretrained backbone.

    Supports any torchvision model with a recognizable classification head
    (ResNet, Wide ResNet, ResNeXt, EfficientNet, DenseNet, etc.).
    """

    def __init__(self, backbone: str = "resnet18", gpu_enabled: bool = True) -> None:
        self._backbone = backbone
        self._device = torch.device(
            "cuda" if gpu_enabled and torch.cuda.is_available() else "cpu"
        )
        self._model: nn.Module | None = None
        self._feature_dim: int = 0

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model, self._feature_dim = _load_backbone(self._backbone, self._device)

    @property
    def feature_dim(self) -> int:
        self._ensure_model()
        return self._feature_dim

    def extract(
        self,
        image_paths: list[Path],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract feature embeddings from a list of image paths.

        Args:
            image_paths: Paths to image files.
            batch_size: Batch size for inference.

        Returns:
            Numpy array of shape (n_images, feature_dim).
        """
        if not image_paths:
            self._ensure_model()
            return np.zeros((0, self._feature_dim), dtype=np.float32)

        self._ensure_model()
        assert self._model is not None

        dataset = ImagePathDataset(image_paths, transform=get_eval_transform())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        features_list: list[np.ndarray] = []

        self._model.eval()
        with torch.no_grad():
            for batch_imgs, _ in loader:
                batch_imgs = batch_imgs.to(self._device)
                feats = self._model(batch_imgs)
                features_list.append(feats.cpu().numpy())

        return np.concatenate(features_list, axis=0).astype(np.float32)
