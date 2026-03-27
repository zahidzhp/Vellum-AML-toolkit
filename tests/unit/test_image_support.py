"""Tests for image support: feature extraction, CNN/ViT adapters, GradCAM, and image pipeline."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import ModalityType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_image_folder(tmp_path: Path, n_per_class: int = 5) -> Path:
    """Create a minimal image classification folder structure."""
    img_dir = tmp_path / "images"
    for class_name in ["cat", "dog", "bird"]:
        class_dir = img_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            )
            img.save(class_dir / f"{class_name}_{i}.jpg")
    return img_dir


# ---------------------------------------------------------------------------
# ImageFeatureExtractor tests
# ---------------------------------------------------------------------------

class TestImageFeatureExtractor:

    def test_extract_returns_correct_shape(self, tmp_path):
        from aml_toolkit.utils.image_feature_extractor import ImageFeatureExtractor

        img_dir = _create_image_folder(tmp_path, n_per_class=3)
        paths = sorted(img_dir.rglob("*.jpg"))
        assert len(paths) == 9

        extractor = ImageFeatureExtractor(backbone="resnet18", gpu_enabled=False)
        features = extractor.extract(paths, batch_size=4)

        assert features.shape[0] == 9
        assert features.shape[1] == 512  # resnet18 feature dim
        assert features.dtype == np.float32

    def test_extract_empty_list(self):
        from aml_toolkit.utils.image_feature_extractor import ImageFeatureExtractor

        extractor = ImageFeatureExtractor(backbone="resnet18", gpu_enabled=False)
        features = extractor.extract([])
        assert features.shape == (0, 512)

    def test_feature_dim_property(self):
        from aml_toolkit.utils.image_feature_extractor import ImageFeatureExtractor

        extractor = ImageFeatureExtractor(backbone="resnet18", gpu_enabled=False)
        assert extractor.feature_dim == 512


# ---------------------------------------------------------------------------
# ImagePathDataset tests
# ---------------------------------------------------------------------------

class TestImagePathDataset:

    def test_dataset_returns_correct_types(self, tmp_path):
        from aml_toolkit.utils.image_feature_extractor import ImagePathDataset

        img_dir = _create_image_folder(tmp_path, n_per_class=2)
        paths = sorted(img_dir.rglob("*.jpg"))
        labels = np.array([0] * 2 + [1] * 2 + [2] * 2)

        dataset = ImagePathDataset(paths, labels)
        tensor, label = dataset[0]

        assert tensor.shape == (3, 224, 224)  # default eval transform
        assert isinstance(label, int)
        assert len(dataset) == 6


# ---------------------------------------------------------------------------
# CNN adapter tests
# ---------------------------------------------------------------------------

class TestCNNAdapter:

    def test_cnn_fit_predict(self, tmp_path):
        from aml_toolkit.models.image.cnn_adapter import CNNAdapter

        img_dir = _create_image_folder(tmp_path, n_per_class=6)
        all_paths = sorted(img_dir.rglob("*.jpg"))
        all_labels = np.array(
            ["cat"] * 6 + ["dog"] * 6 + ["bird"] * 6
        )

        # Split: 12 train, 6 val
        train_paths = all_paths[:12]
        train_labels = all_labels[:12]
        val_paths = all_paths[12:]
        val_labels = all_labels[12:]

        config = ToolkitConfig(
            candidates={"cnn_backbone": "resnet18"},
            runtime_decision={"min_warmup_epochs_neural": 1},
            compute={"gpu_enabled": False},
        )

        adapter = CNNAdapter(seed=42, backbone="resnet18")
        adapter.fit(train_paths, train_labels, val_paths, val_labels, config)

        # Predict
        preds = adapter.predict(val_paths)
        assert len(preds) == 6
        assert all(p in ["cat", "dog", "bird"] for p in preds)

        # Predict proba
        proba = adapter.predict_proba(val_paths)
        assert proba.shape == (6, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

        # Evaluate
        metrics = adapter.evaluate(val_paths, val_labels, ["accuracy", "macro_f1"])
        assert "accuracy" in metrics
        assert "macro_f1" in metrics

        # Training trace
        trace = adapter.get_training_trace()
        assert "val_macro_f1" in trace
        assert len(trace["val_macro_f1"]) > 0

    def test_cnn_serialize(self, tmp_path):
        from aml_toolkit.models.image.cnn_adapter import CNNAdapter

        img_dir = _create_image_folder(tmp_path, n_per_class=4)
        paths = sorted(img_dir.rglob("*.jpg"))
        labels = np.array(["cat"] * 4 + ["dog"] * 4 + ["bird"] * 4)

        config = ToolkitConfig(
            runtime_decision={"min_warmup_epochs_neural": 1},
            compute={"gpu_enabled": False},
        )

        adapter = CNNAdapter(seed=42)
        adapter.fit(paths[:8], labels[:8], paths[8:], labels[8:], config)

        model_path = tmp_path / "cnn_model.pt"
        adapter.serialize(model_path)
        assert model_path.exists()

    def test_cnn_properties(self):
        from aml_toolkit.models.image.cnn_adapter import CNNAdapter

        adapter = CNNAdapter()
        assert adapter.get_model_family() == "cnn"
        assert adapter.is_probabilistic() is True
        assert adapter._supports_gradcam is True


# ---------------------------------------------------------------------------
# ViT adapter tests
# ---------------------------------------------------------------------------

class TestViTAdapter:

    def test_vit_fit_predict(self, tmp_path):
        from aml_toolkit.models.image.vit_adapter import ViTAdapter

        img_dir = _create_image_folder(tmp_path, n_per_class=6)
        all_paths = sorted(img_dir.rglob("*.jpg"))
        all_labels = np.array(
            ["cat"] * 6 + ["dog"] * 6 + ["bird"] * 6
        )

        config = ToolkitConfig(
            candidates={"vit_backbone": "vit_small_patch16_224"},
            runtime_decision={"min_warmup_epochs_neural": 1},
            compute={"gpu_enabled": False},
        )

        adapter = ViTAdapter(seed=42)
        adapter.fit(all_paths[:12], all_labels[:12], all_paths[12:], all_labels[12:], config)

        preds = adapter.predict(all_paths[12:])
        assert len(preds) == 6

        proba = adapter.predict_proba(all_paths[12:])
        assert proba.shape == (6, 3)

    def test_vit_properties(self):
        from aml_toolkit.models.image.vit_adapter import ViTAdapter

        adapter = ViTAdapter()
        assert adapter.get_model_family() == "vit"
        assert adapter.is_probabilistic() is True
        assert adapter._supports_gradcam is False


# ---------------------------------------------------------------------------
# Grad-CAM tests
# ---------------------------------------------------------------------------

class TestGradCAMReal:

    def test_gradcam_produces_real_heatmaps(self, tmp_path):
        from aml_toolkit.explainability.gradcam import GradCAMStrategy
        from aml_toolkit.models.image.cnn_adapter import CNNAdapter

        img_dir = _create_image_folder(tmp_path, n_per_class=4)
        paths = sorted(img_dir.rglob("*.jpg"))
        labels = np.array(["cat"] * 4 + ["dog"] * 4 + ["bird"] * 4)

        config = ToolkitConfig(
            runtime_decision={"min_warmup_epochs_neural": 1},
            compute={"gpu_enabled": False},
        )

        adapter = CNNAdapter(seed=42)
        adapter.fit(paths[:8], labels[:8], paths[8:], labels[8:], config)

        strategy = GradCAMStrategy()
        assert strategy.supports_model(adapter) is True

        output = strategy.explain(
            adapter, paths[:3], labels[:3], tmp_path / "heatmaps", config,
        )

        assert output.supported is not False
        assert len(output.artifact_paths) > 0

        heatmaps = np.load(output.artifact_paths[0])
        assert heatmaps.shape[0] == 3
        assert heatmaps.ndim == 3  # (n, H, W)
        assert heatmaps.max() <= 1.0
        assert heatmaps.min() >= 0.0

    def test_gradcam_unsupported_model(self, tmp_path):
        from unittest.mock import MagicMock
        from aml_toolkit.explainability.gradcam import GradCAMStrategy

        mock_model = MagicMock()
        mock_model.get_model_family.return_value = "rf"
        mock_model._supports_gradcam = False

        strategy = GradCAMStrategy()
        assert strategy.supports_model(mock_model) is False

        config = ToolkitConfig()
        output = strategy.explain(mock_model, [], [], tmp_path, config)
        assert output.supported is False


# ---------------------------------------------------------------------------
# Image intake integration
# ---------------------------------------------------------------------------

class TestImageIntake:

    def test_image_intake_returns_image_paths(self, tmp_path):
        from aml_toolkit.intake.dataset_intake_manager import run_intake

        img_dir = _create_image_folder(tmp_path, n_per_class=15)
        config = ToolkitConfig(dataset={"path": str(img_dir)})
        result = run_intake(config)

        assert result.manifest.modality == ModalityType.IMAGE
        assert "image_paths" in result.data
        assert len(result.data["image_paths"]) == 45
        assert result.split_result is not None
