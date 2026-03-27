"""ViT model adapter: transfer learning with any timm ViT variant."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.interfaces.candidate_model import CandidateModel
from aml_toolkit.utils.image_feature_extractor import (
    ImagePathDataset,
    get_eval_transform,
    get_train_transform,
)

logger = logging.getLogger("aml_toolkit")


class ViTAdapter(CandidateModel):
    """Transfer learning adapter for Vision Transformer image classification.

    Uses timm for model loading. Supports any ViT variant available in timm
    (vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224, etc.).

    Training strategy:
    1. Load pretrained ViT, set num_classes.
    2. Freeze all layers except the classification head.
    3. Train the head, then optionally fine-tune later blocks.
    """

    def __init__(self, seed: int = 42, backbone: str = "vit_small_patch16_224") -> None:
        self._seed = seed
        self._backbone = backbone
        self._model: nn.Module | None = None
        self._label_encoder = LabelEncoder()
        self._num_classes: int = 0
        self._trace: dict[str, list[float]] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._supports_gradcam = False  # ViT uses attention, not convolutions

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        config: ToolkitConfig,
    ) -> None:
        """Train ViT via transfer learning on image paths."""
        import timm

        torch.manual_seed(self._seed)
        backbone = config.candidates.vit_backbone or self._backbone

        # Encode labels
        y_train_arr = np.asarray(y_train)
        y_val_arr = np.asarray(y_val)
        self._label_encoder.fit(np.concatenate([y_train_arr, y_val_arr]))
        y_train_enc = self._label_encoder.transform(y_train_arr)
        y_val_enc = self._label_encoder.transform(y_val_arr)
        self._num_classes = len(self._label_encoder.classes_)

        # Load pretrained ViT
        model = timm.create_model(backbone, pretrained=True, num_classes=self._num_classes)

        # Freeze all except head
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "head"):
            for param in model.head.parameters():
                param.requires_grad = True
        elif hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True

        model = model.to(self._device)
        self._model = model

        # Resolve transforms — use timm's data config if available
        try:
            data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
            image_size = data_cfg.get("input_size", (3, 224, 224))[-1]
        except Exception:
            image_size = 224

        train_tfm = get_train_transform(image_size)
        eval_tfm = get_eval_transform(image_size)

        # DataLoaders
        train_paths = [Path(p) for p in X_train] if not isinstance(X_train[0], Path) else list(X_train)
        val_paths = [Path(p) for p in X_val] if not isinstance(X_val[0], Path) else list(X_val)

        batch_size = min(16, len(train_paths))  # ViT is memory-heavy
        train_loader = DataLoader(
            ImagePathDataset(train_paths, y_train_enc, train_tfm),
            batch_size=batch_size, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            ImagePathDataset(val_paths, y_val_enc, eval_tfm),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
        )
        criterion = nn.CrossEntropyLoss()

        n_epochs = config.runtime_decision.min_warmup_epochs_neural * 2
        self._trace = {"train_loss": [], "val_loss": [], "val_macro_f1": []}

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self._device), labels.to(self._device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            self._trace["train_loss"].append(avg_train_loss)

            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self._device), labels.to(self._device)
                    logits = model(imgs)
                    val_loss += criterion(logits, labels).item()
                    all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / max(len(val_loader), 1)
            val_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
            self._trace["val_loss"].append(avg_val_loss)
            self._trace["val_macro_f1"].append(val_f1)

            logger.info(
                f"ViT [{backbone}] epoch {epoch+1}/{n_epochs}: "
                f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_f1={val_f1:.4f}"
            )

    def predict(self, X: Any) -> np.ndarray:
        preds_enc = self._predict_encoded(X)
        return self._label_encoder.inverse_transform(preds_enc)

    def predict_proba(self, X: Any) -> np.ndarray | None:
        assert self._model is not None
        self._model.eval()
        paths = [Path(p) for p in X] if not isinstance(X[0], Path) else list(X)
        dataset = ImagePathDataset(paths, transform=get_eval_transform())
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(self._device)
                logits = self._model(imgs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    def evaluate(self, X: Any, y: Any, metrics: list[str]) -> dict[str, float]:
        pred = self.predict(X)
        results: dict[str, float] = {}
        for m in metrics:
            if m == "accuracy":
                results[m] = float(accuracy_score(y, pred))
            elif m == "macro_f1":
                results[m] = float(f1_score(y, pred, average="macro", zero_division=0))
        return results

    def get_training_trace(self) -> dict[str, list[float]]:
        return self._trace

    def get_model_family(self) -> str:
        return "vit"

    def is_probabilistic(self) -> bool:
        return True

    def serialize(self, path: Any) -> None:
        assert self._model is not None
        torch.save(self._model.state_dict(), Path(path))

    def _predict_encoded(self, X: Any) -> np.ndarray:
        assert self._model is not None
        self._model.eval()
        paths = [Path(p) for p in X] if not isinstance(X[0], Path) else list(X)
        dataset = ImagePathDataset(paths, transform=get_eval_transform())
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        all_preds: list[np.ndarray] = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(self._device)
                preds = self._model(imgs).argmax(dim=1).cpu().numpy()
                all_preds.append(preds)

        return np.concatenate(all_preds, axis=0)
