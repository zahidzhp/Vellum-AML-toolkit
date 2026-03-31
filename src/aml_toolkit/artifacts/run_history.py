"""Run history artifacts — DatasetSignature and RunHistoryRecord."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class DatasetSignature(BaseModel):
    """Numerical + categorical fingerprint of a dataset for similarity search.

    The `to_vector()` method returns a normalized float32 array suitable for
    cosine similarity. All values are in [0, 1].
    """

    # Categorical metadata (for display and pre-filtering)
    modality: str  # TABULAR | IMAGE | EMBEDDING
    task_type: str  # BINARY | MULTICLASS | MULTILABEL
    n_classes: int

    # Numerical feature vector components
    log_n_samples: float  # log10(n_samples); use 0 if unknown
    log_n_features: float  # log10(n_features); 0 for image modality
    imbalance_ratio: float  # max_class / min_class, capped at 100
    missingness_pct: float  # mean % missing across features, in [0, 1]
    duplicate_pct: float  # % near-duplicate rows, in [0, 1]
    ood_shift_score: float  # 0-1 from drift detector; 0 if not run
    label_noise_score: float  # 0-1 from conflict detector; 0 if not run

    # Derived flags (for fast pre-filtering)
    has_label_noise: bool = False
    has_ood_shift: bool = False
    has_severe_imbalance: bool = False  # imbalance_ratio > 20

    # Image-specific fields (optional)
    image_size_bucket: str = ""  # "tiny"(<64), "small"(64-224), "large"(>224)
    n_channels: int = 0
    backbone: str = ""  # e.g. "resnet18", "vit_small_patch16_224"
    augmentation_used: bool = False

    def to_vector(self) -> np.ndarray:
        """Return normalized feature vector for cosine similarity.

        All components are in [0, 1].
        """
        return np.array(
            [
                min(self.log_n_samples, 6.0) / 6.0,  # normalize by log10(1M)
                min(self.log_n_features, 4.0) / 4.0,  # normalize by log10(10k)
                min(self.imbalance_ratio, 100.0) / 100.0,
                float(np.clip(self.missingness_pct, 0.0, 1.0)),
                float(np.clip(self.duplicate_pct, 0.0, 1.0)),
                float(np.clip(self.ood_shift_score, 0.0, 1.0)),
                float(np.clip(self.label_noise_score, 0.0, 1.0)),
                float(self.has_label_noise),
                float(self.has_ood_shift),
                float(self.has_severe_imbalance),
            ],
            dtype=np.float32,
        )


class RunHistoryRecord(BaseModel):
    """A single completed pipeline run stored in the history."""

    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    dataset_signature: DatasetSignature
    dataset_signature_vector: list[float] = Field(default_factory=list)
    candidate_families_tried: list[str] = Field(default_factory=list)
    best_candidate_id: str = ""
    best_candidate_family: str = ""
    best_macro_f1: float = 0.0
    config_mode: str = ""
    stages_completed: list[str] = Field(default_factory=list)
    abstained: bool = False
    notes: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Auto-populate the vector if not explicitly set."""
        if not self.dataset_signature_vector:
            self.dataset_signature_vector = self.dataset_signature.to_vector().tolist()
