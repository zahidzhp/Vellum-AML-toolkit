"""Build a DatasetSignature from pipeline artifacts."""

from __future__ import annotations

import math
import logging
from typing import Any

from aml_toolkit.artifacts.run_history import DatasetSignature

logger = logging.getLogger("aml_toolkit")


def build_dataset_signature(
    dataset_manifest: Any,
    data_profile: Any,
) -> DatasetSignature:
    """Build a DatasetSignature from DatasetManifest and DataProfile artifacts.

    Falls back gracefully if fields are missing — never raises.

    Args:
        dataset_manifest: DatasetManifest artifact (or dict).
        data_profile: DataProfile artifact (or dict).

    Returns:
        DatasetSignature with numerical vector populated.
    """
    try:
        return _build(dataset_manifest, data_profile)
    except Exception as e:
        logger.warning(f"DatasetSignature build failed, using minimal defaults: {e}")
        return DatasetSignature(
            modality="TABULAR",
            task_type="BINARY",
            n_classes=2,
            log_n_samples=0.0,
            log_n_features=0.0,
            imbalance_ratio=1.0,
            missingness_pct=0.0,
            duplicate_pct=0.0,
            ood_shift_score=0.0,
            label_noise_score=0.0,
        )


def _build(manifest: Any, profile: Any) -> DatasetSignature:
    """Internal builder — may raise."""
    manifest_d = _to_dict(manifest)
    profile_d = _to_dict(profile)

    modality = str(manifest_d.get("modality", "TABULAR")).upper()
    task_type = str(manifest_d.get("task_type", "BINARY")).upper()
    n_classes = int(manifest_d.get("n_classes", 2))
    n_samples = int(manifest_d.get("n_samples", 1))
    n_features = int(manifest_d.get("n_features", 1))

    log_n_samples = math.log10(max(n_samples, 1))
    log_n_features = 0.0 if modality == "IMAGE" else math.log10(max(n_features, 1))

    # Imbalance
    class_counts = manifest_d.get("class_counts", {})
    if class_counts and len(class_counts) >= 2:
        counts = [v for v in class_counts.values() if isinstance(v, (int, float)) and v > 0]
        imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else 1.0
    else:
        imbalance_ratio = float(profile_d.get("imbalance_ratio", 1.0))

    missingness_pct = float(profile_d.get("mean_missingness", 0.0))
    duplicate_pct = float(profile_d.get("duplicate_pct", 0.0))
    ood_shift_score = float(profile_d.get("ood_shift_score", 0.0))
    label_noise_score = float(profile_d.get("label_noise_score", 0.0))

    has_label_noise = bool(profile_d.get("has_label_noise", label_noise_score > 0.1))
    has_ood_shift = bool(profile_d.get("has_ood_shift", ood_shift_score > 0.2))
    has_severe_imbalance = imbalance_ratio > 20.0

    # Image-specific
    image_size_bucket = str(manifest_d.get("image_size_bucket", ""))
    n_channels = int(manifest_d.get("n_channels", 0))
    backbone = str(manifest_d.get("backbone", ""))
    augmentation_used = bool(manifest_d.get("augmentation_used", False))

    return DatasetSignature(
        modality=modality,
        task_type=task_type,
        n_classes=n_classes,
        log_n_samples=log_n_samples,
        log_n_features=log_n_features,
        imbalance_ratio=float(imbalance_ratio),
        missingness_pct=missingness_pct,
        duplicate_pct=duplicate_pct,
        ood_shift_score=ood_shift_score,
        label_noise_score=label_noise_score,
        has_label_noise=has_label_noise,
        has_ood_shift=has_ood_shift,
        has_severe_imbalance=has_severe_imbalance,
        image_size_bucket=image_size_bucket,
        n_channels=n_channels,
        backbone=backbone,
        augmentation_used=augmentation_used,
    )


def _to_dict(obj: Any) -> dict:
    """Convert a Pydantic model or plain dict to a dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return {}
