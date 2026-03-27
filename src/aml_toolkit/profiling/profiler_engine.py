"""Profiler engine: aggregates all profiling modules into a single DataProfile."""

import logging
from typing import Any

import numpy as np
import pandas as pd

from aml_toolkit.artifacts import DataProfile, DatasetManifest
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import ModalityType, RiskFlag
from aml_toolkit.intake.split_builder import SplitResult
from aml_toolkit.profiling.class_distribution import profile_class_distribution
from aml_toolkit.profiling.drift_ood import profile_drift
from aml_toolkit.profiling.duplicates import profile_duplicates
from aml_toolkit.profiling.label_conflicts import detect_label_conflicts
from aml_toolkit.profiling.missingness import profile_missingness
from aml_toolkit.profiling.outliers import profile_outliers

logger = logging.getLogger("aml_toolkit")


def run_profiling(
    data: Any,
    manifest: DatasetManifest,
    split: SplitResult,
    config: ToolkitConfig,
) -> DataProfile:
    """Run all profiling modules and aggregate into a DataProfile.

    Args:
        data: Raw data dict from intake.
        manifest: The DatasetManifest from intake.
        split: The SplitResult with train/val/test indices.
        config: Toolkit configuration.

    Returns:
        DataProfile with statistics and risk flags.
    """
    risk_flags: list[RiskFlag] = []

    # Extract labels
    labels: np.ndarray | None = None
    if "df" in data and manifest.target_column:
        labels = data["df"][manifest.target_column].values
    elif "labels" in data:
        labels = data["labels"]

    # 1. Class distribution
    class_result: dict = {}
    if labels is not None:
        class_result = profile_class_distribution(labels, config)
        risk_flags.extend(class_result.get("risk_flags", []))

    # 2. Missingness (tabular only)
    missingness_summary: dict[str, float] = {}
    if "df" in data and manifest.feature_columns:
        missingness_summary = profile_missingness(data["df"], manifest.feature_columns)

    # 3. Duplicates (tabular only)
    duplicate_summary: dict[str, Any] = {}
    if "df" in data and manifest.feature_columns and config.profiling.duplicate_check_enabled:
        duplicate_summary = profile_duplicates(data["df"], manifest.feature_columns)

    # 4. Label conflicts (tabular only)
    label_conflict_summary: dict[str, Any] = {}
    if "df" in data and manifest.feature_columns and manifest.target_column:
        label_conflict_summary = detect_label_conflicts(
            data["df"], manifest.feature_columns, manifest.target_column
        )
        if label_conflict_summary.get("conflict_groups", 0) > 0:
            risk_flags.append(RiskFlag.LABEL_CONFLICT)

    # 5. Outliers (tabular only)
    outlier_summary: dict[str, Any] = {}
    if "df" in data and manifest.feature_columns:
        outlier_summary = profile_outliers(data["df"], manifest.feature_columns)

    # 6. Drift / OOD (tabular only)
    ood_shift_summary: dict[str, Any] = {}
    if (
        "df" in data
        and manifest.feature_columns
        and config.profiling.ood_shift_enabled
    ):
        ood_shift_summary = profile_drift(
            data["df"],
            manifest.feature_columns,
            split.train_indices,
            split.test_indices,
        )
        if ood_shift_summary.get("total_shifted", 0) > 0:
            risk_flags.append(RiskFlag.OOD_SHIFT)

    # Deduplicate risk flags
    risk_flags = list(dict.fromkeys(risk_flags))

    total_samples = manifest.train_size + manifest.val_size + manifest.test_size

    profile = DataProfile(
        total_samples=total_samples,
        class_counts=class_result.get("class_counts", {}),
        class_ratios=class_result.get("class_ratios", {}),
        imbalance_severity=class_result.get("imbalance_severity", "unknown"),
        missingness_summary=missingness_summary,
        duplicate_summary=duplicate_summary,
        label_conflict_summary=label_conflict_summary,
        outlier_summary=outlier_summary,
        ood_shift_summary=ood_shift_summary,
        risk_flags=risk_flags,
    )

    return profile
