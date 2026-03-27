"""Split auditor: orchestrates all leakage and integrity checks, emits SplitAuditReport."""

import logging
from typing import Any

import numpy as np

from aml_toolkit.artifacts import DatasetManifest, SplitAuditReport
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import SplitStrategy
from aml_toolkit.intake.split_builder import SplitResult

from aml_toolkit.audit.leakage_checks import (
    check_class_absence,
    check_duplicate_overlap,
    check_grouped_leakage,
    check_temporal_leakage,
)

logger = logging.getLogger("aml_toolkit")


def run_split_audit(
    data: Any,
    manifest: DatasetManifest,
    split: SplitResult,
    config: ToolkitConfig,
    labels: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> SplitAuditReport:
    """Run all applicable split audit checks and produce a SplitAuditReport.

    Args:
        data: Raw data dict from intake (contains 'df', 'embeddings', etc.).
        manifest: The DatasetManifest from intake.
        split: The SplitResult with train/val/test indices.
        config: Toolkit configuration.
        labels: Full label array (extracted from data if not provided).
        groups: Group array (optional, for grouped leakage check).
        timestamps: Timestamp array (optional, for temporal leakage check).

    Returns:
        SplitAuditReport with all findings.
    """
    blocking_issues: list[str] = []
    warnings: list[str] = []
    leakage_flags: list[str] = []

    # Extract labels if not provided
    if labels is None:
        if "df" in data and manifest.target_column:
            labels = data["df"][manifest.target_column].values
        elif "labels" in data:
            labels = data["labels"]

    # 1. Duplicate overlap check
    feature_columns = manifest.feature_columns if manifest.feature_columns else None
    dup_result = check_duplicate_overlap(data, split, feature_columns)

    dup_summary = {
        "train_test_overlap": dup_result["train_test_overlap"],
        "train_val_overlap": dup_result["train_val_overlap"],
    }

    if dup_result["train_test_overlap"] > 0:
        leakage_flags.append("duplicate_overlap")
        msg = (
            f"Duplicate overlap: {dup_result['train_test_overlap']} identical "
            f"feature rows found in both train and test splits."
        )
        blocking_issues.append(msg)
        logger.warning(msg)

    if dup_result["train_val_overlap"] > 0:
        msg = (
            f"Duplicate overlap: {dup_result['train_val_overlap']} identical "
            f"feature rows found in both train and val splits."
        )
        warnings.append(msg)

    # 2. Grouped/entity leakage check
    entity_summary: dict[str, str] = {}
    if groups is not None:
        group_result = check_grouped_leakage(groups, split)
        if group_result["total_leaked_groups"] > 0:
            leakage_flags.append("entity_leakage")
            msg = (
                f"Entity leakage: {group_result['total_leaked_groups']} group(s) "
                f"appear in multiple splits."
            )
            blocking_issues.append(msg)
            entity_summary["leaked_count"] = str(group_result["total_leaked_groups"])
            entity_summary["leaked_groups_sample"] = str(
                group_result["leaked_groups"][:5]
            )
            logger.warning(msg)
        else:
            entity_summary["status"] = "clean"

    # 3. Temporal leakage check
    temporal_summary: dict[str, str] = {}
    if timestamps is not None:
        temporal_result = check_temporal_leakage(timestamps, split)
        if temporal_result["train_val_leakage"] or temporal_result["val_test_leakage"]:
            leakage_flags.append("temporal_leakage")
            for detail in temporal_result["details"]:
                blocking_issues.append(detail)
                logger.warning(detail)
            temporal_summary["train_val_leakage"] = str(
                temporal_result["train_val_leakage"]
            )
            temporal_summary["val_test_leakage"] = str(
                temporal_result["val_test_leakage"]
            )
        else:
            temporal_summary["status"] = "clean"

    # 4. Class absence check
    class_absence_flags: list[str] = []
    if labels is not None and manifest.class_labels:
        absence_result = check_class_absence(labels, split, manifest.class_labels)
        for split_name, absent_classes in [
            ("train", absence_result["absent_in_train"]),
            ("val", absence_result["absent_in_val"]),
            ("test", absence_result["absent_in_test"]),
        ]:
            if absent_classes:
                flag = f"Class(es) {absent_classes} absent from {split_name} split."
                class_absence_flags.append(flag)
                if split_name == "train":
                    blocking_issues.append(flag)
                else:
                    warnings.append(flag)

    # Determine if audit passed
    passed = len(blocking_issues) == 0

    report = SplitAuditReport(
        passed=passed,
        leakage_flags=leakage_flags,
        duplicate_overlap_summary=dup_summary,
        entity_leakage_summary=entity_summary,
        temporal_leakage_summary=temporal_summary,
        class_absence_flags=class_absence_flags,
        augmentation_leakage_safe=passed,
        blocking_issues=blocking_issues,
        warnings=warnings,
    )

    return report
