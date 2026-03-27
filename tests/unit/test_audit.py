"""Tests for split auditing, leakage checks, and augmentation guard."""

import numpy as np
import pandas as pd
import pytest

from aml_toolkit.artifacts import SplitAuditReport, DatasetManifest
from aml_toolkit.audit.augmentation_guard import AugmentationGuard
from aml_toolkit.audit.leakage_checks import (
    check_class_absence,
    check_duplicate_overlap,
    check_grouped_leakage,
    check_temporal_leakage,
)
from aml_toolkit.audit.split_auditor import run_split_audit
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import ModalityType, SplitStrategy, TaskType
from aml_toolkit.core.exceptions import SplitIntegrityError
from aml_toolkit.intake.split_builder import SplitResult


# --- Helpers ---


def _make_split(train: list[int], val: list[int], test: list[int]) -> SplitResult:
    return SplitResult(
        train_indices=np.array(train),
        val_indices=np.array(val),
        test_indices=np.array(test),
        strategy=SplitStrategy.STRATIFIED,
    )


def _make_manifest(**kwargs) -> DatasetManifest:
    defaults = dict(
        dataset_id="test",
        modality=ModalityType.TABULAR,
        task_type=TaskType.BINARY,
        split_strategy=SplitStrategy.STRATIFIED,
        class_labels=["0", "1"],
    )
    defaults.update(kwargs)
    return DatasetManifest(**defaults)


# --- Duplicate Overlap Tests ---


class TestDuplicateOverlap:
    def test_no_duplicates(self):
        df = pd.DataFrame({
            "f1": [1, 2, 3, 4, 5, 6],
            "f2": [10, 20, 30, 40, 50, 60],
            "label": [0, 1, 0, 1, 0, 1],
        })
        split = _make_split([0, 1, 2], [3], [4, 5])
        result = check_duplicate_overlap(
            {"df": df}, split, feature_columns=["f1", "f2"]
        )
        assert result["train_test_overlap"] == 0
        assert result["train_val_overlap"] == 0

    def test_duplicates_across_train_test(self):
        # Rows 0 and 4 have identical features
        df = pd.DataFrame({
            "f1": [1, 2, 3, 5, 1],
            "f2": [10, 20, 30, 50, 10],
            "label": [0, 1, 0, 1, 0],
        })
        split = _make_split([0, 1, 2], [3], [4])
        result = check_duplicate_overlap(
            {"df": df}, split, feature_columns=["f1", "f2"]
        )
        assert result["train_test_overlap"] == 1

    def test_duplicates_in_embeddings(self):
        emb = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [1.0, 2.0],  # duplicate of row 0
        ])
        split = _make_split([0, 1], [2], [3])
        result = check_duplicate_overlap({"embeddings": emb}, split)
        assert result["train_test_overlap"] == 1


# --- Grouped Leakage Tests ---


class TestGroupedLeakage:
    def test_no_leakage(self):
        groups = np.array(["A", "A", "B", "B", "C", "C"])
        split = _make_split([0, 1], [2, 3], [4, 5])
        result = check_grouped_leakage(groups, split)
        assert result["total_leaked_groups"] == 0

    def test_group_leaks_across_train_test(self):
        # Group "A" appears in both train and test
        groups = np.array(["A", "A", "B", "B", "A", "C"])
        split = _make_split([0, 1], [2, 3], [4, 5])
        result = check_grouped_leakage(groups, split)
        assert result["total_leaked_groups"] > 0
        assert "A" in result["leaked_groups"]

    def test_group_leaks_across_all_splits(self):
        groups = np.array(["X", "X", "X", "Y", "Y", "Y"])
        split = _make_split([0], [1], [2])
        result = check_grouped_leakage(groups, split)
        assert "X" in result["leaked_groups"]


# --- Temporal Leakage Tests ---


class TestTemporalLeakage:
    def test_no_temporal_leakage(self):
        # Train: [1,2,3], Val: [4,5], Test: [6,7]
        timestamps = np.array([1, 2, 3, 4, 5, 6, 7])
        split = _make_split([0, 1, 2], [3, 4], [5, 6])
        result = check_temporal_leakage(timestamps, split)
        assert result["train_val_leakage"] is False
        assert result["val_test_leakage"] is False

    def test_train_val_temporal_leakage(self):
        # Train has timestamp 5, but val has timestamp 2
        timestamps = np.array([1, 5, 2, 3, 4])
        split = _make_split([0, 1], [2], [3, 4])
        result = check_temporal_leakage(timestamps, split)
        assert result["train_val_leakage"] is True

    def test_val_test_temporal_leakage(self):
        # Val has timestamp 6, test has timestamp 3
        timestamps = np.array([1, 2, 6, 3, 4])
        split = _make_split([0, 1], [2], [3, 4])
        result = check_temporal_leakage(timestamps, split)
        assert result["val_test_leakage"] is True


# --- Class Absence Tests ---


class TestClassAbsence:
    def test_no_absence(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        split = _make_split([0, 1], [2, 3], [4, 5])
        result = check_class_absence(labels, split, ["0", "1"])
        assert result["absent_in_train"] == []
        assert result["absent_in_val"] == []
        assert result["absent_in_test"] == []

    def test_class_absent_in_val(self):
        # Val split only has class 0
        labels = np.array([0, 1, 0, 0, 0, 1])
        split = _make_split([0, 1], [2, 3], [4, 5])
        result = check_class_absence(labels, split, ["0", "1"])
        assert "1" in result["absent_in_val"]

    def test_class_absent_in_test(self):
        labels = np.array([0, 1, 0, 1, 0, 0])
        split = _make_split([0, 1], [2, 3], [4, 5])
        result = check_class_absence(labels, split, ["0", "1"])
        assert "1" in result["absent_in_test"]

    def test_class_absent_in_train_is_blocking(self):
        labels = np.array([0, 0, 0, 1, 0, 1])
        split = _make_split([0, 1, 2], [3], [4, 5])
        result = check_class_absence(labels, split, ["0", "1"])
        assert "1" in result["absent_in_train"]


# --- Augmentation Guard Tests ---


class TestAugmentationGuard:
    def test_blocks_before_split_finalization(self):
        guard = AugmentationGuard()
        with pytest.raises(SplitIntegrityError, match="before split finalization"):
            guard.check_augmentation_allowed()

    def test_blocks_before_audit_passed(self):
        guard = AugmentationGuard()
        guard.mark_split_finalized()
        with pytest.raises(SplitIntegrityError, match="before split audit passed"):
            guard.check_augmentation_allowed()

    def test_allows_after_audit_passed(self):
        guard = AugmentationGuard()
        report = SplitAuditReport(passed=True)
        guard.mark_audit_passed(report)
        guard.check_augmentation_allowed()  # should not raise
        assert guard.is_augmentation_safe is True

    def test_blocks_if_audit_has_blocking_issues(self):
        guard = AugmentationGuard()
        report = SplitAuditReport(
            passed=False,
            blocking_issues=["duplicate leakage detected"],
        )
        guard.mark_audit_passed(report)
        assert guard.is_augmentation_safe is False
        with pytest.raises(SplitIntegrityError):
            guard.check_augmentation_allowed()

    def test_is_augmentation_safe_property(self):
        guard = AugmentationGuard()
        assert guard.is_augmentation_safe is False
        guard.mark_split_finalized()
        assert guard.is_augmentation_safe is False
        guard.mark_audit_passed(SplitAuditReport(passed=True))
        assert guard.is_augmentation_safe is True


# --- Split Auditor Integration Tests ---


class TestSplitAuditor:
    def test_clean_audit(self):
        df = pd.DataFrame({
            "f1": [1, 2, 3, 4, 5, 6],
            "f2": [10, 20, 30, 40, 50, 60],
            "label": [0, 1, 0, 1, 0, 1],
        })
        split = _make_split([0, 1, 2], [3], [4, 5])
        manifest = _make_manifest(
            feature_columns=["f1", "f2"],
            target_column="label",
        )
        config = ToolkitConfig()
        report = run_split_audit(
            data={"df": df},
            manifest=manifest,
            split=split,
            config=config,
            labels=df["label"].values,
        )
        assert report.passed is True
        assert len(report.blocking_issues) == 0

    def test_audit_detects_duplicate_leakage(self):
        df = pd.DataFrame({
            "f1": [1, 2, 3, 1],
            "f2": [10, 20, 30, 10],
            "label": [0, 1, 0, 0],
        })
        split = _make_split([0, 1], [2], [3])
        manifest = _make_manifest(
            feature_columns=["f1", "f2"],
            target_column="label",
        )
        report = run_split_audit(
            data={"df": df},
            manifest=manifest,
            split=split,
            config=ToolkitConfig(),
            labels=df["label"].values,
        )
        assert report.passed is False
        assert "duplicate_overlap" in report.leakage_flags

    def test_audit_detects_grouped_leakage(self):
        df = pd.DataFrame({
            "f1": range(6),
            "label": [0, 1, 0, 1, 0, 1],
        })
        groups = np.array(["A", "A", "B", "B", "A", "C"])
        split = _make_split([0, 1], [2, 3], [4, 5])
        manifest = _make_manifest(feature_columns=["f1"], target_column="label")
        report = run_split_audit(
            data={"df": df},
            manifest=manifest,
            split=split,
            config=ToolkitConfig(),
            labels=df["label"].values,
            groups=groups,
        )
        assert report.passed is False
        assert "entity_leakage" in report.leakage_flags

    def test_audit_detects_temporal_leakage(self):
        df = pd.DataFrame({
            "f1": range(5),
            "label": [0, 1, 0, 1, 0],
        })
        timestamps = np.array([1, 5, 2, 3, 4])
        split = _make_split([0, 1], [2], [3, 4])
        manifest = _make_manifest(feature_columns=["f1"], target_column="label")
        report = run_split_audit(
            data={"df": df},
            manifest=manifest,
            split=split,
            config=ToolkitConfig(),
            labels=df["label"].values,
            timestamps=timestamps,
        )
        assert report.passed is False
        assert "temporal_leakage" in report.leakage_flags

    def test_audit_detects_class_absence_in_val(self):
        df = pd.DataFrame({
            "f1": range(6),
            "label": [0, 1, 0, 0, 0, 1],
        })
        split = _make_split([0, 1], [2, 3], [4, 5])
        manifest = _make_manifest(
            feature_columns=["f1"],
            target_column="label",
        )
        report = run_split_audit(
            data={"df": df},
            manifest=manifest,
            split=split,
            config=ToolkitConfig(),
            labels=df["label"].values,
        )
        # Class absent in val is a warning, not blocking
        assert len(report.class_absence_flags) > 0
        assert any("val" in f for f in report.class_absence_flags)

    def test_audit_report_serializes(self, tmp_path):
        report = SplitAuditReport(
            passed=False,
            leakage_flags=["duplicate_overlap"],
            blocking_issues=["test issue"],
        )
        from aml_toolkit.utils.serialization import save_artifact_json, load_artifact_json

        path = tmp_path / "audit.json"
        save_artifact_json(report, path)
        loaded = load_artifact_json(SplitAuditReport, path)
        assert loaded.passed is False
        assert loaded.leakage_flags == ["duplicate_overlap"]

    def test_class_absence_in_train_is_blocking(self):
        df = pd.DataFrame({
            "f1": range(6),
            "label": [0, 0, 0, 1, 0, 1],
        })
        # Train has only class 0
        split = _make_split([0, 1, 2], [3], [4, 5])
        manifest = _make_manifest(
            feature_columns=["f1"],
            target_column="label",
        )
        report = run_split_audit(
            data={"df": df},
            manifest=manifest,
            split=split,
            config=ToolkitConfig(),
            labels=df["label"].values,
        )
        assert report.passed is False
        assert any("train" in issue for issue in report.blocking_issues)
