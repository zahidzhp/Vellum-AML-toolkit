"""Tests for profiling engine and individual profiling modules."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from aml_toolkit.artifacts import DataProfile, DatasetManifest
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import ModalityType, RiskFlag, SplitStrategy, TaskType
from aml_toolkit.intake.split_builder import SplitResult
from aml_toolkit.profiling.class_distribution import profile_class_distribution
from aml_toolkit.profiling.drift_ood import profile_drift
from aml_toolkit.profiling.duplicates import profile_duplicates
from aml_toolkit.profiling.label_conflicts import detect_label_conflicts
from aml_toolkit.profiling.missingness import profile_missingness
from aml_toolkit.profiling.outliers import profile_outliers
from aml_toolkit.profiling.profiler_engine import run_profiling


def _make_split(n: int) -> SplitResult:
    """Simple 70/15/15 split for n samples."""
    indices = np.arange(n)
    n_test = int(n * 0.15)
    n_val = int(n * 0.15)
    return SplitResult(
        train_indices=indices[: n - n_test - n_val],
        val_indices=indices[n - n_test - n_val : n - n_test],
        test_indices=indices[n - n_test :],
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


# --- Class Distribution Tests ---


class TestClassDistribution:
    def test_balanced(self):
        labels = np.array([0] * 50 + [1] * 50)
        config = ToolkitConfig()
        result = profile_class_distribution(labels, config)
        assert result["imbalance_severity"] == "none"
        assert result["class_counts"]["0"] == 50
        assert result["class_counts"]["1"] == 50
        assert len(result["risk_flags"]) == 0

    def test_moderate_imbalance(self):
        labels = np.array([0] * 90 + [1] * 10)
        config = ToolkitConfig(profiling={"imbalance_ratio_warning": 5.0, "imbalance_ratio_severe": 20.0})
        result = profile_class_distribution(labels, config)
        assert result["imbalance_severity"] == "moderate"
        assert RiskFlag.CLASS_IMBALANCE in result["risk_flags"]

    def test_severe_imbalance(self):
        labels = np.array([0] * 980 + [1] * 20)
        config = ToolkitConfig(profiling={"imbalance_ratio_severe": 20.0})
        result = profile_class_distribution(labels, config)
        assert result["imbalance_severity"] == "severe"

    def test_multilabel(self):
        labels = np.array([[1, 0, 0]] * 80 + [[0, 1, 1]] * 20)
        config = ToolkitConfig()
        result = profile_class_distribution(labels, config)
        assert "0" in result["class_counts"]
        assert len(result["class_counts"]) == 3


# --- Missingness Tests ---


class TestMissingness:
    def test_no_missing(self):
        df = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        result = profile_missingness(df, ["f1", "f2"])
        assert result == {}

    def test_some_missing(self):
        df = pd.DataFrame({"f1": [1, None, 3], "f2": [4, 5, None]})
        result = profile_missingness(df, ["f1", "f2"])
        assert abs(result["f1"] - 1 / 3) < 0.01
        assert abs(result["f2"] - 1 / 3) < 0.01

    def test_empty_dataframe(self):
        df = pd.DataFrame({"f1": pd.Series(dtype=float)})
        result = profile_missingness(df, ["f1"])
        assert result == {}


# --- Duplicate Tests ---


class TestDuplicates:
    def test_no_duplicates(self):
        df = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        result = profile_duplicates(df, ["f1", "f2"])
        assert result["total_duplicates"] == 0

    def test_with_duplicates(self):
        df = pd.DataFrame({"f1": [1, 1, 2], "f2": [10, 10, 20]})
        result = profile_duplicates(df, ["f1", "f2"])
        assert result["total_duplicates"] == 2
        assert result["duplicate_groups"] == 1


# --- Label Conflict Tests ---


class TestLabelConflicts:
    def test_no_conflicts(self):
        df = pd.DataFrame({"f1": [1, 2, 3], "label": [0, 1, 0]})
        result = detect_label_conflicts(df, ["f1"], "label")
        assert result["conflict_count"] == 0

    def test_with_conflicts(self):
        df = pd.DataFrame({
            "f1": [1, 1, 2, 2, 3],
            "label": [0, 1, 0, 0, 1],
        })
        result = detect_label_conflicts(df, ["f1"], "label")
        assert result["conflict_groups"] == 1
        assert result["conflict_count"] == 2
        assert len(result["sample_conflicts"]) > 0

    def test_multiple_feature_conflicts(self):
        df = pd.DataFrame({
            "f1": [1, 1, 2],
            "f2": [10, 10, 20],
            "label": [0, 1, 0],
        })
        result = detect_label_conflicts(df, ["f1", "f2"], "label")
        assert result["conflict_groups"] == 1


# --- Outlier Tests ---


class TestOutliers:
    def test_no_outliers(self):
        df = pd.DataFrame({"f1": np.random.randn(100)})
        result = profile_outliers(df, ["f1"])
        # Normal distribution should have few or no IQR outliers with 100 samples
        assert result["total_outlier_fraction"] < 0.15

    def test_with_extreme_outlier(self):
        values = list(np.random.randn(99)) + [1000.0]
        df = pd.DataFrame({"f1": values})
        result = profile_outliers(df, ["f1"])
        assert result["per_column"].get("f1", 0) >= 1

    def test_non_numeric_skipped(self):
        df = pd.DataFrame({"f1": ["a", "b", "c"]})
        result = profile_outliers(df, ["f1"])
        assert result["total_outlier_rows"] == 0


# --- Drift / OOD Tests ---


class TestDrift:
    def test_no_drift(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
        train_idx = np.arange(140)
        test_idx = np.arange(140, 200)
        result = profile_drift(df, ["f1", "f2"], train_idx, test_idx)
        assert result["total_shifted"] == 0

    def test_with_drift(self):
        np.random.seed(42)
        # Train: normal(0,1), Test: normal(5,1) — massive shift
        train_vals = np.random.randn(100)
        test_vals = np.random.randn(50) + 5
        df = pd.DataFrame({"f1": np.concatenate([train_vals, test_vals])})
        train_idx = np.arange(100)
        test_idx = np.arange(100, 150)
        result = profile_drift(df, ["f1"], train_idx, test_idx)
        assert result["total_shifted"] == 1
        assert "f1" in result["shifted_features"]

    def test_empty_splits(self):
        df = pd.DataFrame({"f1": [1, 2, 3]})
        result = profile_drift(df, ["f1"], np.array([]), np.array([0, 1, 2]))
        assert result["total_shifted"] == 0


# --- Profiler Engine Integration Tests ---


class TestProfilerEngine:
    def test_full_tabular_profile(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "label": np.array([0] * 100 + [1] * 100),
        })
        split = _make_split(n)
        manifest = _make_manifest(
            feature_columns=["f1", "f2"],
            target_column="label",
            train_size=len(split.train_indices),
            val_size=len(split.val_indices),
            test_size=len(split.test_indices),
        )
        config = ToolkitConfig()
        profile = run_profiling(
            data={"df": df},
            manifest=manifest,
            split=split,
            config=config,
        )
        assert isinstance(profile, DataProfile)
        assert profile.total_samples == n
        assert profile.imbalance_severity == "none"
        assert "0" in profile.class_counts
        assert "1" in profile.class_counts

    def test_severe_imbalance_flagged(self):
        n = 200
        df = pd.DataFrame({
            "f1": np.random.randn(n),
            "label": np.array([0] * 190 + [1] * 10),
        })
        split = _make_split(n)
        manifest = _make_manifest(
            feature_columns=["f1"],
            target_column="label",
            train_size=len(split.train_indices),
            val_size=len(split.val_indices),
            test_size=len(split.test_indices),
        )
        config = ToolkitConfig()
        profile = run_profiling({"df": df}, manifest, split, config)
        assert RiskFlag.CLASS_IMBALANCE in profile.risk_flags
        assert profile.imbalance_severity in ("moderate", "severe")

    def test_label_conflicts_flagged(self):
        df = pd.DataFrame({
            "f1": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        split = _make_split(10)
        manifest = _make_manifest(
            feature_columns=["f1"],
            target_column="label",
            train_size=len(split.train_indices),
            val_size=len(split.val_indices),
            test_size=len(split.test_indices),
        )
        profile = run_profiling({"df": df}, manifest, split, ToolkitConfig())
        assert RiskFlag.LABEL_CONFLICT in profile.risk_flags

    def test_profile_serializes(self, tmp_path):
        profile = DataProfile(
            total_samples=100,
            class_counts={"0": 50, "1": 50},
            imbalance_severity="none",
            risk_flags=[RiskFlag.OOD_SHIFT],
        )
        from aml_toolkit.utils.serialization import save_artifact_json, load_artifact_json

        path = tmp_path / "profile.json"
        save_artifact_json(profile, path)
        loaded = load_artifact_json(DataProfile, path)
        assert loaded.total_samples == 100
        assert RiskFlag.OOD_SHIFT in loaded.risk_flags

    def test_ood_shift_flagged(self):
        np.random.seed(42)
        # Train: normal(0,1), Test: normal(5,1)
        train_vals = np.random.randn(100)
        test_vals = np.random.randn(50) + 5
        all_vals = np.concatenate([train_vals, test_vals])
        df = pd.DataFrame({
            "f1": all_vals,
            "label": np.array([0] * 75 + [1] * 75),
        })
        split = SplitResult(
            train_indices=np.arange(100),
            val_indices=np.array([], dtype=int),
            test_indices=np.arange(100, 150),
            strategy=SplitStrategy.STRATIFIED,
        )
        manifest = _make_manifest(
            feature_columns=["f1"],
            target_column="label",
            train_size=100,
            val_size=0,
            test_size=50,
        )
        profile = run_profiling({"df": df}, manifest, split, ToolkitConfig())
        assert RiskFlag.OOD_SHIFT in profile.risk_flags
