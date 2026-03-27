"""Tests for probe engine, baselines, tabular probes, and embedding probes."""

import numpy as np
import pandas as pd
import pytest

from aml_toolkit.artifacts import DatasetManifest, ProbeResult, ProbeResultSet
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import ModalityType, SplitStrategy, TaskType
from aml_toolkit.intake.split_builder import SplitResult
from aml_toolkit.probes.baseline_models import MajorityBaseline, StratifiedBaseline
from aml_toolkit.probes.image_embedding_probes import run_embedding_probe
from aml_toolkit.probes.probe_engine import run_probes
from aml_toolkit.probes.tabular_probes import run_tabular_probe


@pytest.fixture
def tabular_data():
    """Binary classification tabular dataset."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 5)
    # Make labels somewhat learnable
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["label"] = y

    split = SplitResult(
        train_indices=np.arange(140),
        val_indices=np.arange(140, 170),
        test_indices=np.arange(170, 200),
        strategy=SplitStrategy.STRATIFIED,
    )
    manifest = DatasetManifest(
        dataset_id="test",
        modality=ModalityType.TABULAR,
        task_type=TaskType.BINARY,
        split_strategy=SplitStrategy.STRATIFIED,
        target_column="label",
        feature_columns=[f"f{i}" for i in range(5)],
        class_labels=["0", "1"],
        train_size=140,
        val_size=30,
        test_size=30,
    )
    data = {"df": df, "features": [f"f{i}" for i in range(5)], "target": "label", "split": split}
    return data, manifest, split


@pytest.fixture
def embedding_data():
    """Embedding dataset for probe testing."""
    np.random.seed(42)
    n = 150
    embeddings = np.random.randn(n, 32)
    labels = (embeddings[:, 0] > 0).astype(int)

    split = SplitResult(
        train_indices=np.arange(100),
        val_indices=np.arange(100, 125),
        test_indices=np.arange(125, 150),
        strategy=SplitStrategy.STRATIFIED,
    )
    manifest = DatasetManifest(
        dataset_id="emb_test",
        modality=ModalityType.EMBEDDING,
        task_type=TaskType.BINARY,
        split_strategy=SplitStrategy.STRATIFIED,
        class_labels=["0", "1"],
        train_size=100,
        val_size=25,
        test_size=25,
    )
    data = {"embeddings": embeddings, "labels": labels, "split": split}
    return data, manifest, split


# --- Baseline Tests ---


class TestBaselines:
    def test_majority_baseline(self):
        y_train = np.array([0] * 80 + [1] * 20)
        y_val = np.array([0] * 15 + [1] * 5)
        maj = MajorityBaseline()
        maj.fit(y_train)
        preds = maj.predict(np.zeros(20))
        assert all(p == 0 for p in preds)

        result = maj.to_probe_result(y_train, y_val, ["accuracy", "macro_f1"])
        assert result.model_name == "majority_baseline"
        assert "accuracy" in result.val_metrics
        assert "macro_f1" in result.val_metrics

    def test_stratified_baseline(self):
        y_train = np.array([0] * 50 + [1] * 50)
        y_val = np.array([0] * 10 + [1] * 10)
        strat = StratifiedBaseline()
        strat.fit(y_train)
        preds = strat.predict(np.zeros(100))
        # Should have roughly even distribution
        unique, counts = np.unique(preds, return_counts=True)
        assert len(unique) == 2

        result = strat.to_probe_result(y_train, y_val, ["accuracy"])
        assert result.model_name == "stratified_baseline"

    def test_baseline_ranking(self):
        """Majority baseline should have lower macro_f1 than a learnable probe on imbalanced data."""
        y_train = np.array([0] * 90 + [1] * 10)
        y_val = np.array([0] * 15 + [1] * 5)

        maj = MajorityBaseline()
        maj.fit(y_train)
        maj_result = maj.to_probe_result(y_train, y_val, ["macro_f1"])

        # Majority baseline predicts all 0, so macro_f1 should be low
        assert maj_result.val_metrics["macro_f1"] < 0.6


# --- Tabular Probe Tests ---


class TestTabularProbes:
    def test_logistic_probe(self, tabular_data):
        data, manifest, split = tabular_data
        df = data["df"]
        X_train = df[manifest.feature_columns].values[split.train_indices]
        y_train = df["label"].values[split.train_indices]
        X_val = df[manifest.feature_columns].values[split.val_indices]
        y_val = df["label"].values[split.val_indices]

        result = run_tabular_probe("logistic", X_train, y_train, X_val, y_val, ["macro_f1", "accuracy"])
        assert result.model_name == "logistic"
        assert result.val_metrics["macro_f1"] > 0.5
        assert result.fit_time_seconds > 0

    def test_rf_probe(self, tabular_data):
        data, manifest, split = tabular_data
        df = data["df"]
        X_train = df[manifest.feature_columns].values[split.train_indices]
        y_train = df["label"].values[split.train_indices]
        X_val = df[manifest.feature_columns].values[split.val_indices]
        y_val = df["label"].values[split.val_indices]

        result = run_tabular_probe("rf", X_train, y_train, X_val, y_val, ["macro_f1"])
        assert result.model_name == "rf"
        assert "macro_f1" in result.val_metrics

    def test_xgb_probe(self, tabular_data):
        data, manifest, split = tabular_data
        df = data["df"]
        X_train = df[manifest.feature_columns].values[split.train_indices]
        y_train = df["label"].values[split.train_indices]
        X_val = df[manifest.feature_columns].values[split.val_indices]
        y_val = df["label"].values[split.val_indices]

        result = run_tabular_probe("xgb", X_train, y_train, X_val, y_val, ["macro_f1"])
        assert result.model_name == "xgb"
        # XGB may be unavailable (missing libomp); graceful fallback is fine
        if result.notes:
            assert "not available" in result.notes[0]
        else:
            assert "macro_f1" in result.val_metrics

    def test_probe_with_class_weighting(self, tabular_data):
        data, manifest, split = tabular_data
        df = data["df"]
        X_train = df[manifest.feature_columns].values[split.train_indices]
        y_train = df["label"].values[split.train_indices]
        X_val = df[manifest.feature_columns].values[split.val_indices]
        y_val = df["label"].values[split.val_indices]

        result = run_tabular_probe(
            "logistic", X_train, y_train, X_val, y_val,
            ["macro_f1"], intervention_branch="class_weighting",
        )
        assert result.intervention_branch == "class_weighting"
        assert "macro_f1" in result.val_metrics

    def test_unknown_probe_returns_notes(self, tabular_data):
        data, manifest, split = tabular_data
        df = data["df"]
        X_train = df[manifest.feature_columns].values[split.train_indices]
        y_train = df["label"].values[split.train_indices]
        X_val = df[manifest.feature_columns].values[split.val_indices]
        y_val = df["label"].values[split.val_indices]

        result = run_tabular_probe("nonexistent", X_train, y_train, X_val, y_val, ["macro_f1"])
        assert len(result.notes) > 0


# --- Embedding Probe Tests ---


class TestEmbeddingProbes:
    def test_embedding_logistic_probe(self, embedding_data):
        data, manifest, split = embedding_data
        X_train = data["embeddings"][split.train_indices]
        y_train = data["labels"][split.train_indices]
        X_val = data["embeddings"][split.val_indices]
        y_val = data["labels"][split.val_indices]

        result = run_embedding_probe("embedding_logistic", X_train, y_train, X_val, y_val, ["macro_f1"])
        assert result.model_name == "embedding_logistic"
        assert result.modality == "EMBEDDING"
        assert result.val_metrics["macro_f1"] > 0.4

    def test_embedding_mlp_probe(self, embedding_data):
        data, manifest, split = embedding_data
        X_train = data["embeddings"][split.train_indices]
        y_train = data["labels"][split.train_indices]
        X_val = data["embeddings"][split.val_indices]
        y_val = data["labels"][split.val_indices]

        result = run_embedding_probe("embedding_mlp", X_train, y_train, X_val, y_val, ["macro_f1"])
        assert result.model_name == "embedding_mlp"


# --- Probe Engine Integration Tests ---


class TestProbeEngine:
    def test_tabular_probe_engine(self, tabular_data):
        data, manifest, split = tabular_data
        config = ToolkitConfig(
            probes={
                "enabled_probes": ["majority", "stratified", "logistic", "rf"],
                "intervention_branches": ["none", "class_weighting"],
                "metric": "macro_f1",
            }
        )
        result_set = run_probes(data, manifest, split, config)
        assert isinstance(result_set, ProbeResultSet)
        assert len(result_set.baseline_results) == 2  # majority + stratified
        assert len(result_set.shallow_results) >= 2  # logistic + rf
        assert len(result_set.intervention_branch_results) >= 2  # class_weighting for logistic + rf
        assert "macro_f1" in result_set.selected_metrics

    def test_probe_engine_shortlist(self, tabular_data):
        data, manifest, split = tabular_data
        config = ToolkitConfig(
            probes={
                "enabled_probes": ["majority", "logistic", "rf"],
                "intervention_branches": ["none"],
                "metric": "macro_f1",
            }
        )
        result_set = run_probes(data, manifest, split, config)
        assert len(result_set.shortlist_recommendation) >= 1
        # First in shortlist should have highest macro_f1
        if len(result_set.shortlist_recommendation) > 1:
            top_name = result_set.shortlist_recommendation[0]
            top_result = next(r for r in result_set.shallow_results if r.model_name == top_name)
            for r in result_set.shallow_results:
                if r.model_name != top_name and "macro_f1" in r.val_metrics:
                    assert top_result.val_metrics["macro_f1"] >= r.val_metrics["macro_f1"]

    def test_probe_engine_config_driven_selection(self, tabular_data):
        data, manifest, split = tabular_data
        # Only enable logistic
        config = ToolkitConfig(
            probes={
                "enabled_probes": ["logistic"],
                "intervention_branches": ["none"],
                "metric": "macro_f1",
            }
        )
        result_set = run_probes(data, manifest, split, config)
        assert len(result_set.baseline_results) == 0  # no majority/stratified enabled
        assert len(result_set.shallow_results) == 1
        assert result_set.shallow_results[0].model_name == "logistic"

    def test_probe_output_serializes(self, tabular_data, tmp_path):
        data, manifest, split = tabular_data
        config = ToolkitConfig(
            probes={
                "enabled_probes": ["majority", "logistic"],
                "intervention_branches": ["none"],
                "metric": "macro_f1",
            }
        )
        result_set = run_probes(data, manifest, split, config)

        from aml_toolkit.utils.serialization import save_artifact_json, load_artifact_json

        path = tmp_path / "probes.json"
        save_artifact_json(result_set, path)
        loaded = load_artifact_json(ProbeResultSet, path)
        assert len(loaded.baseline_results) == len(result_set.baseline_results)

    def test_embedding_probe_engine(self, embedding_data):
        data, manifest, split = embedding_data
        config = ToolkitConfig(
            probes={
                "enabled_probes": ["majority", "stratified", "embedding_logistic"],
                "intervention_branches": ["none"],
                "metric": "macro_f1",
            }
        )
        result_set = run_probes(data, manifest, split, config)
        assert len(result_set.baseline_results) == 2
        assert len(result_set.shallow_results) >= 1
        emb_result = next(
            (r for r in result_set.shallow_results if r.model_name == "embedding_logistic"),
            None,
        )
        assert emb_result is not None

    def test_intervention_sensitivity_summary(self, tabular_data):
        data, manifest, split = tabular_data
        config = ToolkitConfig(
            probes={
                "enabled_probes": ["logistic"],
                "intervention_branches": ["none", "class_weighting"],
                "metric": "macro_f1",
            }
        )
        result_set = run_probes(data, manifest, split, config)
        # Should have sensitivity entry for logistic_class_weighting
        assert len(result_set.intervention_sensitivity_summary) > 0
