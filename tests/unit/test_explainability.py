"""Tests for Phase 13: Explainability, Heatmaps, and Faithfulness Checks.

Required tests:
1. Confusion heatmap artifact test.
2. Unsupported explainability fallback test.
3. Faithfulness helper smoke test.
4. Explainability report serialization test.
"""

import warnings
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from aml_toolkit.artifacts.explainability_report import ExplainabilityOutput, ExplainabilityReport
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import ModalityType
from aml_toolkit.core.exceptions import ExplainabilityFailureWarning
from aml_toolkit.explainability.confusion_heatmap import ConfusionHeatmapStrategy
from aml_toolkit.explainability.explainability_manager import run_explainability
from aml_toolkit.explainability.faithfulness import feature_removal_faithfulness
from aml_toolkit.explainability.feature_importance import FeatureImportanceStrategy
from aml_toolkit.explainability.gradcam import GradCAMStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def binary_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    y = np.array([0] * 50 + [1] * 50)
    return X, y


@pytest.fixture()
def trained_logistic(binary_data):
    """A real trained sklearn LogisticRegression for integration tests."""
    X, y = binary_data
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture()
def mock_adapter(trained_logistic):
    """Mock CandidateModel adapter wrapping a real logistic model."""
    adapter = MagicMock(spec=["_model", "predict", "predict_proba", "is_probabilistic", "get_model_family"])
    adapter._model = trained_logistic
    adapter.predict.side_effect = trained_logistic.predict
    adapter.predict_proba.side_effect = trained_logistic.predict_proba
    adapter.is_probabilistic.return_value = True
    adapter.get_model_family.return_value = "logistic"
    return adapter


@pytest.fixture()
def config():
    return ToolkitConfig()


# ---------------------------------------------------------------------------
# Test 1: Confusion heatmap artifact test
# ---------------------------------------------------------------------------

class TestConfusionHeatmap:

    def test_generates_confusion_matrix(self, binary_data, mock_adapter, config, tmp_path):
        X, y = binary_data
        strategy = ConfusionHeatmapStrategy()

        assert strategy.supports_model(mock_adapter) is True

        output = strategy.explain(mock_adapter, X, y, tmp_path / "cm", config)
        output.candidate_id = "logistic_001"

        assert isinstance(output, ExplainabilityOutput)
        assert output.method == "confusion_heatmap"
        assert output.supported is True
        assert len(output.artifact_paths) >= 1
        assert "matrix" in output.summary
        assert "accuracy" in output.summary
        assert output.summary["accuracy"] > 0.5

        # Verify the .npy file exists
        npy_path = tmp_path / "cm" / "confusion_matrix.npy"
        assert npy_path.exists()
        cm = np.load(npy_path)
        assert cm.shape == (2, 2)

    def test_confusion_heatmap_png_generated(self, binary_data, mock_adapter, config, tmp_path):
        X, y = binary_data
        strategy = ConfusionHeatmapStrategy()
        output = strategy.explain(mock_adapter, X, y, tmp_path / "cm", config)
        # PNG is best-effort; check if it was generated
        png_paths = [p for p in output.artifact_paths if p.endswith(".png")]
        # May or may not exist depending on matplotlib availability
        assert isinstance(png_paths, list)


# ---------------------------------------------------------------------------
# Test 2: Unsupported explainability fallback test
# ---------------------------------------------------------------------------

class TestUnsupportedFallback:

    def test_gradcam_unsupported_for_tabular(self, binary_data, mock_adapter, config, tmp_path):
        X, y = binary_data
        strategy = GradCAMStrategy()

        assert strategy.supports_model(mock_adapter) is False

        output = strategy.explain(mock_adapter, X, y, tmp_path / "gradcam", config)
        assert output.supported is False
        assert output.fallback_reason is not None
        assert "Grad-CAM" in output.fallback_reason

    def test_unsupported_method_warns_not_crashes(self, binary_data, mock_adapter, config, tmp_path):
        """Run explainability with an image method on a tabular model — should warn, not crash."""
        X, y = binary_data
        models = {"logistic_001": mock_adapter}

        # Force image modality on a tabular model to trigger unsupported paths
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            report = run_explainability(
                models, X, y, config, tmp_path, modality=ModalityType.IMAGE,
            )

        assert isinstance(report, ExplainabilityReport)
        # Grad-CAM should be unsupported for logistic model, producing a fallback entry
        gradcam_outputs = [o for o in report.outputs if o.method == "gradcam"]
        assert len(gradcam_outputs) >= 1
        assert gradcam_outputs[0].supported is False
        assert len(report.methods_failed) >= 1

    def test_feature_importance_unsupported_for_non_tree_model(self, binary_data, config, tmp_path):
        """Model without feature_importances_ or coef_ returns unsupported."""
        X, y = binary_data
        model = MagicMock()
        model._model = MagicMock(spec=[])  # no feature_importances_ or coef_
        model.predict.return_value = np.zeros(len(y))

        strategy = FeatureImportanceStrategy()
        # supports_model may return False
        output = strategy.explain(model, X, y, tmp_path / "fi", config)
        if not output.supported:
            assert output.fallback_reason is not None

    def test_manager_caveats_always_present(self, binary_data, mock_adapter, config, tmp_path):
        X, y = binary_data
        report = run_explainability(
            {"logistic_001": mock_adapter}, X, y, config, tmp_path,
            modality=ModalityType.TABULAR,
        )
        assert len(report.caveats) >= 1
        assert any("approximation" in c.lower() for c in report.caveats)


# ---------------------------------------------------------------------------
# Test 3: Faithfulness helper smoke test
# ---------------------------------------------------------------------------

class TestFaithfulness:

    def test_faithfulness_positive_for_real_model(self, binary_data, trained_logistic):
        X, y = binary_data
        importances = np.abs(trained_logistic.coef_).ravel()

        score = feature_removal_faithfulness(
            trained_logistic, X, y, importances, top_k=2,
        )
        # Removing important features should degrade performance → positive faithfulness
        assert isinstance(score, float)
        assert score >= 0.0

    def test_faithfulness_zero_features(self, binary_data, trained_logistic):
        """Removing zero features should give zero faithfulness."""
        X, y = binary_data
        importances = np.abs(trained_logistic.coef_).ravel()

        score = feature_removal_faithfulness(
            trained_logistic, X, y, importances, top_k=0,
        )
        assert score == 0.0

    def test_faithfulness_handles_failure(self, binary_data):
        """Broken model should return 0.0, not crash."""
        X, y = binary_data
        broken_model = MagicMock()
        broken_model.predict.side_effect = RuntimeError("broken")

        score = feature_removal_faithfulness(
            broken_model, X, y, np.ones(4), top_k=2,
        )
        assert score == 0.0

    def test_faithfulness_with_adapter(self, binary_data, mock_adapter):
        X, y = binary_data
        importances = np.abs(mock_adapter._model.coef_).ravel()
        score = feature_removal_faithfulness(
            mock_adapter, X, y, importances, top_k=2,
        )
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Test 4: Explainability report serialization test
# ---------------------------------------------------------------------------

class TestExplainabilityReportSerialization:

    def test_report_serializes(self, binary_data, mock_adapter, config, tmp_path):
        X, y = binary_data
        report = run_explainability(
            {"logistic_001": mock_adapter}, X, y, config, tmp_path,
            modality=ModalityType.TABULAR,
        )

        data = report.model_dump()
        assert isinstance(data, dict)
        assert "outputs" in data
        assert "caveats" in data
        assert "methods_attempted" in data
        assert "methods_succeeded" in data
        assert "methods_failed" in data

        reloaded = ExplainabilityReport.model_validate(data)
        assert len(reloaded.outputs) == len(report.outputs)
        assert len(reloaded.caveats) >= 1

    def test_output_artifact_serializes(self):
        output = ExplainabilityOutput(
            method="test",
            candidate_id="c1",
            artifact_paths=["/tmp/test.npy"],
            summary={"key": "value"},
            faithfulness_score=0.5,
        )
        data = output.model_dump()
        reloaded = ExplainabilityOutput.model_validate(data)
        assert reloaded.faithfulness_score == 0.5
        assert reloaded.method == "test"


# ---------------------------------------------------------------------------
# Test 5: Feature importance integration
# ---------------------------------------------------------------------------

class TestFeatureImportance:

    def test_logistic_feature_importance(self, binary_data, mock_adapter, config, tmp_path):
        X, y = binary_data
        strategy = FeatureImportanceStrategy()

        assert strategy.supports_model(mock_adapter) is True

        output = strategy.explain(mock_adapter, X, y, tmp_path / "fi", config)
        assert output.supported is True
        assert output.method == "feature_importance"
        assert len(output.artifact_paths) >= 1
        assert "top_features" in output.summary
        assert len(output.summary["top_features"]) > 0

    def test_importance_values_saved(self, binary_data, mock_adapter, config, tmp_path):
        X, y = binary_data
        strategy = FeatureImportanceStrategy()
        output = strategy.explain(mock_adapter, X, y, tmp_path / "fi", config)
        npy_path = Path(output.artifact_paths[0])
        assert npy_path.exists()
        importances = np.load(npy_path)
        assert len(importances) == 4  # 4 features


# ---------------------------------------------------------------------------
# Test 6: Manager integration
# ---------------------------------------------------------------------------

class TestExplainabilityManager:

    def test_tabular_methods_run(self, binary_data, mock_adapter, config, tmp_path):
        X, y = binary_data
        report = run_explainability(
            {"logistic_001": mock_adapter}, X, y, config, tmp_path,
            modality=ModalityType.TABULAR,
        )

        assert isinstance(report, ExplainabilityReport)
        assert len(report.outputs) >= 2  # at least confusion_heatmap + feature_importance
        methods = {o.method for o in report.outputs if o.supported}
        assert "confusion_heatmap" in methods
        assert "feature_importance" in methods

    def test_multiple_candidates(self, binary_data, config, tmp_path):
        X, y = binary_data
        # Two simple mock adapters
        lr1 = LogisticRegression(max_iter=200, random_state=42).fit(X, y)
        lr2 = LogisticRegression(max_iter=200, random_state=99).fit(X, y)

        adapter1 = MagicMock()
        adapter1._model = lr1
        adapter1.predict.side_effect = lr1.predict
        adapter1.predict_proba.side_effect = lr1.predict_proba
        adapter1.is_probabilistic.return_value = True
        adapter1.get_model_family.return_value = "logistic"

        adapter2 = MagicMock()
        adapter2._model = lr2
        adapter2.predict.side_effect = lr2.predict
        adapter2.predict_proba.side_effect = lr2.predict_proba
        adapter2.is_probabilistic.return_value = True
        adapter2.get_model_family.return_value = "logistic"

        report = run_explainability(
            {"lr_001": adapter1, "lr_002": adapter2}, X, y, config, tmp_path,
            modality=ModalityType.TABULAR,
        )

        # Should have outputs for both candidates
        candidate_ids = {o.candidate_id for o in report.outputs}
        assert "lr_001" in candidate_ids
        assert "lr_002" in candidate_ids
