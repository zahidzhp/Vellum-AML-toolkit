"""Interface conformance tests using lightweight dummy implementations."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aml_toolkit.artifacts import (
    CalibrationResult,
    DataProfile,
    DatasetManifest,
    EnsembleReport,
    FinalReport,
    ProbeResult,
)
from aml_toolkit.artifacts.explainability_report import ExplainabilityOutput
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import (
    InterventionType,
    ModalityType,
    SplitStrategy,
    TaskType,
)
from aml_toolkit.interfaces import (
    Calibrator,
    CandidateModel,
    DatasetLoader,
    EnsembleStrategy,
    ExplainabilityStrategy,
    Intervention,
    ModelFamilyMetadata,
    ModelRegistry,
    ProbeModel,
    Profiler,
    Reporter,
)


# --- Dummy implementations ---


class DummyDatasetLoader(DatasetLoader):
    def load(self, config):
        manifest = DatasetManifest(
            dataset_id="dummy",
            modality=ModalityType.TABULAR,
            task_type=TaskType.BINARY,
            split_strategy=SplitStrategy.STRATIFIED,
        )
        return manifest, {"X": [], "y": []}

    def supports_modality(self, modality):
        return modality == "TABULAR"


class DummyProfiler(Profiler):
    def profile(self, data, manifest, config):
        return DataProfile(total_samples=100)

    def name(self):
        return "dummy_profiler"


class DummyProbeModel(ProbeModel):
    def fit(self, X_train, y_train, config):
        pass

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return None

    def evaluate(self, X, y, metrics):
        return {"accuracy": 0.5}

    def name(self):
        return "dummy_probe"

    def to_probe_result(self, intervention_branch="none"):
        return ProbeResult(
            model_name="dummy_probe",
            intervention_branch=intervention_branch,
            val_metrics={"accuracy": 0.5},
        )


class DummyIntervention(Intervention):
    def apply(self, X_train, y_train, config):
        return X_train, y_train

    def intervention_type(self):
        return InterventionType.CLASS_WEIGHTING

    def is_applicable(self, config):
        return True


class DummyCandidateModel(CandidateModel):
    def fit(self, X_train, y_train, X_val, y_val, config):
        pass

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)

    def evaluate(self, X, y, metrics):
        return {"macro_f1": 0.5}

    def get_training_trace(self):
        return {"val_loss": [0.7, 0.6, 0.5]}

    def get_model_family(self):
        return "dummy"

    def is_probabilistic(self):
        return True

    def serialize(self, path):
        pass


class DummyCalibrator(Calibrator):
    def fit(self, probabilities, y_true):
        pass

    def calibrate(self, probabilities):
        return probabilities

    def evaluate(self, probabilities_before, probabilities_after, y_true):
        return CalibrationResult(
            candidate_id="dummy",
            method="dummy",
            ece_before=0.1,
            ece_after=0.05,
        )

    def method_name(self):
        return "dummy_calibrator"


class DummyEnsembleStrategy(EnsembleStrategy):
    def combine(self, predictions, weights=None):
        return predictions[0]

    def evaluate_gain(self, individual_scores, ensemble_score, config):
        return True

    def to_report(self, member_ids, individual_scores, ensemble_score, selected):
        return EnsembleReport(
            ensemble_selected=selected,
            strategy="dummy",
            member_ids=member_ids,
        )

    def strategy_name(self):
        return "dummy_ensemble"


class DummyExplainabilityStrategy(ExplainabilityStrategy):
    def explain(self, model, X, y, output_dir, config):
        return ExplainabilityOutput(
            method="dummy",
            candidate_id="dummy",
        )

    def supports_model(self, model):
        return True

    def method_name(self):
        return "dummy_explain"


class DummyReporter(Reporter):
    def generate(self, artifacts, output_dir, config):
        return FinalReport(run_id="dummy_run")

    def format_name(self):
        return "dummy"


# --- Tests ---


class TestDatasetLoader:
    def test_dummy_loads(self):
        loader = DummyDatasetLoader()
        config = ToolkitConfig()
        manifest, data = loader.load(config)
        assert manifest.dataset_id == "dummy"
        assert isinstance(manifest, DatasetManifest)

    def test_supports_modality(self):
        loader = DummyDatasetLoader()
        assert loader.supports_modality("TABULAR") is True
        assert loader.supports_modality("IMAGE") is False


class TestProfiler:
    def test_dummy_profiles(self):
        profiler = DummyProfiler()
        manifest = DatasetManifest(
            dataset_id="test",
            modality=ModalityType.TABULAR,
            task_type=TaskType.BINARY,
            split_strategy=SplitStrategy.STRATIFIED,
        )
        profile = profiler.profile({}, manifest, ToolkitConfig())
        assert profile.total_samples == 100
        assert profiler.name() == "dummy_profiler"


class TestProbeModel:
    def test_dummy_probe_lifecycle(self):
        probe = DummyProbeModel()
        probe.fit([1, 2, 3], [0, 1, 0], ToolkitConfig())
        preds = probe.predict([1, 2])
        assert len(preds) == 2
        assert probe.predict_proba([1]) is None
        metrics = probe.evaluate([1], [0], ["accuracy"])
        assert "accuracy" in metrics
        result = probe.to_probe_result("class_weighting")
        assert result.intervention_branch == "class_weighting"


class TestIntervention:
    def test_dummy_intervention(self):
        intervention = DummyIntervention()
        X, y = intervention.apply([1, 2], [0, 1], ToolkitConfig())
        assert X == [1, 2]
        assert intervention.intervention_type() == InterventionType.CLASS_WEIGHTING
        assert intervention.is_applicable(ToolkitConfig()) is True


class TestCandidateModel:
    def test_dummy_candidate_lifecycle(self):
        model = DummyCandidateModel()
        model.fit([1, 2], [0, 1], [3], [1], ToolkitConfig())
        preds = model.predict([1, 2])
        assert len(preds) == 2
        proba = model.predict_proba([1, 2])
        assert proba.shape == (2, 2)
        trace = model.get_training_trace()
        assert "val_loss" in trace
        assert model.get_model_family() == "dummy"
        assert model.is_probabilistic() is True


class TestCalibrator:
    def test_dummy_calibrator(self):
        cal = DummyCalibrator()
        probs = np.array([0.3, 0.7])
        cal.fit(probs, np.array([0, 1]))
        calibrated = cal.calibrate(probs)
        np.testing.assert_array_equal(calibrated, probs)
        result = cal.evaluate(probs, calibrated, np.array([0, 1]))
        assert result.ece_before == 0.1
        assert cal.method_name() == "dummy_calibrator"


class TestEnsembleStrategy:
    def test_dummy_ensemble(self):
        strategy = DummyEnsembleStrategy()
        preds = [np.array([0, 1]), np.array([1, 0])]
        combined = strategy.combine(preds)
        assert len(combined) == 2
        assert strategy.evaluate_gain({"a": 0.8}, 0.85, ToolkitConfig()) is True
        report = strategy.to_report(["a", "b"], {"a": 0.8, "b": 0.7}, 0.85, True)
        assert report.ensemble_selected is True
        assert strategy.strategy_name() == "dummy_ensemble"


class TestExplainabilityStrategy:
    def test_dummy_explainability(self):
        strategy = DummyExplainabilityStrategy()
        assert strategy.supports_model(None) is True
        output = strategy.explain(None, None, None, Path("/tmp"), ToolkitConfig())
        assert output.method == "dummy"
        assert strategy.method_name() == "dummy_explain"


class TestReporter:
    def test_dummy_reporter(self):
        reporter = DummyReporter()
        report = reporter.generate({}, Path("/tmp"), ToolkitConfig())
        assert report.run_id == "dummy_run"
        assert reporter.format_name() == "dummy"


class TestModelRegistry:
    def test_register_and_retrieve(self):
        registry = ModelRegistry()
        metadata = ModelFamilyMetadata(
            family_name="dummy",
            display_name="Dummy Model",
            supported_modalities=[ModalityType.TABULAR],
        )
        registry.register("dummy", DummyCandidateModel, metadata)
        assert registry.get_adapter("dummy") is DummyCandidateModel
        assert registry.get_metadata("dummy").display_name == "Dummy Model"

    def test_list_families(self):
        registry = ModelRegistry()
        metadata = ModelFamilyMetadata(
            family_name="dummy",
            display_name="Dummy",
            supported_modalities=[ModalityType.TABULAR, ModalityType.EMBEDDING],
        )
        registry.register("dummy", DummyCandidateModel, metadata)
        assert "dummy" in registry.list_families()
        assert "dummy" in registry.list_families_for_modality(ModalityType.TABULAR)
        assert "dummy" not in registry.list_families_for_modality(ModalityType.IMAGE)

    def test_missing_family_raises(self):
        registry = ModelRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get_adapter("nonexistent")
        with pytest.raises(KeyError, match="not registered"):
            registry.get_metadata("nonexistent")

    def test_neural_metadata(self):
        registry = ModelRegistry()
        metadata = ModelFamilyMetadata(
            family_name="cnn",
            display_name="CNN",
            supported_modalities=[ModalityType.IMAGE],
            is_neural=True,
            default_warmup_epochs=10,
            supports_gradcam=True,
        )
        registry.register("cnn", DummyCandidateModel, metadata)
        meta = registry.get_metadata("cnn")
        assert meta.is_neural is True
        assert meta.default_warmup_epochs == 10
        assert meta.supports_gradcam is True
