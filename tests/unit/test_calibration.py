"""Tests for Phase 11: Calibration and Threshold Optimization.

Required tests:
1. Calibration report generation.
2. Threshold optimization test.
3. ECE/Brier primary objective config test.
4. Non-probabilistic candidate handling test.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from aml_toolkit.artifacts import CalibrationReport, CalibrationResult
from aml_toolkit.calibration.calibration_manager import run_calibration
from aml_toolkit.calibration.isotonic import IsotonicCalibrator
from aml_toolkit.calibration.metrics import brier_score, expected_calibration_error
from aml_toolkit.calibration.temperature_scaling import TemperatureScalingCalibrator
from aml_toolkit.calibration.threshold_optimizer import ThresholdOptimizer
from aml_toolkit.core.config import ToolkitConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_probabilistic_model(proba: np.ndarray):
    """Create a mock CandidateModel that returns the given probabilities."""
    model = MagicMock()
    model.is_probabilistic.return_value = True
    model.predict_proba.return_value = proba
    return model


def _make_non_probabilistic_model():
    model = MagicMock()
    model.is_probabilistic.return_value = False
    model.predict_proba.return_value = None
    return model


@pytest.fixture()
def binary_data():
    """Validation data for binary classification with known probabilities."""
    rng = np.random.RandomState(42)
    n = 200
    y_val = np.array([0] * 100 + [1] * 100)
    # Decent model: positive class gets higher probabilities
    proba_pos = np.where(y_val == 1, rng.uniform(0.5, 0.95, n), rng.uniform(0.05, 0.5, n))
    proba_neg = 1.0 - proba_pos
    proba_2d = np.column_stack([proba_neg, proba_pos])
    X_val = rng.randn(n, 4)
    return X_val, y_val, proba_2d, proba_pos


@pytest.fixture()
def config():
    return ToolkitConfig()


# ---------------------------------------------------------------------------
# Test 1: Calibration report generation
# ---------------------------------------------------------------------------

class TestCalibrationReportGeneration:
    """Verify run_calibration produces a complete CalibrationReport."""

    def test_single_candidate_report(self, binary_data, config):
        X_val, y_val, proba_2d, _ = binary_data
        model = _make_probabilistic_model(proba_2d)

        report = run_calibration({"logistic_001": model}, X_val, y_val, config)

        assert isinstance(report, CalibrationReport)
        assert len(report.results) == 1
        result = report.results[0]
        assert result.candidate_id == "logistic_001"
        assert result.method in ("temperature_scaling", "isotonic")
        assert result.ece_before is not None
        assert result.ece_after is not None
        assert result.brier_before is not None
        assert result.brier_after is not None

    def test_multiple_candidates_report(self, binary_data, config):
        X_val, y_val, proba_2d, _ = binary_data
        models = {
            "logistic_001": _make_probabilistic_model(proba_2d),
            "rf_001": _make_probabilistic_model(proba_2d),
        }

        report = run_calibration(models, X_val, y_val, config)
        assert len(report.results) == 2
        ids = {r.candidate_id for r in report.results}
        assert ids == {"logistic_001", "rf_001"}

    def test_report_serializes(self, binary_data, config):
        X_val, y_val, proba_2d, _ = binary_data
        model = _make_probabilistic_model(proba_2d)

        report = run_calibration({"logistic_001": model}, X_val, y_val, config)
        data = report.model_dump()
        assert isinstance(data, dict)
        reloaded = CalibrationReport.model_validate(data)
        assert len(reloaded.results) == 1

    def test_calibration_improves_or_maintains_ece(self, binary_data, config):
        """Calibration should not significantly worsen ECE."""
        X_val, y_val, proba_2d, _ = binary_data
        model = _make_probabilistic_model(proba_2d)

        report = run_calibration({"logistic_001": model}, X_val, y_val, config)
        result = report.results[0]
        # Allow small tolerance - calibration should generally help
        assert result.ece_after <= result.ece_before + 0.05


# ---------------------------------------------------------------------------
# Test 2: Threshold optimization
# ---------------------------------------------------------------------------

class TestThresholdOptimization:

    def test_threshold_optimizer_finds_reasonable_value(self, binary_data):
        _, y_val, _, proba_pos = binary_data
        optimizer = ThresholdOptimizer(metric="f1")
        best = optimizer.optimize(y_val, proba_pos)

        assert 0.01 <= best <= 0.99
        assert optimizer.best_score > 0.5  # should find a decent F1

    def test_threshold_optimizer_default_is_near_05(self):
        """For a well-separated problem, optimal threshold should be near 0.5."""
        rng = np.random.RandomState(99)
        y = np.array([0] * 100 + [1] * 100)
        proba = np.where(y == 1, rng.uniform(0.7, 1.0, 200), rng.uniform(0.0, 0.3, 200))

        optimizer = ThresholdOptimizer(metric="f1")
        best = optimizer.optimize(y, proba)
        assert 0.2 <= best <= 0.8

    def test_threshold_in_calibration_report(self, binary_data, config):
        """Threshold should appear in the calibration result."""
        X_val, y_val, proba_2d, _ = binary_data
        model = _make_probabilistic_model(proba_2d)

        report = run_calibration({"logistic_001": model}, X_val, y_val, config)
        result = report.results[0]
        assert result.threshold_optimized is not None
        assert 0.01 <= result.threshold_optimized <= 0.99

    def test_macro_f1_metric(self):
        rng = np.random.RandomState(42)
        y = np.array([0] * 50 + [1] * 50)
        proba = np.where(y == 1, rng.uniform(0.6, 0.9, 100), rng.uniform(0.1, 0.4, 100))

        optimizer = ThresholdOptimizer(metric="macro_f1")
        best = optimizer.optimize(y, proba)
        assert 0.01 <= best <= 0.99
        assert optimizer.best_score > 0.0


# ---------------------------------------------------------------------------
# Test 3: ECE/Brier primary objective config test
# ---------------------------------------------------------------------------

class TestPrimaryObjectiveConfig:

    def test_default_primary_objective_is_ece(self, config):
        assert config.calibration.primary_objective == "ece"

    def test_brier_primary_objective(self, binary_data):
        config = ToolkitConfig(calibration={"primary_objective": "brier"})
        X_val, y_val, proba_2d, _ = binary_data
        model = _make_probabilistic_model(proba_2d)

        report = run_calibration({"logistic_001": model}, X_val, y_val, config)
        assert report.primary_objective == "brier"
        assert report.results[0].objective_metric == "brier"

    def test_ece_primary_objective(self, binary_data, config):
        X_val, y_val, proba_2d, _ = binary_data
        model = _make_probabilistic_model(proba_2d)

        report = run_calibration({"logistic_001": model}, X_val, y_val, config)
        assert report.primary_objective == "ece"
        assert report.results[0].objective_metric == "ece"

    def test_different_objectives_may_select_different_methods(self, binary_data):
        """Different primary objectives can lead to different calibration method selection."""
        X_val, y_val, proba_2d, _ = binary_data
        model_ece = _make_probabilistic_model(proba_2d)
        model_brier = _make_probabilistic_model(proba_2d)

        config_ece = ToolkitConfig(calibration={"primary_objective": "ece"})
        config_brier = ToolkitConfig(calibration={"primary_objective": "brier"})

        report_ece = run_calibration({"c1": model_ece}, X_val, y_val, config_ece)
        report_brier = run_calibration({"c1": model_brier}, X_val, y_val, config_brier)

        # Both should produce valid results (method may or may not differ)
        assert report_ece.results[0].method in ("temperature_scaling", "isotonic")
        assert report_brier.results[0].method in ("temperature_scaling", "isotonic")


# ---------------------------------------------------------------------------
# Test 4: Non-probabilistic candidate handling
# ---------------------------------------------------------------------------

class TestNonProbabilisticHandling:

    def test_non_probabilistic_model_skipped(self, binary_data, config):
        X_val, y_val, _, _ = binary_data
        model = _make_non_probabilistic_model()

        report = run_calibration({"svm_001": model}, X_val, y_val, config)
        assert len(report.results) == 1
        result = report.results[0]
        assert result.method == "none"
        assert result.ece_before is None
        assert result.ece_after is None
        assert any("Non-probabilistic" in n for n in result.notes)

    def test_predict_proba_returns_none_skipped(self, binary_data, config):
        X_val, y_val, _, _ = binary_data
        model = MagicMock()
        model.is_probabilistic.return_value = True
        model.predict_proba.return_value = None

        report = run_calibration({"broken_001": model}, X_val, y_val, config)
        result = report.results[0]
        assert result.method == "none"
        assert any("None" in n for n in result.notes)

    def test_mixed_probabilistic_and_non(self, binary_data, config):
        X_val, y_val, proba_2d, _ = binary_data
        models = {
            "logistic_001": _make_probabilistic_model(proba_2d),
            "svm_001": _make_non_probabilistic_model(),
        }

        report = run_calibration(models, X_val, y_val, config)
        assert len(report.results) == 2

        prob_result = next(r for r in report.results if r.candidate_id == "logistic_001")
        nonprob_result = next(r for r in report.results if r.candidate_id == "svm_001")

        assert prob_result.method in ("temperature_scaling", "isotonic")
        assert nonprob_result.method == "none"


# ---------------------------------------------------------------------------
# Test 5: Metrics unit tests
# ---------------------------------------------------------------------------

class TestCalibrationMetrics:

    def test_perfect_calibration_ece_zero(self):
        """Perfectly calibrated predictions should have ECE near zero."""
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        proba = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ece = expected_calibration_error(y, proba)
        assert ece < 0.05

    def test_bad_calibration_ece_high(self):
        """Overconfident wrong predictions should have high ECE."""
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        proba = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
        ece = expected_calibration_error(y, proba)
        assert ece > 0.5

    def test_brier_perfect(self):
        y = np.array([0, 1, 0, 1])
        proba = np.array([0.0, 1.0, 0.0, 1.0])
        assert brier_score(y, proba) == 0.0

    def test_brier_worst(self):
        y = np.array([0, 1, 0, 1])
        proba = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(y, proba) == 1.0

    def test_empty_inputs(self):
        assert expected_calibration_error(np.array([]), np.array([])) == 0.0
        assert brier_score(np.array([]), np.array([])) == 0.0


# ---------------------------------------------------------------------------
# Test 6: Individual calibrator tests
# ---------------------------------------------------------------------------

class TestTemperatureScaling:

    def test_fit_and_calibrate(self, binary_data):
        _, y_val, _, proba_pos = binary_data
        cal = TemperatureScalingCalibrator()
        cal.fit(proba_pos, y_val)

        assert cal.temperature > 0
        calibrated = cal.calibrate(proba_pos)
        assert calibrated.shape == proba_pos.shape
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_evaluate_returns_result(self, binary_data):
        _, y_val, _, proba_pos = binary_data
        cal = TemperatureScalingCalibrator()
        cal.fit(proba_pos, y_val)
        calibrated = cal.calibrate(proba_pos)

        result = cal.evaluate(proba_pos, calibrated, y_val)
        assert isinstance(result, CalibrationResult)
        assert result.method == "temperature_scaling"
        assert result.ece_before is not None
        assert result.ece_after is not None

    def test_method_name(self):
        assert TemperatureScalingCalibrator().method_name() == "temperature_scaling"


class TestIsotonicCalibrator:

    def test_fit_and_calibrate(self, binary_data):
        _, y_val, _, proba_pos = binary_data
        cal = IsotonicCalibrator()
        cal.fit(proba_pos, y_val)

        calibrated = cal.calibrate(proba_pos)
        assert calibrated.shape == proba_pos.shape
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_unfitted_raises(self):
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.calibrate(np.array([0.5]))

    def test_method_name(self):
        assert IsotonicCalibrator().method_name() == "isotonic"
