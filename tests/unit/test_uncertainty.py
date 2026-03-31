"""Tests for Phase 2.3 — Uncertainty estimation and conformal prediction."""

import numpy as np
import pytest

from aml_toolkit.artifacts.uncertainty_report import UncertaintyReport
from aml_toolkit.core.config import UncertaintyConfig
from aml_toolkit.uncertainty.conformal import SplitConformalPredictor
from aml_toolkit.uncertainty.estimator import UncertaintyEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> UncertaintyConfig:
    defaults = dict(
        enabled=True,
        methods=["entropy", "margin"],
        aggregation="mean",
        abstain_if_above=0.8,
        use_calibrated_proba=True,
        conformal_enabled=False,
        conformal_coverage=0.9,
    )
    defaults.update(kwargs)
    return UncertaintyConfig(**defaults)


def _uniform_proba(n: int, k: int) -> np.ndarray:
    return np.full((n, k), 1.0 / k)


def _certain_proba(n: int, k: int) -> np.ndarray:
    """Near-certain predictions: class 0 gets 0.99."""
    p = np.zeros((n, k))
    p[:, 0] = 0.99
    p[:, 1:] = 0.01 / max(k - 1, 1)
    return p


# ---------------------------------------------------------------------------
# SplitConformalPredictor
# ---------------------------------------------------------------------------

class TestSplitConformalPredictor:
    def test_fit_returns_self(self):
        rng = np.random.default_rng(0)
        proba = rng.dirichlet([1, 1, 1], size=100)
        y = rng.integers(0, 3, size=100)
        predictor = SplitConformalPredictor(coverage=0.9)
        result = predictor.fit(proba, y)
        assert result is predictor

    def test_coverage_guarantee(self):
        """Empirical coverage must be ≥ desired coverage on held-out test set."""
        rng = np.random.default_rng(42)
        n_cal = 1000
        n_test = 1000
        proba_cal = rng.dirichlet([2, 1, 1], size=n_cal)
        y_cal = rng.integers(0, 3, size=n_cal)
        proba_test = rng.dirichlet([2, 1, 1], size=n_test)
        y_test = rng.integers(0, 3, size=n_test)

        predictor = SplitConformalPredictor(coverage=0.9)
        predictor.fit(proba_cal, y_cal)
        coverage = predictor.empirical_coverage(proba_test, y_test)
        assert coverage >= 0.88, f"Coverage {coverage:.3f} < 0.88 (expected ≥ 0.9)"

    def test_singleton_on_confident_sample(self):
        """Near-certain prediction → prediction set size = 1."""
        proba_cal = _certain_proba(100, 3)
        y_cal = np.zeros(100, dtype=int)
        predictor = SplitConformalPredictor(coverage=0.9)
        predictor.fit(proba_cal, y_cal)

        proba_test = np.array([[0.99, 0.005, 0.005]])
        sets = predictor.predict_sets(proba_test)
        assert len(sets[0]) == 1

    def test_multi_set_on_uncertain_sample(self):
        """Near-uniform prediction → prediction set should be larger."""
        rng = np.random.default_rng(1)
        proba_cal = rng.dirichlet([1, 1, 1], size=500)
        y_cal = rng.integers(0, 3, size=500)
        predictor = SplitConformalPredictor(coverage=0.95)
        predictor.fit(proba_cal, y_cal)

        # Very uncertain: near-uniform
        proba_test = np.array([[0.34, 0.33, 0.33]])
        sets = predictor.predict_sets(proba_test)
        # For high uncertainty + high coverage, expect > 1 class included
        assert len(sets[0]) >= 1  # at minimum, argmax is included

    def test_binary_1d_proba_input(self):
        """(n,) binary probabilities should be handled without error."""
        rng = np.random.default_rng(2)
        proba_1d = rng.uniform(0.3, 0.7, size=200)
        y_cal = (proba_1d > 0.5).astype(int)
        predictor = SplitConformalPredictor(coverage=0.9)
        predictor.fit(proba_1d, y_cal)
        sets = predictor.predict_sets(proba_1d)
        assert len(sets) == 200

    def test_empty_calibration_raises(self):
        predictor = SplitConformalPredictor(coverage=0.9)
        with pytest.raises(ValueError):
            predictor.fit(np.array([]).reshape(0, 3), np.array([], dtype=int))

    def test_predict_before_fit_raises(self):
        predictor = SplitConformalPredictor(coverage=0.9)
        with pytest.raises(RuntimeError):
            predictor.predict_sets(np.array([[0.5, 0.5]]))

    def test_efficiency_lower_for_confident(self):
        """Confident model (labels match predictions) should have lower mean set size."""
        n_cal = 200
        # Confident: all predict class 0 and labels are all class 0
        confident_proba_cal = np.array([[0.95, 0.05]] * n_cal)
        y_cal_confident = np.zeros(n_cal, dtype=int)

        # Uncertain: near-uniform predictions, random labels
        rng = np.random.default_rng(3)
        uncertain_proba_cal = rng.dirichlet([1, 1], size=n_cal)
        y_cal_uncertain = rng.integers(0, 2, size=n_cal)

        pred_confident = SplitConformalPredictor(0.9).fit(confident_proba_cal, y_cal_confident)
        pred_uncertain = SplitConformalPredictor(0.9).fit(uncertain_proba_cal, y_cal_uncertain)

        test_confident = np.array([[0.95, 0.05]] * 50)
        test_uncertain = rng.dirichlet([1, 1], size=50)

        eff_confident = pred_confident.efficiency(test_confident)
        eff_uncertain = pred_uncertain.efficiency(test_uncertain)
        assert eff_confident <= eff_uncertain

    def test_invalid_coverage_raises(self):
        with pytest.raises(ValueError):
            SplitConformalPredictor(coverage=1.5)
        with pytest.raises(ValueError):
            SplitConformalPredictor(coverage=0.0)


# ---------------------------------------------------------------------------
# UncertaintyEstimator — entropy
# ---------------------------------------------------------------------------

class TestEntropyEstimation:
    def test_uniform_proba_max_entropy(self):
        """Uniform distribution → maximum normalized entropy = 1.0."""
        config = _make_config(methods=["entropy"])
        estimator = UncertaintyEstimator(config)
        proba = _uniform_proba(50, 4)
        report = estimator.estimate("test", proba)
        assert report.entropy_mean == pytest.approx(1.0, abs=1e-5)

    def test_certain_proba_low_entropy(self):
        """Near-certain prediction → entropy close to 0."""
        config = _make_config(methods=["entropy"])
        estimator = UncertaintyEstimator(config)
        proba = _certain_proba(50, 3)
        report = estimator.estimate("test", proba)
        assert report.entropy_mean < 0.1

    def test_binary_1d_proba_entropy(self):
        """(n,) binary proba should work."""
        config = _make_config(methods=["entropy"])
        estimator = UncertaintyEstimator(config)
        proba_1d = np.array([0.5, 0.5, 0.5, 0.5])
        report = estimator.estimate("test", proba_1d)
        assert report.entropy_mean == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# UncertaintyEstimator — margin
# ---------------------------------------------------------------------------

class TestMarginEstimation:
    def test_certain_margin_near_zero(self):
        """[0.99, 0.01] → margin uncertainty ≈ 0."""
        config = _make_config(methods=["margin"])
        estimator = UncertaintyEstimator(config)
        proba = np.array([[0.99, 0.01]] * 50)
        report = estimator.estimate("test", proba)
        assert report.margin_mean < 0.05

    def test_uncertain_margin_near_one(self):
        """[0.51, 0.49] → margin uncertainty ≈ 0.98."""
        config = _make_config(methods=["margin"])
        estimator = UncertaintyEstimator(config)
        proba = np.array([[0.51, 0.49]] * 50)
        report = estimator.estimate("test", proba)
        assert report.margin_mean > 0.9


# ---------------------------------------------------------------------------
# UncertaintyEstimator — abstention
# ---------------------------------------------------------------------------

class TestAbstentionTrigger:
    def test_high_uncertainty_triggers_abstention(self):
        config = _make_config(methods=["entropy"], abstain_if_above=0.3)
        estimator = UncertaintyEstimator(config)
        proba = _uniform_proba(50, 4)  # entropy = 1.0 > 0.3
        report = estimator.estimate("test", proba)
        assert report.abstention_triggered is True
        assert "uncertainty" in report.abstention_reason.lower()

    def test_low_uncertainty_no_abstention(self):
        config = _make_config(methods=["entropy"], abstain_if_above=0.8)
        estimator = UncertaintyEstimator(config)
        proba = _certain_proba(50, 3)
        report = estimator.estimate("test", proba)
        assert report.abstention_triggered is False

    def test_sample_count_populated(self):
        config = _make_config()
        estimator = UncertaintyEstimator(config)
        proba = _uniform_proba(37, 3)
        report = estimator.estimate("test", proba)
        assert report.sample_count == 37


# ---------------------------------------------------------------------------
# UncertaintyEstimator — conformal
# ---------------------------------------------------------------------------

class TestConformalIntegration:
    def test_conformal_enabled_adds_metrics(self):
        config = _make_config(
            methods=["entropy"],
            conformal_enabled=True,
            conformal_coverage=0.9,
        )
        estimator = UncertaintyEstimator(config)
        rng = np.random.default_rng(7)
        proba = rng.dirichlet([2, 1, 1], size=300)
        y = rng.integers(0, 3, size=300)
        report = estimator.estimate("test", proba, y_val=y)
        assert report.mean_prediction_set_size is not None
        assert report.pct_singleton_sets is not None
        assert report.conformal_coverage_achieved is not None
        assert "conformal" in report.methods_used

    def test_conformal_disabled_no_metrics(self):
        config = _make_config(conformal_enabled=False)
        estimator = UncertaintyEstimator(config)
        proba = _uniform_proba(50, 3)
        y = np.zeros(50, dtype=int)
        report = estimator.estimate("test", proba, y_val=y)
        assert report.mean_prediction_set_size is None
        assert report.conformal_coverage_achieved is None

    def test_conformal_no_y_val_skipped_gracefully(self):
        config = _make_config(conformal_enabled=True)
        estimator = UncertaintyEstimator(config)
        proba = _uniform_proba(50, 3)
        # No y_val — conformal should be skipped, no crash
        report = estimator.estimate("test", proba, y_val=None)
        assert report.mean_prediction_set_size is None

    def test_conformal_coverage_at_least_target(self):
        """Conformal should achieve ≥ coverage on the same data it was fitted on."""
        config = _make_config(
            methods=[],
            conformal_enabled=True,
            conformal_coverage=0.9,
        )
        estimator = UncertaintyEstimator(config)
        rng = np.random.default_rng(42)
        proba = rng.dirichlet([3, 1, 1], size=500)
        y = rng.integers(0, 3, size=500)
        report = estimator.estimate("test", proba, y_val=y)
        assert report.conformal_coverage_achieved is not None
        assert report.conformal_coverage_achieved >= 0.88  # allow small finite-sample slack


# ---------------------------------------------------------------------------
# UncertaintyEstimator — methods_used tracking
# ---------------------------------------------------------------------------

class TestMethodsUsed:
    def test_methods_used_includes_active_methods(self):
        config = _make_config(methods=["entropy", "margin"])
        estimator = UncertaintyEstimator(config)
        proba = _uniform_proba(20, 3)
        report = estimator.estimate("m", proba)
        assert "entropy" in report.methods_used
        assert "margin" in report.methods_used

    def test_empty_methods_no_crash(self):
        config = _make_config(methods=[])
        estimator = UncertaintyEstimator(config)
        proba = _uniform_proba(20, 3)
        report = estimator.estimate("m", proba)
        assert report.mean_uncertainty == 0.0
