"""Tests for Phase 12: Ensemble Builder.

Required tests:
1. Ensemble accepted on real gain case.
2. Ensemble rejected on marginal gain case.
3. Ensemble report serialization test.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from aml_toolkit.artifacts import EnsembleReport
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.ensemble.ensemble_manager import run_ensemble
from aml_toolkit.ensemble.soft_voting import SoftVotingStrategy
from aml_toolkit.ensemble.weighted_averaging import WeightedAveragingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(proba: np.ndarray, score: float = 0.7):
    """Create a mock CandidateModel returning given probabilities."""
    model = MagicMock()
    model.is_probabilistic.return_value = True
    model.predict_proba.return_value = proba
    preds = (proba[:, 1] >= 0.5).astype(int) if proba.ndim == 2 else (proba >= 0.5).astype(int)
    model.predict.return_value = preds
    return model


def _make_non_prob_model():
    model = MagicMock()
    model.is_probabilistic.return_value = False
    model.predict_proba.return_value = None
    return model


@pytest.fixture()
def binary_labels():
    return np.array([0] * 50 + [1] * 50)


@pytest.fixture()
def X_val():
    rng = np.random.RandomState(42)
    return rng.randn(100, 4)


@pytest.fixture()
def config():
    return ToolkitConfig()


# ---------------------------------------------------------------------------
# Test 1: Ensemble accepted on real gain case
# ---------------------------------------------------------------------------

class TestEnsembleAccepted:

    def test_complementary_models_produce_gain(self, X_val, binary_labels, config):
        """Two models with complementary errors should yield measurable gain."""
        n = len(binary_labels)
        rng = np.random.RandomState(42)

        # Model A: decent on class 0 but makes errors on class 1
        proba_a_pos = np.zeros(n)
        proba_a_pos[:50] = rng.uniform(0.05, 0.2, 50)   # class 0 → low prob (correct)
        proba_a_pos[50:] = rng.uniform(0.35, 0.55, 50)   # class 1 → weak/wrong

        # Model B: decent on class 1 but makes errors on class 0
        proba_b_pos = np.zeros(n)
        proba_b_pos[:50] = rng.uniform(0.4, 0.6, 50)     # class 0 → wrong
        proba_b_pos[50:] = rng.uniform(0.8, 0.95, 50)    # class 1 → strong (correct)

        proba_a = np.column_stack([1 - proba_a_pos, proba_a_pos])
        proba_b = np.column_stack([1 - proba_b_pos, proba_b_pos])

        models = {
            "model_a": _make_model(proba_a),
            "model_b": _make_model(proba_b),
        }

        # Use a low threshold so complementary gain is accepted
        low_threshold_config = ToolkitConfig(ensemble={"marginal_gain_threshold": 0.001})
        report = run_ensemble(models, X_val, binary_labels, low_threshold_config)

        assert isinstance(report, EnsembleReport)
        assert report.ensemble_selected is True
        assert report.strategy in ("soft_voting", "weighted_averaging")
        assert len(report.member_ids) == 2
        assert report.marginal_gain is not None
        assert report.marginal_gain >= 0.001

    def test_ensemble_score_higher_than_individuals(self, X_val, binary_labels):
        """Ensemble score should be >= best individual when accepted."""
        n = len(binary_labels)
        rng = np.random.RandomState(7)

        proba_a_pos = np.where(binary_labels == 0, rng.uniform(0.05, 0.3, n), rng.uniform(0.55, 0.8, n))
        proba_b_pos = np.where(binary_labels == 1, rng.uniform(0.65, 0.95, n), rng.uniform(0.15, 0.4, n))

        models = {
            "a": _make_model(np.column_stack([1 - proba_a_pos, proba_a_pos])),
            "b": _make_model(np.column_stack([1 - proba_b_pos, proba_b_pos])),
        }

        config = ToolkitConfig(ensemble={"marginal_gain_threshold": 0.001})
        report = run_ensemble(models, X_val, binary_labels, config)

        if report.ensemble_selected:
            best_individual = max(report.individual_scores.values())
            assert report.ensemble_score >= best_individual


# ---------------------------------------------------------------------------
# Test 2: Ensemble rejected on marginal gain case
# ---------------------------------------------------------------------------

class TestEnsembleRejected:

    def test_near_identical_models_rejected(self, X_val, binary_labels, config):
        """Two models with nearly identical predictions offer no gain."""
        n = len(binary_labels)
        rng = np.random.RandomState(42)

        proba_pos = np.where(binary_labels == 1, rng.uniform(0.6, 0.9, n), rng.uniform(0.1, 0.4, n))
        proba = np.column_stack([1 - proba_pos, proba_pos])

        # Identical models
        models = {
            "model_a": _make_model(proba),
            "model_b": _make_model(proba),
        }

        report = run_ensemble(models, X_val, binary_labels, config)

        assert report.ensemble_selected is False
        assert report.rejection_reason is not None
        assert "gain" in report.rejection_reason.lower() or "marginal" in report.rejection_reason.lower()

    def test_single_model_rejected(self, X_val, binary_labels, config):
        """Fewer than 2 probabilistic candidates → no ensemble."""
        n = len(binary_labels)
        rng = np.random.RandomState(42)
        proba_pos = rng.uniform(0.3, 0.7, n)
        proba = np.column_stack([1 - proba_pos, proba_pos])

        models = {"only_one": _make_model(proba)}
        report = run_ensemble(models, X_val, binary_labels, config)

        assert report.ensemble_selected is False
        assert "Fewer than 2" in report.rejection_reason

    def test_non_probabilistic_models_excluded(self, X_val, binary_labels, config):
        """Non-probabilistic models should not participate in ensemble."""
        n = len(binary_labels)
        rng = np.random.RandomState(42)
        proba_pos = rng.uniform(0.3, 0.7, n)
        proba = np.column_stack([1 - proba_pos, proba_pos])

        models = {
            "prob_model": _make_model(proba),
            "svm": _make_non_prob_model(),
        }

        report = run_ensemble(models, X_val, binary_labels, config)
        assert report.ensemble_selected is False
        assert "Fewer than 2" in report.rejection_reason

    def test_high_threshold_rejects(self, X_val, binary_labels):
        """Very high gain threshold should reject even decent ensembles."""
        n = len(binary_labels)
        rng = np.random.RandomState(42)

        proba_a_pos = np.where(binary_labels == 0, rng.uniform(0.05, 0.3, n), rng.uniform(0.55, 0.8, n))
        proba_b_pos = np.where(binary_labels == 1, rng.uniform(0.65, 0.95, n), rng.uniform(0.15, 0.4, n))

        models = {
            "a": _make_model(np.column_stack([1 - proba_a_pos, proba_a_pos])),
            "b": _make_model(np.column_stack([1 - proba_b_pos, proba_b_pos])),
        }

        # Unreasonably high threshold
        config = ToolkitConfig(ensemble={"marginal_gain_threshold": 0.5})
        report = run_ensemble(models, X_val, binary_labels, config)

        assert report.ensemble_selected is False
        assert report.rejection_reason is not None


# ---------------------------------------------------------------------------
# Test 3: Ensemble report serialization
# ---------------------------------------------------------------------------

class TestEnsembleReportSerialization:

    def test_report_serializes(self, X_val, binary_labels, config):
        n = len(binary_labels)
        rng = np.random.RandomState(42)
        proba_pos = np.where(binary_labels == 1, rng.uniform(0.6, 0.9, n), rng.uniform(0.1, 0.4, n))
        proba = np.column_stack([1 - proba_pos, proba_pos])

        models = {"a": _make_model(proba), "b": _make_model(proba)}
        report = run_ensemble(models, X_val, binary_labels, config)

        data = report.model_dump()
        assert isinstance(data, dict)
        assert "ensemble_selected" in data
        assert "rejection_reason" in data

        reloaded = EnsembleReport.model_validate(data)
        assert reloaded.ensemble_selected == report.ensemble_selected

    def test_accepted_report_has_all_fields(self, X_val, binary_labels):
        n = len(binary_labels)
        rng = np.random.RandomState(42)

        proba_a_pos = np.where(binary_labels == 0, rng.uniform(0.05, 0.3, n), rng.uniform(0.55, 0.8, n))
        proba_b_pos = np.where(binary_labels == 1, rng.uniform(0.65, 0.95, n), rng.uniform(0.15, 0.4, n))

        models = {
            "a": _make_model(np.column_stack([1 - proba_a_pos, proba_a_pos])),
            "b": _make_model(np.column_stack([1 - proba_b_pos, proba_b_pos])),
        }

        config = ToolkitConfig(ensemble={"marginal_gain_threshold": 0.001})
        report = run_ensemble(models, X_val, binary_labels, config)

        if report.ensemble_selected:
            assert report.strategy is not None
            assert len(report.member_ids) >= 2
            assert report.ensemble_score is not None
            assert report.marginal_gain is not None
            assert report.rejection_reason is None


# ---------------------------------------------------------------------------
# Test 4: Strategy unit tests
# ---------------------------------------------------------------------------

class TestSoftVotingStrategy:

    def test_combine_equal_weights(self):
        a = np.array([0.2, 0.8, 0.5])
        b = np.array([0.4, 0.6, 0.5])
        strategy = SoftVotingStrategy()
        combined = strategy.combine([a, b])
        np.testing.assert_allclose(combined, [0.3, 0.7, 0.5])

    def test_combine_with_weights(self):
        a = np.array([0.2, 0.8])
        b = np.array([0.4, 0.6])
        strategy = SoftVotingStrategy()
        combined = strategy.combine([a, b], weights=[3.0, 1.0])
        # (0.2*3 + 0.4*1)/4 = 1.0/4 = 0.25,  (0.8*3 + 0.6*1)/4 = 3.0/4 = 0.75
        np.testing.assert_allclose(combined, [0.25, 0.75])

    def test_evaluate_gain_true(self):
        config = ToolkitConfig(ensemble={"marginal_gain_threshold": 0.01})
        strategy = SoftVotingStrategy()
        assert strategy.evaluate_gain({"a": 0.7, "b": 0.65}, 0.72, config) is True

    def test_evaluate_gain_false(self):
        config = ToolkitConfig(ensemble={"marginal_gain_threshold": 0.01})
        strategy = SoftVotingStrategy()
        assert strategy.evaluate_gain({"a": 0.7, "b": 0.65}, 0.705, config) is False

    def test_strategy_name(self):
        assert SoftVotingStrategy().strategy_name() == "soft_voting"


class TestWeightedAveragingStrategy:

    def test_combine_weighted(self):
        a = np.array([0.2, 0.8])
        b = np.array([0.6, 0.4])
        strategy = WeightedAveragingStrategy()
        # weights [0.8, 0.6] → (0.2*0.8 + 0.6*0.6)/1.4, (0.8*0.8 + 0.4*0.6)/1.4
        combined = strategy.combine([a, b], weights=[0.8, 0.6])
        expected_0 = (0.2 * 0.8 + 0.6 * 0.6) / 1.4
        expected_1 = (0.8 * 0.8 + 0.4 * 0.6) / 1.4
        np.testing.assert_allclose(combined, [expected_0, expected_1], atol=1e-6)

    def test_strategy_name(self):
        assert WeightedAveragingStrategy().strategy_name() == "weighted_averaging"


# ---------------------------------------------------------------------------
# Test 5: Config controls
# ---------------------------------------------------------------------------

class TestEnsembleConfig:

    def test_default_strategies(self, config):
        assert "soft_voting" in config.ensemble.enabled_strategies
        assert "weighted_averaging" in config.ensemble.enabled_strategies

    def test_default_threshold(self, config):
        assert config.ensemble.marginal_gain_threshold == 0.01

    def test_max_ensemble_size(self, config):
        assert config.ensemble.max_ensemble_size == 3

    def test_custom_threshold_via_config(self):
        config = ToolkitConfig(ensemble={"marginal_gain_threshold": 0.05})
        assert config.ensemble.marginal_gain_threshold == 0.05
