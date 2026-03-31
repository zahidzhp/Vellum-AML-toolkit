"""Tests for Phase 2.4 — Greedy diverse ensemble and diversity metrics."""

import numpy as np
import pytest

from aml_toolkit.core.config import DynamicEnsembleConfig
from aml_toolkit.ensemble.diversity_metrics import (
    ambiguity_decomposition,
    ensemble_diversity_score,
    pairwise_disagreement,
)
from aml_toolkit.ensemble.greedy_diverse import GreedyDiverseEnsemble


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> DynamicEnsembleConfig:
    defaults = dict(
        enabled=True,
        max_members=4,
        diversity_threshold=0.05,
        use_uncertainty_weights=False,
    )
    defaults.update(kwargs)
    return DynamicEnsembleConfig(**defaults)


def _identical_models(n: int, n_classes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Two identical probability arrays."""
    rng = np.random.default_rng(0)
    p = rng.dirichlet([3, 1, 1] if n_classes == 3 else [3, 1], size=n)
    return p, p.copy()


def _complementary_models(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model A is right on half, Model B is right on the other half, Model C agrees with A."""
    y = np.array([0, 1] * (n // 2))
    # Model A: confident on first half (class 0)
    p_a = np.zeros((n, 2))
    p_a[:, 0] = 0.9
    p_a[:, 1] = 0.1

    # Model B: confident on second half (class 1) — complementary to A
    p_b = np.zeros((n, 2))
    p_b[:, 0] = 0.1
    p_b[:, 1] = 0.9

    # Model C: same as A (clone)
    p_c = p_a.copy()

    return p_a, p_b, p_c


# ---------------------------------------------------------------------------
# pairwise_disagreement
# ---------------------------------------------------------------------------

class TestPairwiseDisagreement:
    def test_identical_models_zero_disagreement(self):
        preds = np.array([0, 1, 2, 0, 1])
        assert pairwise_disagreement(preds, preds.copy()) == pytest.approx(0.0)

    def test_opposite_models_full_disagreement(self):
        preds_a = np.array([0, 0, 0, 0])
        preds_b = np.array([1, 1, 1, 1])
        assert pairwise_disagreement(preds_a, preds_b) == pytest.approx(1.0)

    def test_half_disagreement(self):
        preds_a = np.array([0, 0, 1, 1])
        preds_b = np.array([0, 1, 0, 1])
        assert pairwise_disagreement(preds_a, preds_b) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ensemble_diversity_score
# ---------------------------------------------------------------------------

class TestEnsembleDiversityScore:
    def test_single_model_returns_zero(self):
        preds = [np.array([0, 1, 0, 1])]
        assert ensemble_diversity_score(preds) == pytest.approx(0.0)

    def test_identical_models_zero_diversity(self):
        preds = np.array([0, 1, 2, 0])
        assert ensemble_diversity_score([preds, preds.copy(), preds.copy()]) == pytest.approx(0.0)

    def test_diverse_models_positive_score(self):
        preds_a = np.array([0, 0, 0, 0])
        preds_b = np.array([1, 1, 1, 1])
        assert ensemble_diversity_score([preds_a, preds_b]) == pytest.approx(1.0)

    def test_empty_returns_zero(self):
        assert ensemble_diversity_score([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ambiguity_decomposition
# ---------------------------------------------------------------------------

class TestAmbiguityDecomposition:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(0)
        probas = [rng.dirichlet([2, 1], size=100) for _ in range(3)]
        y = rng.integers(0, 2, size=100)
        result = ambiguity_decomposition(probas, y)
        assert "bias" in result
        assert "variance" in result
        assert "diversity" in result
        assert "error" in result

    def test_error_nonnegative(self):
        rng = np.random.default_rng(1)
        probas = [rng.dirichlet([1, 1, 1], size=100) for _ in range(3)]
        y = rng.integers(0, 3, size=100)
        result = ambiguity_decomposition(probas, y)
        assert result["error"] >= 0.0

    def test_diversity_nonnegative(self):
        rng = np.random.default_rng(2)
        probas = [rng.dirichlet([1, 1], size=100) for _ in range(4)]
        y = rng.integers(0, 2, size=100)
        result = ambiguity_decomposition(probas, y)
        assert result["diversity"] >= 0.0

    def test_empty_probas_returns_zeros(self):
        result = ambiguity_decomposition([], np.array([0, 1]))
        assert result["error"] == 0.0

    def test_identical_models_zero_diversity(self):
        rng = np.random.default_rng(3)
        p = rng.dirichlet([3, 1], size=100)
        y = rng.integers(0, 2, size=100)
        result = ambiguity_decomposition([p, p.copy(), p.copy()], y)
        # All models identical → diversity contribution should be 0
        assert result["diversity"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# GreedyDiverseEnsemble.select
# ---------------------------------------------------------------------------

class TestGreedyDiverseEnsemble:
    def test_single_candidate_returns_single_member(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=100)
        p = rng.dirichlet([3, 1], size=100)
        config = _make_config()
        ensemble = GreedyDiverseEnsemble(config)
        report = ensemble.select({"m1": p}, y)
        assert "m1" in report.member_ids

    def test_selects_complementary_over_clone(self):
        """Model B (complements A) should be preferred over Model C (clone of A)."""
        n = 100
        y = np.array([0, 1] * (n // 2))
        p_a, p_b, p_c = _complementary_models(n)

        config = _make_config(max_members=3, diversity_threshold=0.05)
        ensemble = GreedyDiverseEnsemble(config)

        # Precompute individual scores to ensure A starts
        from sklearn.metrics import f1_score
        scores = {
            "a": f1_score(y, np.argmax(p_a, axis=1), average="macro", zero_division=0),
            "b": f1_score(y, np.argmax(p_b, axis=1), average="macro", zero_division=0),
            "c": f1_score(y, np.argmax(p_c, axis=1), average="macro", zero_division=0),
        }
        report = ensemble.select({"a": p_a, "b": p_b, "c": p_c}, y, candidate_scores=scores)
        # B should be selected (complements A), not just C (clone of A)
        assert "b" in report.member_ids

    def test_clone_not_added_below_threshold(self):
        """Two identical models → diversity below threshold → only one selected."""
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=100)
        p_a, p_b = _identical_models(100, n_classes=2)

        config = _make_config(max_members=3, diversity_threshold=0.5)  # high threshold
        ensemble = GreedyDiverseEnsemble(config)
        report = ensemble.select({"a": p_a, "b": p_b}, y)
        # Should not add the clone
        assert "b" not in report.member_ids or "a" not in report.member_ids

    def test_max_members_enforced(self):
        """Never exceed max_members."""
        rng = np.random.default_rng(5)
        y = rng.integers(0, 3, size=200)
        probas = {f"m{i}": rng.dirichlet([2, 1, 1], size=200) for i in range(10)}

        config = _make_config(max_members=2)
        ensemble = GreedyDiverseEnsemble(config)
        report = ensemble.select(probas, y)
        assert len(report.member_ids) <= 2

    def test_diversity_score_in_report(self):
        rng = np.random.default_rng(9)
        y = rng.integers(0, 2, size=100)
        p_a = rng.dirichlet([3, 1], size=100)
        p_b = rng.dirichlet([1, 3], size=100)
        config = _make_config(max_members=3, diversity_threshold=0.0)
        ensemble = GreedyDiverseEnsemble(config)
        report = ensemble.select({"a": p_a, "b": p_b}, y)
        assert "diversity_score" in report.model_dump()
        assert report.diversity_score >= 0.0

    def test_ambiguity_decomposition_in_report(self):
        rng = np.random.default_rng(10)
        y = rng.integers(0, 2, size=100)
        probas = {f"m{i}": rng.dirichlet([2, 1], size=100) for i in range(3)}
        config = _make_config(max_members=3, diversity_threshold=0.0)
        ensemble = GreedyDiverseEnsemble(config)
        report = ensemble.select(probas, y)
        assert isinstance(report.ambiguity_decomposition, dict)

    def test_empty_probas_returns_report(self):
        config = _make_config()
        ensemble = GreedyDiverseEnsemble(config)
        report = ensemble.select({}, np.array([0, 1]))
        assert report.rejection_reason is not None

    def test_binary_1d_proba_works(self):
        """(n,) binary proba should be accepted without error."""
        rng = np.random.default_rng(11)
        y = rng.integers(0, 2, size=100)
        p1 = rng.uniform(0.3, 0.7, size=100)
        p2 = rng.uniform(0.3, 0.7, size=100)
        config = _make_config(max_members=3, diversity_threshold=0.0)
        ensemble = GreedyDiverseEnsemble(config)
        report = ensemble.select({"a": p1, "b": p2}, y)
        assert report is not None
