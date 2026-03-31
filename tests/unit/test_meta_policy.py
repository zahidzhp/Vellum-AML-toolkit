"""Tests for Phase 2.5 — Meta-policy engine."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from aml_toolkit.artifacts.meta_policy_recommendation import MetaPolicyRecommendation
from aml_toolkit.artifacts.run_history import DatasetSignature, RunHistoryRecord
from aml_toolkit.core.config import MetaPolicyConfig
from aml_toolkit.meta_policy.meta_policy_engine import MetaPolicyEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> MetaPolicyConfig:
    defaults = dict(
        enabled=True,
        exploration_weight=0.3,
        allow_skip_low_value=True,
        allow_reorder_only=True,
        never_override_user_constraints=True,
        compute_budget_aware=True,
        similarity_method="cosine",
        recency_decay=0.9,
    )
    defaults.update(kwargs)
    return MetaPolicyConfig(**defaults)


def _make_sig(**kwargs) -> DatasetSignature:
    defaults = dict(
        modality="TABULAR",
        task_type="BINARY",
        n_classes=2,
        log_n_samples=4.0,
        log_n_features=2.0,
        imbalance_ratio=2.0,
        missingness_pct=0.05,
        duplicate_pct=0.01,
        ood_shift_score=0.0,
        label_noise_score=0.0,
    )
    defaults.update(kwargs)
    return DatasetSignature(**defaults)


def _make_record(
    sig: DatasetSignature | None = None,
    best_family: str = "rf",
    ts: datetime | None = None,
    **kwargs,
) -> RunHistoryRecord:
    if sig is None:
        sig = _make_sig()
    if ts is None:
        ts = datetime.now(timezone.utc)
    return RunHistoryRecord(
        run_id=f"run_{best_family}",
        timestamp=ts,
        dataset_signature=sig,
        best_candidate_id=f"{best_family}_001",
        best_candidate_family=best_family,
        best_macro_f1=0.85,
        config_mode="BALANCED",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# MetaPolicyRecommendation
# ---------------------------------------------------------------------------

class TestMetaPolicyRecommendation:
    def test_default_fields(self):
        rec = MetaPolicyRecommendation()
        assert rec.original_order == []
        assert rec.recommended_order == []
        assert rec.compute_budget_fractions == {}
        assert rec.history_records_used == 0

    def test_budget_fractions_field(self):
        rec = MetaPolicyRecommendation(
            original_order=["a", "b"],
            recommended_order=["b", "a"],
            compute_budget_fractions={"a": 0.4, "b": 0.6},
        )
        assert rec.compute_budget_fractions["b"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# MetaPolicyEngine — no history (equal budget)
# ---------------------------------------------------------------------------

class TestMetaPolicyEngineNoHistory:
    def test_empty_history_returns_equal_budget(self):
        engine = MetaPolicyEngine(_make_config())
        sig = _make_sig()
        candidates = ["rf_001", "xgb_001", "logistic_001"]
        rec = engine.recommend(candidates, sig, [])
        fracs = rec.compute_budget_fractions
        assert set(fracs.keys()) == set(candidates)
        total = sum(fracs.values())
        assert total == pytest.approx(1.0, abs=1e-6)
        # All fractions should be equal
        for frac in fracs.values():
            assert frac == pytest.approx(1.0 / 3, abs=0.01)

    def test_empty_candidates_returns_empty(self):
        engine = MetaPolicyEngine(_make_config())
        rec = engine.recommend([], _make_sig(), [])
        assert rec.original_order == []
        assert rec.recommended_order == []

    def test_single_candidate_equal_budget(self):
        engine = MetaPolicyEngine(_make_config())
        rec = engine.recommend(["rf_001"], _make_sig(), [])
        assert rec.compute_budget_fractions.get("rf_001") == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# MetaPolicyEngine — with history (cosine scoring)
# ---------------------------------------------------------------------------

class TestMetaPolicyEngineWithHistory:
    def test_output_is_subset_of_input(self):
        engine = MetaPolicyEngine(_make_config())
        sig = _make_sig()
        candidates = ["rf_001", "xgb_001", "logistic_001"]
        history = [_make_record(sig=sig, best_family="xgb")]
        rec = engine.recommend(candidates, sig, history)
        assert set(rec.recommended_order).issubset(set(candidates))
        assert len(rec.recommended_order) == len(candidates)

    def test_budget_fractions_sum_to_one(self):
        engine = MetaPolicyEngine(_make_config())
        sig = _make_sig()
        candidates = ["rf_001", "xgb_001", "logistic_001"]
        history = [_make_record(sig=sig, best_family="rf")]
        rec = engine.recommend(candidates, sig, history)
        total = sum(rec.compute_budget_fractions.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_promising_candidate_gets_more_budget(self):
        """Family that won on very similar datasets should get higher budget fraction."""
        engine = MetaPolicyEngine(_make_config(exploration_weight=0.1))
        sig = _make_sig(log_n_samples=4.0, imbalance_ratio=2.0)
        # Build history: rf won on very similar datasets
        history = [_make_record(sig=_make_sig(log_n_samples=4.0, imbalance_ratio=2.0), best_family="rf")
                   for _ in range(5)]
        candidates = ["rf_001", "xgb_001"]
        rec = engine.recommend(candidates, sig, history)
        # rf should get more budget than xgb
        rf_frac = rec.compute_budget_fractions.get("rf_001", 0.0)
        xgb_frac = rec.compute_budget_fractions.get("xgb_001", 0.0)
        assert rf_frac > xgb_frac, f"rf={rf_frac:.3f}, xgb={xgb_frac:.3f}"

    def test_cross_modality_records_excluded(self):
        """Image records should not influence tabular candidate scoring."""
        engine = MetaPolicyEngine(_make_config(exploration_weight=0.0))
        tabular_sig = _make_sig(modality="TABULAR")
        image_record = _make_record(sig=_make_sig(modality="IMAGE"), best_family="cnn")
        candidates = ["rf_001", "cnn_001"]
        rec = engine.recommend(candidates, tabular_sig, [image_record])
        # With no relevant history, should fall back to equal budget
        total = sum(rec.compute_budget_fractions.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_history_records_used_populated(self):
        engine = MetaPolicyEngine(_make_config())
        sig = _make_sig()
        history = [_make_record() for _ in range(7)]
        rec = engine.recommend(["rf_001", "xgb_001"], sig, history)
        assert rec.history_records_used == 7

    def test_recency_decay_recent_gets_more_weight(self):
        """Recent history records for a family should give higher score than old ones."""
        engine = MetaPolicyEngine(_make_config(exploration_weight=0.0, recency_decay=0.5))
        sig = _make_sig()
        old_ts = datetime.now(timezone.utc) - timedelta(days=100)
        recent_ts = datetime.now(timezone.utc) - timedelta(minutes=1)

        # Old record: xgb won
        old_record = RunHistoryRecord(
            run_id="old",
            timestamp=old_ts,
            dataset_signature=sig,
            dataset_signature_vector=sig.to_vector().tolist(),
            best_candidate_id="xgb_001",
            best_candidate_family="xgb",
            best_macro_f1=0.85,
            config_mode="BALANCED",
        )
        # Recent record: rf won
        recent_record = RunHistoryRecord(
            run_id="recent",
            timestamp=recent_ts,
            dataset_signature=sig,
            dataset_signature_vector=sig.to_vector().tolist(),
            best_candidate_id="rf_001",
            best_candidate_family="rf",
            best_macro_f1=0.85,
            config_mode="BALANCED",
        )
        candidates = ["rf_001", "xgb_001"]
        rec = engine.recommend(candidates, sig, [old_record, recent_record])
        rf_frac = rec.compute_budget_fractions.get("rf_001", 0.5)
        xgb_frac = rec.compute_budget_fractions.get("xgb_001", 0.5)
        # rf should get more budget due to recent win
        assert rf_frac >= xgb_frac, f"rf={rf_frac:.3f}, xgb={xgb_frac:.3f}"

    def test_never_adds_outside_user_candidates(self):
        """Recommended order must be a permutation of input candidates only."""
        engine = MetaPolicyEngine(_make_config())
        sig = _make_sig()
        candidates = ["rf_001", "logistic_001"]
        history = [_make_record(best_family="xgb")]  # xgb not in candidates
        rec = engine.recommend(candidates, sig, history)
        for cid in rec.recommended_order:
            assert cid in candidates

    def test_graceful_failure_returns_equal_budget(self):
        """If internal scoring fails, fall back to equal budget."""
        engine = MetaPolicyEngine(_make_config())
        # Pass garbage history
        candidates = ["rf_001", "xgb_001"]
        rec = engine.recommend(candidates, _make_sig(), [None])  # type: ignore
        total = sum(rec.compute_budget_fractions.values())
        assert total == pytest.approx(1.0, abs=1e-6)
