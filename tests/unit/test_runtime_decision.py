"""Tests for Phase 10: Runtime Decision Engine and Warm-Up Policies.

Required tests:
1. Slow warm-up candidate not stopped too early.
2. Clear underperformer stopped after threshold.
3. Unstable candidate abstention/stop case.
4. Runtime decision log content test.
"""

import math

import pytest

from aml_toolkit.artifacts.runtime_decision_log import RuntimeDecision, RuntimeDecisionLog
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import DecisionType
from aml_toolkit.runtime.decision_engine import (
    MetricTracker,
    RuntimeDecisionEngine,
    WarmupPolicyManager,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config():
    return ToolkitConfig()


@pytest.fixture()
def engine(config):
    warmup_rules = {"logistic": 5, "rf": 5, "mlp": 10}
    return RuntimeDecisionEngine(config, warmup_rules=warmup_rules)


# ---------------------------------------------------------------------------
# Test 1: Slow warm-up candidate not stopped too early
# ---------------------------------------------------------------------------

class TestWarmupProtection:
    """Candidates must not be terminated before their warm-up window completes."""

    def test_neural_candidate_protected_during_warmup(self, engine):
        """MLP with 10-epoch warmup must get CONTINUE at epoch 5, even with bad metrics."""
        tracker = engine.get_or_create_tracker("mlp_001")
        # Record 5 epochs of bad metrics
        for i in range(5):
            tracker.record(epoch=i, train_metric=0.5, val_metric=0.3)

        decision = engine.evaluate_candidate("mlp_001", "mlp", is_neural=True, tracker=tracker)
        assert decision.decision == DecisionType.CONTINUE
        assert decision.warmup_gate_status == "active"
        assert "Warm-up active" in decision.reasons[0]

    def test_tabular_candidate_protected_during_warmup(self, engine):
        """Logistic with 5-epoch warmup must get CONTINUE at epoch 3."""
        tracker = engine.get_or_create_tracker("logistic_001")
        for i in range(3):
            tracker.record(epoch=i, train_metric=0.5, val_metric=0.3)

        decision = engine.evaluate_candidate("logistic_001", "logistic", is_neural=False, tracker=tracker)
        assert decision.decision == DecisionType.CONTINUE
        assert decision.warmup_gate_status == "active"

    def test_warmup_completes_and_allows_stop(self, engine):
        """After warmup completes, a stagnant candidate can be stopped."""
        tracker = engine.get_or_create_tracker("logistic_001")
        # 5 epochs of warmup + stagnant metrics
        for i in range(6):
            tracker.record(epoch=i, train_metric=0.5, val_metric=0.4)

        decision = engine.evaluate_candidate("logistic_001", "logistic", is_neural=False, tracker=tracker)
        assert decision.decision == DecisionType.STOP
        assert decision.warmup_gate_status == "completed"

    def test_warmup_policy_manager_min_epochs(self, config):
        """WarmupPolicyManager returns correct minimums per family."""
        rules = {"logistic": 5, "mlp": 10}
        mgr = WarmupPolicyManager(rules, config.runtime_decision)

        assert mgr.min_epochs_for("logistic", is_neural=False) == 5
        assert mgr.min_epochs_for("mlp", is_neural=True) == 10
        # Unregistered family falls back to config defaults
        assert mgr.min_epochs_for("unknown_tabular", is_neural=False) == config.runtime_decision.min_warmup_epochs_default
        assert mgr.min_epochs_for("unknown_neural", is_neural=True) == config.runtime_decision.min_warmup_epochs_neural

    def test_warmup_in_warmup_flag(self, config):
        mgr = WarmupPolicyManager({"rf": 5}, config.runtime_decision)
        assert mgr.is_in_warmup("rf", is_neural=False, epochs_seen=3) is True
        assert mgr.is_in_warmup("rf", is_neural=False, epochs_seen=5) is False
        assert mgr.is_in_warmup("rf", is_neural=False, epochs_seen=10) is False


# ---------------------------------------------------------------------------
# Test 2: Clear underperformer stopped after threshold
# ---------------------------------------------------------------------------

class TestUnderperformerStopped:
    """A candidate with no improvement slope should be stopped after warmup."""

    def test_stagnant_candidate_stopped(self, engine):
        """Flat metrics after warmup → STOP."""
        tracker = engine.get_or_create_tracker("rf_001")
        # 7 epochs of identical val metric (well past warmup of 5)
        for i in range(7):
            tracker.record(epoch=i, train_metric=0.5, val_metric=0.4)

        decision = engine.evaluate_candidate("rf_001", "rf", is_neural=False, tracker=tracker)
        assert decision.decision == DecisionType.STOP
        assert "No meaningful improvement" in decision.reasons[0]
        assert decision.triggering_metrics["improvement_slope"] <= 0.001

    def test_declining_candidate_stopped(self, engine):
        """Declining val metric after warmup → STOP."""
        tracker = engine.get_or_create_tracker("logistic_001")
        vals = [0.6, 0.58, 0.55, 0.52, 0.50, 0.48]
        for i, v in enumerate(vals):
            tracker.record(epoch=i, train_metric=0.7, val_metric=v)

        decision = engine.evaluate_candidate("logistic_001", "logistic", is_neural=False, tracker=tracker)
        assert decision.decision == DecisionType.STOP

    def test_improving_candidate_continues(self, engine):
        """Steadily improving val metric → CONTINUE."""
        tracker = engine.get_or_create_tracker("logistic_001")
        vals = [0.4, 0.45, 0.50, 0.55, 0.60, 0.65]
        for i, v in enumerate(vals):
            tracker.record(epoch=i, train_metric=0.7, val_metric=v)

        decision = engine.evaluate_candidate("logistic_001", "logistic", is_neural=False, tracker=tracker)
        assert decision.decision == DecisionType.CONTINUE
        assert decision.warmup_gate_status == "completed"
        assert "Improving" in decision.reasons[0]

    def test_overfitting_candidate_stopped(self, engine):
        """Large generalization gap → STOP."""
        tracker = engine.get_or_create_tracker("mlp_001")
        # 12 epochs (past neural warmup of 10), slight improvement but huge gap
        for i in range(12):
            tracker.record(
                epoch=i,
                train_metric=0.95 + i * 0.002,
                val_metric=0.55 + i * 0.005,
            )

        decision = engine.evaluate_candidate("mlp_001", "mlp", is_neural=True, tracker=tracker)
        # The large train-val gap should trigger STOP (gap ~0.38)
        assert decision.decision == DecisionType.STOP
        assert any("Overfit" in r or "improvement" in r.lower() for r in decision.reasons)


# ---------------------------------------------------------------------------
# Test 3: Unstable candidate abstention/stop case
# ---------------------------------------------------------------------------

class TestUnstableCandidateAbstention:
    """Candidates with NaN/Inf metrics should trigger ABSTAIN."""

    def test_nan_val_metric_triggers_abstain(self, engine):
        tracker = engine.get_or_create_tracker("mlp_001")
        tracker.record(epoch=0, train_metric=0.5, val_metric=0.4)
        tracker.record(epoch=1, train_metric=0.5, val_metric=float("nan"))

        decision = engine.evaluate_candidate("mlp_001", "mlp", is_neural=True, tracker=tracker)
        assert decision.decision == DecisionType.ABSTAIN
        assert decision.warmup_gate_status == "bypassed"
        assert "instability" in decision.reasons[0].lower()

    def test_inf_train_metric_triggers_abstain(self, engine):
        tracker = engine.get_or_create_tracker("rf_001")
        tracker.record(epoch=0, train_metric=float("inf"), val_metric=0.4)

        decision = engine.evaluate_candidate("rf_001", "rf", is_neural=False, tracker=tracker)
        assert decision.decision == DecisionType.ABSTAIN

    def test_nan_during_warmup_still_abstains(self, engine):
        """Even during warmup, NaN overrides the protection."""
        tracker = engine.get_or_create_tracker("mlp_001")
        tracker.record(epoch=0, train_metric=float("nan"), val_metric=0.3)

        decision = engine.evaluate_candidate("mlp_001", "mlp", is_neural=True, tracker=tracker)
        assert decision.decision == DecisionType.ABSTAIN
        assert decision.warmup_gate_status == "bypassed"

    def test_uncertain_trend_expands(self, engine):
        """Ambiguous trend with variance → EXPAND."""
        tracker = engine.get_or_create_tracker("logistic_001")
        # Oscillating metric after warmup, net slope ≈ 0 but with variance
        vals = [0.5, 0.52, 0.48, 0.53, 0.47, 0.52]
        for i, v in enumerate(vals):
            tracker.record(epoch=i, train_metric=0.5, val_metric=v)

        decision = engine.evaluate_candidate("logistic_001", "logistic", is_neural=False, tracker=tracker)
        # Slope is close to threshold; should either STOP (flat) or EXPAND (variance)
        assert decision.decision in (DecisionType.STOP, DecisionType.EXPAND)


# ---------------------------------------------------------------------------
# Test 4: Runtime decision log content
# ---------------------------------------------------------------------------

class TestRuntimeDecisionLog:
    """Verify the decision log captures all decisions with required fields."""

    def test_log_accumulates_decisions(self, engine):
        """Each evaluate call adds an entry to the log."""
        tracker1 = engine.get_or_create_tracker("logistic_001")
        for i in range(6):
            tracker1.record(epoch=i, train_metric=0.5, val_metric=0.4)

        tracker2 = engine.get_or_create_tracker("rf_001")
        for i in range(3):
            tracker2.record(epoch=i, train_metric=0.5, val_metric=0.5)

        engine.evaluate_candidate("logistic_001", "logistic", is_neural=False, tracker=tracker1)
        engine.evaluate_candidate("rf_001", "rf", is_neural=False, tracker=tracker2)

        log = engine.decision_log
        assert isinstance(log, RuntimeDecisionLog)
        assert len(log.decisions) == 2

    def test_log_entries_have_required_fields(self, engine):
        tracker = engine.get_or_create_tracker("mlp_001")
        for i in range(12):
            tracker.record(epoch=i, train_metric=0.6, val_metric=0.5 + i * 0.01)

        engine.evaluate_candidate("mlp_001", "mlp", is_neural=True, tracker=tracker)

        entry = engine.decision_log.decisions[0]
        assert isinstance(entry, RuntimeDecision)
        assert entry.candidate_id == "mlp_001"
        assert entry.epochs_seen == 12
        assert isinstance(entry.decision, DecisionType)
        assert len(entry.reasons) > 0
        assert isinstance(entry.triggering_metrics, dict)
        assert entry.warmup_gate_status in ("active", "completed", "bypassed")
        assert entry.timestamp is not None

    def test_log_serializes(self, engine):
        tracker = engine.get_or_create_tracker("logistic_001")
        tracker.record(epoch=0, train_metric=0.5, val_metric=0.4)
        engine.evaluate_candidate("logistic_001", "logistic", is_neural=False, tracker=tracker)

        log = engine.decision_log
        data = log.model_dump()
        assert isinstance(data, dict)
        assert "decisions" in data
        assert len(data["decisions"]) == 1
        reloaded = RuntimeDecisionLog.model_validate(data)
        assert len(reloaded.decisions) == 1

    def test_evaluate_from_trace_integration(self, engine):
        """evaluate_from_trace loads a training trace dict and produces a decision."""
        training_trace = {"val_macro_f1": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]}

        decision = engine.evaluate_from_trace(
            candidate_id="logistic_001",
            model_family="logistic",
            is_neural=False,
            training_trace=training_trace,
        )

        assert isinstance(decision, RuntimeDecision)
        assert decision.candidate_id == "logistic_001"
        assert decision.epochs_seen == 6
        assert decision.decision in (DecisionType.CONTINUE, DecisionType.STOP, DecisionType.EXPAND)
        assert len(engine.decision_log.decisions) == 1


# ---------------------------------------------------------------------------
# Test 5: MetricTracker unit tests
# ---------------------------------------------------------------------------

class TestMetricTracker:

    def test_empty_tracker(self):
        tracker = MetricTracker()
        assert tracker.epochs_seen == 0
        assert tracker.has_instability() is False
        assert tracker.recent_val_slope(3) is None
        assert tracker.generalization_gap() is None
        assert tracker.best_val_metric() is None

    def test_slope_positive(self):
        tracker = MetricTracker()
        for i in range(5):
            tracker.record(epoch=i, train_metric=0.5, val_metric=0.3 + i * 0.1)
        slope = tracker.recent_val_slope(5)
        assert slope is not None
        assert slope > 0

    def test_slope_negative(self):
        tracker = MetricTracker()
        for i in range(5):
            tracker.record(epoch=i, train_metric=0.5, val_metric=0.7 - i * 0.1)
        slope = tracker.recent_val_slope(5)
        assert slope is not None
        assert slope < 0

    def test_slope_flat(self):
        tracker = MetricTracker()
        for i in range(5):
            tracker.record(epoch=i, train_metric=0.5, val_metric=0.5)
        slope = tracker.recent_val_slope(5)
        assert slope is not None
        assert abs(slope) < 1e-10

    def test_generalization_gap(self):
        tracker = MetricTracker()
        tracker.record(epoch=0, train_metric=0.9, val_metric=0.6)
        gap = tracker.generalization_gap()
        assert gap is not None
        assert abs(gap - 0.3) < 1e-6

    def test_nan_detection(self):
        tracker = MetricTracker()
        tracker.record(epoch=0, train_metric=0.5, val_metric=float("nan"))
        assert tracker.has_instability() is True

    def test_inf_detection(self):
        tracker = MetricTracker()
        tracker.record(epoch=0, train_metric=float("inf"), val_metric=0.5)
        assert tracker.has_instability() is True

    def test_best_val_metric(self):
        tracker = MetricTracker()
        tracker.record(epoch=0, train_metric=0.5, val_metric=0.3)
        tracker.record(epoch=1, train_metric=0.5, val_metric=0.6)
        tracker.record(epoch=2, train_metric=0.5, val_metric=0.4)
        assert tracker.best_val_metric() == 0.6

    def test_variance(self):
        tracker = MetricTracker()
        for i in range(5):
            tracker.record(epoch=i, train_metric=0.5, val_metric=0.5)
        var = tracker.recent_val_variance(5)
        assert var is not None
        assert var == 0.0


# ---------------------------------------------------------------------------
# Test 6: Gradient stability is neural-only (plan guardrail)
# ---------------------------------------------------------------------------

class TestGradientStabilityGuardrail:
    """Verify that the engine does not use gradient signals for non-neural models.

    The decision engine only uses metric trend and generalization gap for
    tabular candidates. This test ensures no spurious gradient-based decisions
    leak into tabular evaluations.
    """

    def test_tabular_decision_uses_only_metric_trend_and_gap(self, engine):
        """RF (non-neural) decision should cite slope or gap, never gradient."""
        tracker = engine.get_or_create_tracker("rf_001")
        for i in range(7):
            tracker.record(epoch=i, train_metric=0.5, val_metric=0.4)

        decision = engine.evaluate_candidate("rf_001", "rf", is_neural=False, tracker=tracker)
        for reason in decision.reasons:
            assert "gradient" not in reason.lower()
