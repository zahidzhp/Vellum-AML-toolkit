"""Runtime decision engine: rule-based, config-driven training decisions with warm-up policies.

Implements the pseudo-code from the system design:
  1. ABSTAIN on resource failure or critical instability (NaN, crash).
  2. CONTINUE during warm-up (epoch < min_epoch_threshold).
  3. After warm-up:
     - STOP if underperforming and stable (no improvement slope).
     - CONTINUE if improving.
     - EXPAND if uncertain (ambiguous trend, high variance).

Gradient stability is only a valid signal for neural models.
For tabular candidates (XGBoost, RF, logistic), use metric trend and
generalization gap only.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from aml_toolkit.artifacts.runtime_decision_log import RuntimeDecision, RuntimeDecisionLog
from aml_toolkit.core.config import RuntimeDecisionConfig, ToolkitConfig
from aml_toolkit.core.enums import DecisionType

logger = logging.getLogger("aml_toolkit")


# ---------------------------------------------------------------------------
# Metric Tracker
# ---------------------------------------------------------------------------

@dataclass
class MetricSnapshot:
    """Per-epoch metric snapshot for a candidate."""

    epoch: int
    train_metric: float | None = None
    val_metric: float | None = None


class MetricTracker:
    """Tracks per-epoch metrics for a single candidate and computes derived signals.

    Provides:
    - recent improvement slope (linear regression over last `patience` epochs)
    - generalization gap (val - train metric, where a large gap signals overfitting)
    - instability detection (NaN / Inf in metric history)
    - variance of recent metric values
    """

    def __init__(self) -> None:
        self._history: list[MetricSnapshot] = []

    @property
    def epochs_seen(self) -> int:
        return len(self._history)

    def record(self, epoch: int, train_metric: float | None, val_metric: float | None) -> None:
        self._history.append(MetricSnapshot(epoch=epoch, train_metric=train_metric, val_metric=val_metric))

    def has_instability(self) -> bool:
        """Check if any recorded metric is NaN or Inf."""
        for snap in self._history:
            for val in (snap.train_metric, snap.val_metric):
                if val is not None and (math.isnan(val) or math.isinf(val)):
                    return True
        return False

    def recent_val_slope(self, window: int) -> float | None:
        """Compute the linear-regression slope of validation metric over the last `window` epochs.

        Positive slope means improvement (higher metric = better).
        Returns None if insufficient data.
        """
        recent = [s.val_metric for s in self._history[-window:] if s.val_metric is not None]
        if len(recent) < 2:
            return None
        x = np.arange(len(recent), dtype=float)
        y = np.array(recent, dtype=float)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return None
        # Simple OLS slope
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    def recent_val_variance(self, window: int) -> float | None:
        """Variance of validation metric over the last `window` epochs."""
        recent = [s.val_metric for s in self._history[-window:] if s.val_metric is not None]
        if len(recent) < 2:
            return None
        arr = np.array(recent, dtype=float)
        if np.any(np.isnan(arr)):
            return None
        return float(np.var(arr))

    def generalization_gap(self) -> float | None:
        """Gap between the latest train and val metrics.

        Returns (train_metric - val_metric). A positive value means
        train > val, which may signal overfitting.
        """
        if not self._history:
            return None
        latest = self._history[-1]
        if latest.train_metric is None or latest.val_metric is None:
            return None
        return latest.train_metric - latest.val_metric

    def best_val_metric(self) -> float | None:
        vals = [s.val_metric for s in self._history if s.val_metric is not None]
        if not vals:
            return None
        return max(vals)

    def latest_val_metric(self) -> float | None:
        for snap in reversed(self._history):
            if snap.val_metric is not None:
                return snap.val_metric
        return None


# ---------------------------------------------------------------------------
# Warm-Up Policy Manager
# ---------------------------------------------------------------------------

class WarmupPolicyManager:
    """Enforces per-candidate warm-up periods.

    No candidate may be terminated before its architecture-specific
    minimum epoch threshold, except on hard failure (NaN, crash, OOM).
    """

    def __init__(self, warmup_rules: dict[str, int], config: RuntimeDecisionConfig) -> None:
        self._warmup_rules = warmup_rules
        self._config = config

    def min_epochs_for(self, model_family: str, is_neural: bool) -> int:
        """Return the minimum warm-up epochs for a given model family."""
        if model_family in self._warmup_rules:
            return self._warmup_rules[model_family]
        return self._config.min_warmup_epochs_neural if is_neural else self._config.min_warmup_epochs_default

    def is_in_warmup(self, model_family: str, is_neural: bool, epochs_seen: int) -> bool:
        """Return True if the candidate is still within its warm-up window."""
        return epochs_seen < self.min_epochs_for(model_family, is_neural)


# ---------------------------------------------------------------------------
# Runtime Decision Engine
# ---------------------------------------------------------------------------

class RuntimeDecisionEngine:
    """Rule-based decision engine that evaluates candidate training progress.

    Consumes metric traces from the training executor and produces
    RuntimeDecision entries. All logic is config-driven and transparent.
    """

    def __init__(self, config: ToolkitConfig, warmup_rules: dict[str, int] | None = None) -> None:
        self._config = config
        self._rd_config = config.runtime_decision
        self._warmup_mgr = WarmupPolicyManager(warmup_rules or {}, self._rd_config)
        self._trackers: dict[str, MetricTracker] = {}
        self._log = RuntimeDecisionLog()

    @property
    def decision_log(self) -> RuntimeDecisionLog:
        return self._log

    def get_or_create_tracker(self, candidate_id: str) -> MetricTracker:
        if candidate_id not in self._trackers:
            self._trackers[candidate_id] = MetricTracker()
        return self._trackers[candidate_id]

    def evaluate_candidate(
        self,
        candidate_id: str,
        model_family: str,
        is_neural: bool,
        tracker: MetricTracker,
    ) -> RuntimeDecision:
        """Evaluate a single candidate and produce a decision.

        Decision rules (in priority order):
        1. ABSTAIN if critical instability (NaN/Inf).
        2. CONTINUE if in warm-up.
        3. STOP if stagnant + no improvement slope for `patience` epochs.
        4. STOP if overfitting beyond overfit_gap_limit.
        5. CONTINUE if improving (positive slope above threshold).
        6. EXPAND if uncertain (ambiguous slope, high variance).
        7. Default CONTINUE.
        """
        epochs_seen = tracker.epochs_seen
        reasons: list[str] = []
        triggering_metrics: dict[str, float] = {}

        # --- Rule 1: Critical instability ---
        if tracker.has_instability():
            reasons.append("Critical metric instability detected (NaN/Inf).")
            decision = self._make_decision(
                candidate_id, epochs_seen, DecisionType.ABSTAIN,
                reasons, triggering_metrics, warmup_status="bypassed",
            )
            return decision

        # --- Rule 2: Warm-up gate ---
        in_warmup = self._warmup_mgr.is_in_warmup(model_family, is_neural, epochs_seen)
        warmup_status = "active" if in_warmup else "completed"

        if in_warmup:
            min_epochs = self._warmup_mgr.min_epochs_for(model_family, is_neural)
            reasons.append(
                f"Warm-up active: {epochs_seen}/{min_epochs} epochs. "
                "No early termination allowed."
            )
            decision = self._make_decision(
                candidate_id, epochs_seen, DecisionType.CONTINUE,
                reasons, triggering_metrics, warmup_status=warmup_status,
            )
            return decision

        # --- Post warm-up analysis ---
        patience = self._rd_config.patience
        slope = tracker.recent_val_slope(patience)
        gap = tracker.generalization_gap()
        variance = tracker.recent_val_variance(patience)
        latest_val = tracker.latest_val_metric()

        if slope is not None:
            triggering_metrics["improvement_slope"] = slope
        if gap is not None:
            triggering_metrics["generalization_gap"] = gap
        if variance is not None:
            triggering_metrics["recent_variance"] = variance
        if latest_val is not None:
            triggering_metrics["latest_val_metric"] = latest_val

        # --- Rule 3: Stagnant / no improvement ---
        if slope is not None and slope <= self._rd_config.improvement_slope_threshold:
            # For neural models, also consider gradient stability as an additional signal
            # (but we don't stop based on gradient alone for non-neural)
            reasons.append(
                f"No meaningful improvement: slope={slope:.6f} <= "
                f"threshold={self._rd_config.improvement_slope_threshold} "
                f"over last {patience} epochs."
            )
            decision = self._make_decision(
                candidate_id, epochs_seen, DecisionType.STOP,
                reasons, triggering_metrics, warmup_status=warmup_status,
            )
            return decision

        # --- Rule 4: Overfitting ---
        if gap is not None and gap > self._rd_config.overfit_gap_limit:
            reasons.append(
                f"Overfitting detected: generalization gap={gap:.4f} > "
                f"limit={self._rd_config.overfit_gap_limit}."
            )
            decision = self._make_decision(
                candidate_id, epochs_seen, DecisionType.STOP,
                reasons, triggering_metrics, warmup_status=warmup_status,
            )
            return decision

        # --- Rule 5: Improving ---
        if slope is not None and slope > self._rd_config.improvement_slope_threshold:
            reasons.append(
                f"Improving: slope={slope:.6f} > "
                f"threshold={self._rd_config.improvement_slope_threshold}."
            )
            decision = self._make_decision(
                candidate_id, epochs_seen, DecisionType.CONTINUE,
                reasons, triggering_metrics, warmup_status=warmup_status,
            )
            return decision

        # --- Rule 6: Uncertain / high variance ---
        if variance is not None and variance > 0:
            reasons.append(
                f"Uncertain trend: slope ambiguous, variance={variance:.6f}. "
                "Expanding candidate pool."
            )
            decision = self._make_decision(
                candidate_id, epochs_seen, DecisionType.EXPAND,
                reasons, triggering_metrics, warmup_status=warmup_status,
            )
            return decision

        # --- Rule 7: Default ---
        reasons.append("Insufficient signal for decision; continuing by default.")
        decision = self._make_decision(
            candidate_id, epochs_seen, DecisionType.CONTINUE,
            reasons, triggering_metrics, warmup_status=warmup_status,
        )
        return decision

    def evaluate_from_trace(
        self,
        candidate_id: str,
        model_family: str,
        is_neural: bool,
        training_trace: dict[str, list[float]],
        metrics: dict[str, float] | None = None,
    ) -> RuntimeDecision:
        """Convenience method: load a training trace into a tracker and evaluate.

        This is the primary integration point with Phase 9's ExecutionResult.
        """
        tracker = self.get_or_create_tracker(candidate_id)

        # Populate tracker from training trace
        val_key = _find_val_metric_key(training_trace)
        train_key = _find_train_metric_key(training_trace)

        val_values = training_trace.get(val_key, []) if val_key else []
        train_values = training_trace.get(train_key, []) if train_key else []

        max_len = max(len(val_values), len(train_values), 1)
        for i in range(max_len):
            train_val = train_values[i] if i < len(train_values) else None
            val_val = val_values[i] if i < len(val_values) else None
            tracker.record(epoch=i, train_metric=train_val, val_metric=val_val)

        return self.evaluate_candidate(candidate_id, model_family, is_neural, tracker)

    def _make_decision(
        self,
        candidate_id: str,
        epochs_seen: int,
        decision: DecisionType,
        reasons: list[str],
        triggering_metrics: dict[str, float],
        warmup_status: str,
    ) -> RuntimeDecision:
        entry = RuntimeDecision(
            candidate_id=candidate_id,
            epochs_seen=epochs_seen,
            decision=decision,
            reasons=reasons,
            triggering_metrics=triggering_metrics,
            warmup_gate_status=warmup_status,
        )
        self._log.decisions.append(entry)
        logger.info(
            f"Runtime decision for {candidate_id}: {decision.value} "
            f"(warmup={warmup_status}, epochs={epochs_seen}). "
            f"Reasons: {'; '.join(reasons)}"
        )
        return entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_val_metric_key(trace: dict[str, list[float]]) -> str | None:
    """Find the validation metric key in a training trace dict."""
    for key in trace:
        if "val" in key.lower():
            return key
    return None


def _find_train_metric_key(trace: dict[str, list[float]]) -> str | None:
    """Find the training metric key in a training trace dict."""
    for key in trace:
        if "train" in key.lower():
            return key
    return None
