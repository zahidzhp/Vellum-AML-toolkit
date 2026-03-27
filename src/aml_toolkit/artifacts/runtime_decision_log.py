"""Typed artifact for runtime decision engine output."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from aml_toolkit.core.enums import DecisionType


class RuntimeDecision(BaseModel):
    """A single runtime decision for a candidate model."""

    candidate_id: str
    epochs_seen: int
    decision: DecisionType
    reasons: list[str] = Field(default_factory=list)
    triggering_metrics: dict[str, float] = Field(default_factory=dict)
    warmup_gate_status: str = "pending"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class RuntimeDecisionLog(BaseModel):
    """Sequence of runtime decisions across all candidates."""

    decisions: list[RuntimeDecision] = Field(default_factory=list)
