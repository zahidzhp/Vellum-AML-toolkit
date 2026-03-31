"""Meta-policy recommendation artifact."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MetaPolicyRecommendation(BaseModel):
    """Recommendation from the meta-policy engine for candidate ordering and budgeting."""

    original_order: list[str] = Field(default_factory=list)
    recommended_order: list[str] = Field(default_factory=list)
    skipped_candidates: list[str] = Field(default_factory=list)
    rationale: dict[str, str] = Field(default_factory=dict)
    history_records_used: int = 0

    # Compute budget allocation: fractions per candidate (sum ≈ 1.0)
    compute_budget_fractions: dict[str, float] = Field(default_factory=dict)
    # e.g. {"rf_001": 0.4, "xgb_001": 0.4, "logistic_001": 0.2}

    notes: list[str] = Field(default_factory=list)
