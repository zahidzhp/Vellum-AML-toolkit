"""Uncertainty report artifact."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UncertaintyReport(BaseModel):
    """Per-candidate uncertainty estimation results."""

    candidate_id: str
    methods_used: list[str] = Field(default_factory=list)
    mean_uncertainty: float = 0.0
    pct_high_uncertainty: float = 0.0  # % samples above abstention threshold
    abstention_triggered: bool = False
    abstention_reason: str = ""
    sample_count: int = 0

    # Conformal prediction fields
    conformal_coverage_achieved: float | None = None  # empirical coverage on val set
    mean_prediction_set_size: float | None = None  # efficiency metric (lower = more confident)
    pct_singleton_sets: float | None = None  # % samples with a unique prediction

    # Per-method scores (optional detail)
    entropy_mean: float | None = None
    margin_mean: float | None = None
