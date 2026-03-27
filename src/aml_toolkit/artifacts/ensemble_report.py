"""Typed artifact for ensemble building output."""

from pydantic import BaseModel, Field


class EnsembleReport(BaseModel):
    """Report on ensemble selection decision."""

    ensemble_selected: bool = False
    strategy: str | None = None
    member_ids: list[str] = Field(default_factory=list)
    individual_scores: dict[str, float] = Field(default_factory=dict)
    ensemble_score: float | None = None
    marginal_gain: float | None = None
    gain_threshold: float = 0.01
    rejection_reason: str | None = None
    notes: list[str] = Field(default_factory=list)
