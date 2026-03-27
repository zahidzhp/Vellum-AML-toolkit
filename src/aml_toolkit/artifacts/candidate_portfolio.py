"""Typed artifact for candidate model selection output."""

from typing import Any

from pydantic import BaseModel, Field


class CandidateEntry(BaseModel):
    """A single candidate model with metadata."""

    candidate_id: str
    model_family: str
    model_name: str
    warmup_epochs: int = 5
    budget_allocation: float = 1.0
    rejection_reason: str | None = None


class CandidatePortfolio(BaseModel):
    """Portfolio of candidate models selected for training."""

    candidate_models: list[CandidateEntry] = Field(default_factory=list)
    selected_families: list[str] = Field(default_factory=list)
    budget_allocations: dict[str, float] = Field(default_factory=dict)
    warmup_rules: dict[str, int] = Field(default_factory=dict)
    rejection_reasons: dict[str, str] = Field(default_factory=dict)
