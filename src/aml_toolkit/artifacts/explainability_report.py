"""Typed artifact for explainability output."""

from typing import Any

from pydantic import BaseModel, Field


class ExplainabilityOutput(BaseModel):
    """Output from a single explainability method."""

    method: str
    candidate_id: str
    artifact_paths: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    faithfulness_score: float | None = None
    supported: bool = True
    fallback_reason: str | None = None


class ExplainabilityReport(BaseModel):
    """Aggregated explainability outputs with caveats."""

    outputs: list[ExplainabilityOutput] = Field(default_factory=list)
    methods_attempted: list[str] = Field(default_factory=list)
    methods_succeeded: list[str] = Field(default_factory=list)
    methods_failed: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
