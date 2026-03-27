"""Typed artifact for probe engine output."""

from typing import Any

from pydantic import BaseModel, Field


class ProbeResult(BaseModel):
    """Result from a single probe run."""

    model_name: str
    intervention_branch: str = "none"
    train_metrics: dict[str, float] = Field(default_factory=dict)
    val_metrics: dict[str, float] = Field(default_factory=dict)
    fit_time_seconds: float = 0.0
    modality: str = ""
    notes: list[str] = Field(default_factory=list)


class ProbeResultSet(BaseModel):
    """Aggregated results from all probe runs."""

    baseline_results: list[ProbeResult] = Field(default_factory=list)
    shallow_results: list[ProbeResult] = Field(default_factory=list)
    intervention_branch_results: list[ProbeResult] = Field(default_factory=list)
    selected_metrics: list[str] = Field(default_factory=list)
    intervention_sensitivity_summary: dict[str, Any] = Field(default_factory=dict)
    shortlist_recommendation: list[str] = Field(default_factory=list)
