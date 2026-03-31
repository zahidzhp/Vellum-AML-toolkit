"""Typed artifact for the final pipeline report."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from aml_toolkit.core.enums import AbstentionReason, PipelineStage


class FinalReport(BaseModel):
    """Complete pipeline report summarizing all stages."""

    run_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    final_status: PipelineStage = PipelineStage.COMPLETED
    abstention_reason: AbstentionReason | None = None
    dataset_summary: dict[str, Any] = Field(default_factory=dict)
    split_audit_summary: dict[str, Any] = Field(default_factory=dict)
    profile_summary: dict[str, Any] = Field(default_factory=dict)
    probe_summary: dict[str, Any] = Field(default_factory=dict)
    intervention_summary: dict[str, Any] = Field(default_factory=dict)
    candidate_summary: dict[str, Any] = Field(default_factory=dict)
    runtime_decision_summary: dict[str, Any] = Field(default_factory=dict)
    calibration_summary: dict[str, Any] = Field(default_factory=dict)
    ensemble_summary: dict[str, Any] = Field(default_factory=dict)
    explainability_summary: dict[str, Any] = Field(default_factory=dict)
    final_recommendation: str = ""
    stages_completed: list[PipelineStage] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    plot_paths: dict[str, str] = Field(default_factory=dict)
