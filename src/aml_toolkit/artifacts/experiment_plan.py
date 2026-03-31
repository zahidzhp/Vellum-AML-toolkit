"""Experiment plan artifact — output of the agentic planner."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExperimentProposal(BaseModel):
    """A single actionable proposal from the experiment planner."""

    action: str  # e.g. "add_class_weighting", "reduce_max_candidates"
    rationale: str
    config_patch: dict[str, Any] = Field(default_factory=dict)
    priority: int = 5  # 1 = highest priority
    source: str = "rule_engine"  # "rule_engine" | "llm"
    tags: list[str] = Field(default_factory=list)


class ExperimentPlan(BaseModel):
    """Full experiment plan returned by the ExperimentPlanner."""

    proposals: list[ExperimentProposal] = Field(default_factory=list)
    mode: str = "propose_only"  # "propose_only" | "auto_apply"
    history_records_used: int = 0
    rules_evaluated: int = 0
    rules_triggered: int = 0
    notes: list[str] = Field(default_factory=list)
