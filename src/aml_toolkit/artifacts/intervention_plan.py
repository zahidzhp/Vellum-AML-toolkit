"""Typed artifact for intervention planning output."""

from pydantic import BaseModel, Field

from aml_toolkit.core.enums import InterventionType


class InterventionEntry(BaseModel):
    """A single selected or rejected intervention with rationale."""

    intervention_type: InterventionType
    selected: bool
    rationale: str = ""


class InterventionPlan(BaseModel):
    """Structured plan of which interventions to apply and why."""

    selected_interventions: list[InterventionEntry] = Field(default_factory=list)
    rejected_interventions: list[InterventionEntry] = Field(default_factory=list)
    rationale: str = ""
    safety_constraints: list[str] = Field(default_factory=list)
    execution_order: list[InterventionType] = Field(default_factory=list)
