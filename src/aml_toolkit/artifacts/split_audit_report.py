"""Typed artifact for split auditing output."""

from pydantic import BaseModel, Field


class SplitAuditReport(BaseModel):
    """Results of split integrity validation and leakage checks."""

    passed: bool
    leakage_flags: list[str] = Field(default_factory=list)
    duplicate_overlap_summary: dict[str, int] = Field(default_factory=dict)
    entity_leakage_summary: dict[str, str] = Field(default_factory=dict)
    temporal_leakage_summary: dict[str, str] = Field(default_factory=dict)
    class_absence_flags: list[str] = Field(default_factory=list)
    augmentation_leakage_safe: bool = True
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
