"""Typed artifact for data profiling output."""

from typing import Any

from pydantic import BaseModel, Field

from aml_toolkit.core.enums import RiskFlag


class DataProfile(BaseModel):
    """Dataset health profile with risk flags and statistical summaries."""

    total_samples: int = 0
    class_counts: dict[str, int] = Field(default_factory=dict)
    class_ratios: dict[str, float] = Field(default_factory=dict)
    imbalance_severity: str = "none"
    missingness_summary: dict[str, float] = Field(default_factory=dict)
    duplicate_summary: dict[str, Any] = Field(default_factory=dict)
    label_conflict_summary: dict[str, Any] = Field(default_factory=dict)
    outlier_summary: dict[str, Any] = Field(default_factory=dict)
    subgroup_summary: dict[str, Any] = Field(default_factory=dict)
    ood_shift_summary: dict[str, Any] = Field(default_factory=dict)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
