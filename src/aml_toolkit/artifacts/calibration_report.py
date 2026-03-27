"""Typed artifact for calibration and threshold optimization output."""

from pydantic import BaseModel, Field


class CalibrationResult(BaseModel):
    """Calibration result for a single candidate."""

    candidate_id: str
    method: str
    ece_before: float | None = None
    ece_after: float | None = None
    brier_before: float | None = None
    brier_after: float | None = None
    threshold_optimized: float | None = None
    objective_metric: str = "ece"
    notes: list[str] = Field(default_factory=list)


class CalibrationReport(BaseModel):
    """Aggregated calibration results across candidates."""

    results: list[CalibrationResult] = Field(default_factory=list)
    primary_objective: str = "ece"
    warnings: list[str] = Field(default_factory=list)
