"""Rule-based intervention planner: consumes audit, profile, and probe evidence to produce an InterventionPlan."""

import logging
from typing import Any

from aml_toolkit.artifacts import (
    DataProfile,
    InterventionEntry,
    InterventionPlan,
    ProbeResultSet,
    SplitAuditReport,
)
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import InterventionType, RiskFlag
from aml_toolkit.interventions.augmentation import evaluate_augmentation
from aml_toolkit.interventions.resampling import (
    evaluate_oversampling,
    evaluate_undersampling,
)
from aml_toolkit.interventions.thresholding import (
    evaluate_calibration_required,
    evaluate_thresholding,
)
from aml_toolkit.interventions.weighting import evaluate_weighting

logger = logging.getLogger("aml_toolkit")


def plan_interventions(
    profile: DataProfile,
    audit_report: SplitAuditReport,
    probe_results: ProbeResultSet | None,
    config: ToolkitConfig,
) -> InterventionPlan:
    """Build a structured intervention plan from evidence.

    Evaluation order:
    1. Class weighting (first-line, safest)
    2. Oversampling (blocked if label noise is high)
    3. Undersampling (fallback)
    4. Augmentation (blocked if audit failed, FR-128)
    5. Thresholding (if imbalance detected)
    6. Calibration (flagged as required if imbalanced + config demands it)

    If no intervention is viable and severe risk flags exist, the planner
    may suggest abstention.

    Args:
        profile: DataProfile from profiling.
        audit_report: SplitAuditReport from auditing.
        probe_results: ProbeResultSet from probes (optional).
        config: Toolkit configuration.

    Returns:
        InterventionPlan with selected/rejected interventions and execution order.
    """
    selected: list[InterventionEntry] = []
    rejected: list[InterventionEntry] = []
    safety_constraints: list[str] = []
    execution_order: list[InterventionType] = []

    # 1. Class weighting
    weight_ok, weight_reason = evaluate_weighting(profile, config)
    entry = InterventionEntry(
        intervention_type=InterventionType.CLASS_WEIGHTING,
        selected=weight_ok,
        rationale=weight_reason,
    )
    if weight_ok:
        selected.append(entry)
        execution_order.append(InterventionType.CLASS_WEIGHTING)
        logger.info(f"Selected: CLASS_WEIGHTING — {weight_reason}")
    else:
        rejected.append(entry)

    # 2. Oversampling
    os_ok, os_reason = evaluate_oversampling(profile, probe_results, config)
    entry = InterventionEntry(
        intervention_type=InterventionType.OVERSAMPLING,
        selected=os_ok,
        rationale=os_reason,
    )
    if os_ok:
        selected.append(entry)
        execution_order.append(InterventionType.OVERSAMPLING)
        logger.info(f"Selected: OVERSAMPLING — {os_reason}")
    else:
        rejected.append(entry)
        if "noise" in os_reason.lower() or "conflict" in os_reason.lower():
            safety_constraints.append(
                "Oversampling blocked due to label noise/conflict risk."
            )

    # 3. Undersampling
    us_ok, us_reason = evaluate_undersampling(profile, config)
    entry = InterventionEntry(
        intervention_type=InterventionType.UNDERSAMPLING,
        selected=us_ok,
        rationale=us_reason,
    )
    if us_ok:
        selected.append(entry)
        execution_order.append(InterventionType.UNDERSAMPLING)
    else:
        rejected.append(entry)

    # 4. Augmentation
    aug_ok, aug_reason = evaluate_augmentation(profile, audit_report, config)
    entry = InterventionEntry(
        intervention_type=InterventionType.AUGMENTATION,
        selected=aug_ok,
        rationale=aug_reason,
    )
    if aug_ok:
        selected.append(entry)
        execution_order.append(InterventionType.AUGMENTATION)
    else:
        rejected.append(entry)
        if "FR-128" in aug_reason:
            safety_constraints.append(
                "Augmentation blocked: split audit requirement (FR-128)."
            )

    # 5. Thresholding
    thresh_ok, thresh_reason = evaluate_thresholding(profile, config)
    entry = InterventionEntry(
        intervention_type=InterventionType.THRESHOLDING,
        selected=thresh_ok,
        rationale=thresh_reason,
    )
    if thresh_ok:
        selected.append(entry)
        execution_order.append(InterventionType.THRESHOLDING)
    else:
        rejected.append(entry)

    # 6. Calibration requirement
    cal_ok, cal_reason = evaluate_calibration_required(profile, config)
    entry = InterventionEntry(
        intervention_type=InterventionType.CALIBRATION,
        selected=cal_ok,
        rationale=cal_reason,
    )
    if cal_ok:
        selected.append(entry)
        execution_order.append(InterventionType.CALIBRATION)
    else:
        rejected.append(entry)

    # Build overall rationale
    if selected:
        rationale = (
            f"{len(selected)} intervention(s) selected based on "
            f"profile risk flags: {[f.value for f in profile.risk_flags]}."
        )
    else:
        rationale = "No interventions selected; data appears clean or no applicable interventions."

    # Check for abstention suggestion
    if _should_suggest_abstention(profile, audit_report, selected):
        safety_constraints.append(
            "WARNING: Severe risk flags with limited viable interventions. "
            "Consider abstention if candidate quality is insufficient."
        )

    plan = InterventionPlan(
        selected_interventions=selected,
        rejected_interventions=rejected,
        rationale=rationale,
        safety_constraints=safety_constraints,
        execution_order=execution_order,
    )

    return plan


def _should_suggest_abstention(
    profile: DataProfile,
    audit_report: SplitAuditReport,
    selected: list[InterventionEntry],
) -> bool:
    """Check if abstention should be suggested due to severe unaddressed risks."""
    severe_flags = {RiskFlag.LEAKAGE, RiskFlag.LABEL_NOISE}
    active_severe = severe_flags & set(profile.risk_flags)

    if not audit_report.passed:
        return True

    if active_severe and not selected:
        return True

    return False
