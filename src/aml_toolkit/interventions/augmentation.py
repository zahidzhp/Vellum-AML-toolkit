"""Rule evaluation for data augmentation intervention."""

from aml_toolkit.artifacts import DataProfile, SplitAuditReport
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import InterventionType


def evaluate_augmentation(
    profile: DataProfile,
    audit_report: SplitAuditReport,
    config: ToolkitConfig,
) -> tuple[bool, str]:
    """Evaluate whether data augmentation should be selected.

    Augmentation is only allowed if the split audit passed (FR-128)
    and augmentation is in the allowed types.

    Args:
        profile: DataProfile from profiling.
        audit_report: SplitAuditReport to check augmentation safety.
        config: Toolkit configuration.

    Returns:
        Tuple of (selected, rationale).
    """
    allowed = InterventionType.AUGMENTATION.value in config.interventions.allowed_types

    if not allowed:
        return False, "Augmentation not in allowed intervention types."

    if not audit_report.augmentation_leakage_safe:
        return False, (
            "Augmentation rejected: split audit did not confirm augmentation safety. "
            "Augmentation is only allowed after split finalization with no blocking issues (FR-128)."
        )

    if not audit_report.passed:
        return False, (
            "Augmentation rejected: split audit has blocking issues. "
            "Augmentation cannot proceed until split integrity is confirmed."
        )

    return True, "Augmentation eligible: split audit passed and augmentation is allowed."
