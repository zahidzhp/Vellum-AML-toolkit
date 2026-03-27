"""Augmentation guard: blocks augmentation before split finalization (FR-128)."""

from aml_toolkit.artifacts import SplitAuditReport
from aml_toolkit.core.exceptions import SplitIntegrityError


class AugmentationGuard:
    """Enforces that augmentation/resampling only runs after split audit passes.

    This guard is checked before any intervention that modifies training data.
    It prevents augmentation leakage by ensuring the split is finalized and
    validated before any data transformation occurs.
    """

    def __init__(self) -> None:
        self._split_finalized = False
        self._audit_passed = False

    def mark_split_finalized(self) -> None:
        """Mark that splits have been created."""
        self._split_finalized = True

    def mark_audit_passed(self, audit_report: SplitAuditReport) -> None:
        """Mark that the split audit has been completed.

        Args:
            audit_report: The audit report. Only marks as passed if the
                report has no blocking issues.
        """
        self._split_finalized = True
        self._audit_passed = audit_report.passed and not audit_report.blocking_issues

    def check_augmentation_allowed(self) -> None:
        """Check if augmentation/resampling is safe to proceed.

        Raises:
            SplitIntegrityError: If augmentation is requested before
                split finalization or audit passage.
        """
        if not self._split_finalized:
            raise SplitIntegrityError(
                "Augmentation requested before split finalization. "
                "Splits must be created before any data augmentation or resampling. "
                "This prevents augmentation leakage (FR-128)."
            )
        if not self._audit_passed:
            raise SplitIntegrityError(
                "Augmentation requested before split audit passed. "
                "The split audit must complete with no blocking issues before "
                "any data augmentation or resampling is allowed (FR-128)."
            )

    @property
    def is_augmentation_safe(self) -> bool:
        """Return True if augmentation is currently allowed."""
        return self._split_finalized and self._audit_passed
