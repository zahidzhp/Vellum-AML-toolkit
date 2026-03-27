"""Rule evaluation for calibration requirement and thresholding intervention."""

from aml_toolkit.artifacts import DataProfile
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import InterventionType, RiskFlag


def evaluate_calibration_required(
    profile: DataProfile,
    config: ToolkitConfig,
) -> tuple[bool, str]:
    """Evaluate whether calibration should be required.

    When imbalance exists and the config demands it, calibration is flagged
    as a required post-training step.

    Args:
        profile: DataProfile from profiling.
        config: Toolkit configuration.

    Returns:
        Tuple of (required, rationale).
    """
    allowed = InterventionType.CALIBRATION.value in config.interventions.allowed_types

    if not allowed:
        return False, "Calibration not in allowed intervention types."

    if (
        config.interventions.require_calibration_when_imbalanced
        and RiskFlag.CLASS_IMBALANCE in profile.risk_flags
    ):
        return True, (
            "Calibration required: class imbalance detected and "
            "require_calibration_when_imbalanced is enabled in config."
        )

    return False, "Calibration not required by current policy."


def evaluate_thresholding(
    profile: DataProfile,
    config: ToolkitConfig,
) -> tuple[bool, str]:
    """Evaluate whether threshold optimization should be selected.

    Args:
        profile: DataProfile from profiling.
        config: Toolkit configuration.

    Returns:
        Tuple of (selected, rationale).
    """
    allowed = InterventionType.THRESHOLDING.value in config.interventions.allowed_types

    if not allowed:
        return False, "Thresholding not in allowed intervention types."

    if RiskFlag.CLASS_IMBALANCE in profile.risk_flags:
        return True, (
            "Threshold optimization selected: class imbalance detected. "
            "Default 0.5 threshold is likely suboptimal."
        )

    return False, "Threshold optimization not needed; classes appear balanced."
