"""Rule evaluation for class weighting intervention."""

from aml_toolkit.artifacts import DataProfile
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import InterventionType, RiskFlag


def evaluate_weighting(
    profile: DataProfile,
    config: ToolkitConfig,
) -> tuple[bool, str]:
    """Evaluate whether class weighting should be selected.

    Class weighting is preferred as the first-line intervention for imbalance
    because it does not alter the data distribution (no synthetic samples).

    Args:
        profile: DataProfile from profiling.
        config: Toolkit configuration.

    Returns:
        Tuple of (selected, rationale).
    """
    allowed = InterventionType.CLASS_WEIGHTING.value in config.interventions.allowed_types

    if not allowed:
        return False, "Class weighting not in allowed intervention types."

    if RiskFlag.CLASS_IMBALANCE not in profile.risk_flags:
        return False, "No class imbalance detected; weighting not needed."

    return True, (
        f"Class imbalance detected (severity: {profile.imbalance_severity}). "
        "Class weighting selected as first-line intervention because it adjusts "
        "loss contribution without altering the data."
    )
