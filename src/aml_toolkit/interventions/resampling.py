"""Rule evaluation for oversampling and undersampling interventions."""

from aml_toolkit.artifacts import DataProfile, ProbeResultSet
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import InterventionType, RiskFlag


def evaluate_oversampling(
    profile: DataProfile,
    probe_results: ProbeResultSet | None,
    config: ToolkitConfig,
) -> tuple[bool, str]:
    """Evaluate whether oversampling (e.g. SMOTE) should be selected.

    Oversampling is blocked when label noise/conflict risk is high, because
    generating synthetic samples from noisy data amplifies the noise.

    Args:
        profile: DataProfile from profiling.
        probe_results: ProbeResultSet (for sensitivity info).
        config: Toolkit configuration.

    Returns:
        Tuple of (selected, rationale).
    """
    allowed = InterventionType.OVERSAMPLING.value in config.interventions.allowed_types

    if not allowed:
        return False, "Oversampling not in allowed intervention types."

    if RiskFlag.CLASS_IMBALANCE not in profile.risk_flags:
        return False, "No class imbalance detected; oversampling not needed."

    # Block if label noise/conflict risk is too high
    conflict_fraction = profile.label_conflict_summary.get("conflict_fraction", 0.0)
    noise_threshold = config.interventions.oversampling_noise_risk_threshold

    if RiskFlag.LABEL_CONFLICT in profile.risk_flags or conflict_fraction > noise_threshold:
        return False, (
            f"Oversampling rejected: label conflict/noise risk is too high "
            f"(conflict fraction: {conflict_fraction:.3f}, "
            f"threshold: {noise_threshold}). "
            "Oversampling would amplify noisy labels."
        )

    if RiskFlag.LABEL_NOISE in profile.risk_flags:
        return False, (
            "Oversampling rejected: LABEL_NOISE risk flag is active. "
            "Generating synthetic samples from noisy data is unsafe."
        )

    return True, (
        f"Class imbalance detected (severity: {profile.imbalance_severity}). "
        "Oversampling selected because no significant label noise/conflict detected."
    )


def evaluate_undersampling(
    profile: DataProfile,
    config: ToolkitConfig,
) -> tuple[bool, str]:
    """Evaluate whether undersampling should be selected.

    Undersampling is a safe fallback when oversampling is blocked, but it
    discards majority-class data which may hurt performance on large enough datasets.

    Args:
        profile: DataProfile from profiling.
        config: Toolkit configuration.

    Returns:
        Tuple of (selected, rationale).
    """
    allowed = InterventionType.UNDERSAMPLING.value in config.interventions.allowed_types

    if not allowed:
        return False, "Undersampling not in allowed intervention types."

    if RiskFlag.CLASS_IMBALANCE not in profile.risk_flags:
        return False, "No class imbalance detected; undersampling not needed."

    return True, (
        f"Class imbalance detected (severity: {profile.imbalance_severity}). "
        "Undersampling available as fallback when oversampling is blocked."
    )
