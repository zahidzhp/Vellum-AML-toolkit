"""Class distribution profiling and imbalance detection."""

import numpy as np

from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import RiskFlag


def profile_class_distribution(
    labels: np.ndarray,
    config: ToolkitConfig,
) -> dict:
    """Compute class counts, ratios, and imbalance severity.

    Args:
        labels: 1D label array.
        config: Toolkit configuration (profiling thresholds).

    Returns:
        Dict with class_counts, class_ratios, imbalance_ratio, imbalance_severity,
        and risk_flags.
    """
    if labels.ndim == 2:
        # Multilabel: compute per-label positive counts
        class_counts = {str(i): int(labels[:, i].sum()) for i in range(labels.shape[1])}
        total = labels.shape[0]
        class_ratios = {k: v / total for k, v in class_counts.items()}
        # For multilabel, imbalance is based on the most/least frequent label
        counts = list(class_counts.values())
        max_count = max(counts) if counts else 1
        min_count = min(counts) if counts else 1
    else:
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = {str(u): int(c) for u, c in zip(unique, counts)}
        total = int(labels.shape[0])
        class_ratios = {k: v / total for k, v in class_counts.items()}
        max_count = int(counts.max())
        min_count = int(counts.min())

    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    warning_threshold = config.profiling.imbalance_ratio_warning
    severe_threshold = config.profiling.imbalance_ratio_severe

    risk_flags: list[RiskFlag] = []
    if imbalance_ratio >= severe_threshold:
        severity = "severe"
        risk_flags.append(RiskFlag.CLASS_IMBALANCE)
    elif imbalance_ratio >= warning_threshold:
        severity = "moderate"
        risk_flags.append(RiskFlag.CLASS_IMBALANCE)
    else:
        severity = "none"

    return {
        "class_counts": class_counts,
        "class_ratios": class_ratios,
        "imbalance_ratio": imbalance_ratio,
        "imbalance_severity": severity,
        "risk_flags": risk_flags,
    }
