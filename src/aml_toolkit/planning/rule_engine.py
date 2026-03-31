"""Standalone rule engine for experiment planning.

Rules are separated from the planner to enable independent testing.
Each rule has a condition (run_state → bool) and a proposal factory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from aml_toolkit.artifacts.experiment_plan import ExperimentProposal

logger = logging.getLogger("aml_toolkit")


@dataclass
class PlannerRule:
    """A single rule: condition + proposal generator."""

    name: str
    condition: Callable[[dict[str, Any]], bool]
    make_proposal: Callable[[dict[str, Any], Any], ExperimentProposal]
    tags: list[str] = field(default_factory=list)


def _safe_condition(rule: PlannerRule, run_state: dict[str, Any]) -> bool:
    """Evaluate a rule's condition, returning False on any error."""
    try:
        return bool(rule.condition(run_state))
    except Exception as e:
        logger.debug(f"Rule '{rule.name}' condition error (skipped): {e}")
        return False


# ---------------------------------------------------------------------------
# Built-in rules (tabular + image)
# ---------------------------------------------------------------------------

RULES: list[PlannerRule] = [
    # --- Tabular: severe imbalance → try class weighting ---
    PlannerRule(
        name="high_imbalance_try_class_weighting",
        condition=lambda s: (
            s.get("imbalance_severity") == "severe"
            and "CLASS_WEIGHTING" in s.get("allowed_interventions", [])
        ),
        make_proposal=lambda s, cfg: ExperimentProposal(
            action="add_class_weighting",
            rationale=(
                "Imbalance ratio is severe (>20x). Class weighting is a low-risk "
                "first intervention before oversampling."
            ),
            config_patch={"interventions": {"allowed_types": ["CLASS_WEIGHTING"]}},
            priority=1,
            tags=["imbalance", "tabular"],
        ),
        tags=["imbalance", "tabular"],
    ),

    # --- Tabular: high ECE after calibration → reduce candidates ---
    PlannerRule(
        name="poor_calibration_reduce_candidates",
        condition=lambda s: any(
            float(r.get("ece_after", 0.0)) > 0.15
            for r in s.get("calibration_results", [])
        ),
        make_proposal=lambda s, cfg: ExperimentProposal(
            action="reduce_max_candidates",
            rationale=(
                "High ECE after calibration suggests models may be overfitting. "
                "Reducing max candidates focuses resources on fewer, better-tuned models."
            ),
            config_patch={
                "candidates": {
                    "max_candidates": max(
                        1,
                        (cfg.candidates.max_candidates - 1) if cfg else 2,
                    )
                }
            },
            priority=2,
            tags=["calibration", "overfit"],
        ),
        tags=["calibration", "overfit"],
    ),

    # --- Tabular: high label noise → enable noise-robust interventions ---
    PlannerRule(
        name="label_noise_enable_robust_training",
        condition=lambda s: s.get("has_label_noise", False),
        make_proposal=lambda s, cfg: ExperimentProposal(
            action="note_label_noise_detected",
            rationale=(
                "Label noise detected. Consider label smoothing or noise-robust loss "
                "functions. Cross-validation is recommended over single-split evaluation."
            ),
            config_patch={},
            priority=3,
            tags=["label_noise", "tabular"],
        ),
        tags=["label_noise"],
    ),

    # --- Both: OOD shift → recommend conservative mode ---
    PlannerRule(
        name="ood_shift_recommend_conservative",
        condition=lambda s: s.get("has_ood_shift", False),
        make_proposal=lambda s, cfg: ExperimentProposal(
            action="recommend_conservative_mode",
            rationale=(
                "OOD distribution shift detected between train and test. "
                "Conservative operating mode is recommended to avoid overconfident predictions."
            ),
            config_patch={"mode": "CONSERVATIVE"},
            priority=2,
            tags=["ood_shift", "distribution_shift"],
        ),
        tags=["ood_shift"],
    ),

    # --- Image: small dataset → recommend augmentation ---
    PlannerRule(
        name="image_small_dataset_try_augmentation",
        condition=lambda s: (
            s.get("modality") == "IMAGE"
            and float(s.get("n_samples", 10000)) < 5000
            and "AUGMENTATION" in s.get("allowed_interventions", [])
        ),
        make_proposal=lambda s, cfg: ExperimentProposal(
            action="enable_augmentation",
            rationale=(
                "Small image dataset (<5k samples). Data augmentation is highly "
                "recommended to prevent overfitting."
            ),
            config_patch={"interventions": {"allowed_types": ["AUGMENTATION"]}},
            priority=1,
            tags=["image", "small_dataset", "augmentation"],
        ),
        tags=["image", "augmentation"],
    ),

    # --- Image: large images + small dataset → prefer resnet18 over ViT ---
    PlannerRule(
        name="image_small_dataset_prefer_resnet",
        condition=lambda s: (
            s.get("modality") == "IMAGE"
            and float(s.get("n_samples", 10000)) < 5000
        ),
        make_proposal=lambda s, cfg: ExperimentProposal(
            action="prefer_resnet18_backbone",
            rationale=(
                "Small image dataset (<5k samples). ResNet18 typically generalizes "
                "better than ViT on small datasets due to inductive biases."
            ),
            config_patch={"candidates": {"cnn_backbone": "resnet18"}},
            priority=2,
            tags=["image", "backbone"],
        ),
        tags=["image", "backbone"],
    ),

    # --- Both: high uncertainty mean → suggest k-fold cross-validation ---
    PlannerRule(
        name="high_uncertainty_suggest_kfold",
        condition=lambda s: float(s.get("mean_uncertainty", 0.0)) > 0.7,
        make_proposal=lambda s, cfg: ExperimentProposal(
            action="suggest_kfold_cross_validation",
            rationale=(
                "High mean uncertainty detected. k-fold cross-validation provides "
                "more reliable uncertainty estimates than a single validation split."
            ),
            config_patch={"advanced": {"uncertainty": {"use_cross_val": True}}},
            priority=3,
            tags=["uncertainty", "cross_validation"],
        ),
        tags=["uncertainty"],
    ),
]


def evaluate_rules(
    run_state: dict[str, Any],
    config: Any,
    rules: list[PlannerRule] | None = None,
) -> list[ExperimentProposal]:
    """Evaluate all rules against the current run state.

    Args:
        run_state: Dict with current pipeline state (profile, calibration, etc.)
        config: ToolkitConfig for constraint checking in proposals.
        rules: Rules to evaluate (defaults to RULES).

    Returns:
        List of triggered proposals sorted by priority.
    """
    if rules is None:
        rules = RULES

    proposals: list[ExperimentProposal] = []
    for rule in rules:
        if _safe_condition(rule, run_state):
            try:
                proposal = rule.make_proposal(run_state, config)
                proposals.append(proposal)
                logger.debug(f"Rule '{rule.name}' triggered: {proposal.action}")
            except Exception as e:
                logger.warning(f"Rule '{rule.name}' proposal failed: {e}")

    proposals.sort(key=lambda p: p.priority)
    return proposals
