"""Soft voting ensemble strategy: averages probability outputs across models."""

import numpy as np

from aml_toolkit.artifacts import EnsembleReport
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.interfaces.ensemble_strategy import EnsembleStrategy


class SoftVotingStrategy(EnsembleStrategy):
    """Combine models by averaging their probability outputs (equal weights)."""

    def combine(
        self,
        predictions: list[np.ndarray],
        weights: list[float] | None = None,
    ) -> np.ndarray:
        if not predictions:
            raise ValueError("No predictions to combine.")
        stacked = np.stack(predictions)
        if weights is not None:
            w = np.array(weights).reshape(-1, *([1] * (stacked.ndim - 1)))
            return (stacked * w).sum(axis=0) / sum(weights)
        return stacked.mean(axis=0)

    def evaluate_gain(
        self,
        individual_scores: dict[str, float],
        ensemble_score: float,
        config: ToolkitConfig,
    ) -> bool:
        if not individual_scores:
            return False
        best_individual = max(individual_scores.values())
        gain = ensemble_score - best_individual
        return gain >= config.ensemble.marginal_gain_threshold

    def to_report(
        self,
        member_ids: list[str],
        individual_scores: dict[str, float],
        ensemble_score: float,
        selected: bool,
    ) -> EnsembleReport:
        best_individual = max(individual_scores.values()) if individual_scores else 0.0
        gain = ensemble_score - best_individual
        rejection_reason = None
        if not selected:
            rejection_reason = (
                f"Marginal gain {gain:.4f} below threshold. "
                f"Best individual score: {best_individual:.4f}, ensemble: {ensemble_score:.4f}."
            )
        return EnsembleReport(
            ensemble_selected=selected,
            strategy=self.strategy_name(),
            member_ids=member_ids,
            individual_scores=individual_scores,
            ensemble_score=ensemble_score,
            marginal_gain=gain,
            rejection_reason=rejection_reason,
        )

    def strategy_name(self) -> str:
        return "soft_voting"
