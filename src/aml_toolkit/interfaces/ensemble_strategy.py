"""Interface for ensemble combination strategies."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from aml_toolkit.artifacts import EnsembleReport
from aml_toolkit.core.config import ToolkitConfig


class EnsembleStrategy(ABC):
    """Abstract contract for combining multiple candidate models.

    Ensemble strategies combine predictions from multiple models and evaluate
    whether the combination provides meaningful improvement.
    """

    @abstractmethod
    def combine(
        self,
        predictions: list[np.ndarray],
        weights: list[float] | None = None,
    ) -> np.ndarray:
        """Combine predictions from multiple models.

        Args:
            predictions: List of prediction arrays (one per model).
            weights: Optional weights per model.

        Returns:
            Combined prediction array.
        """

    @abstractmethod
    def evaluate_gain(
        self,
        individual_scores: dict[str, float],
        ensemble_score: float,
        config: ToolkitConfig,
    ) -> bool:
        """Determine whether ensemble provides sufficient marginal gain.

        Args:
            individual_scores: Scores of individual models.
            ensemble_score: Score of the combined ensemble.
            config: Toolkit configuration (gain threshold, etc.).

        Returns:
            True if the ensemble should be selected.
        """

    @abstractmethod
    def to_report(
        self,
        member_ids: list[str],
        individual_scores: dict[str, float],
        ensemble_score: float,
        selected: bool,
    ) -> EnsembleReport:
        """Package the ensemble evaluation into a report.

        Args:
            member_ids: IDs of the member models.
            individual_scores: Per-model scores.
            ensemble_score: Combined score.
            selected: Whether the ensemble was chosen.

        Returns:
            EnsembleReport artifact.
        """

    @abstractmethod
    def strategy_name(self) -> str:
        """Return the ensemble strategy identifier."""
