"""Uncertainty estimation — entropy, margin, and conformal prediction sets."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from aml_toolkit.artifacts.uncertainty_report import UncertaintyReport
from aml_toolkit.core.config import UncertaintyConfig
from aml_toolkit.uncertainty.conformal import SplitConformalPredictor

logger = logging.getLogger("aml_toolkit")


class UncertaintyEstimator:
    """Estimates predictive uncertainty for a set of candidate models.

    Supports:
    - Entropy: H(p) = -Σ p_k log(p_k)
    - Margin: 1 - (p_max - p_second_max)
    - Conformal prediction sets (SplitConformalPredictor)

    All methods operate on probability arrays — modality agnostic.
    For images, use the calibrated probas already stored in cal_report.plot_data.
    """

    def __init__(self, config: UncertaintyConfig):
        self.config = config

    def estimate(
        self,
        candidate_id: str,
        proba: np.ndarray,
        y_val: np.ndarray | None = None,
    ) -> UncertaintyReport:
        """Estimate uncertainty for a single candidate.

        Args:
            candidate_id: Identifier for the candidate model.
            proba: Predicted probabilities, shape (n, K) or (n,) for binary.
            y_val: True labels (required for conformal fitting and coverage checks).

        Returns:
            UncertaintyReport with all computed metrics.
        """
        report = UncertaintyReport(candidate_id=candidate_id)

        try:
            proba = np.asarray(proba, dtype=np.float64)
            proba_2d = self._ensure_2d(proba)
            n = len(proba_2d)
            report.sample_count = n

            uncertainty_scores: list[np.ndarray] = []

            if "entropy" in self.config.methods:
                entropy = self._compute_entropy(proba_2d)
                report.entropy_mean = float(np.mean(entropy))
                uncertainty_scores.append(entropy)
                report.methods_used.append("entropy")

            if "margin" in self.config.methods:
                margin_uncertainty = self._compute_margin_uncertainty(proba_2d)
                report.margin_mean = float(np.mean(margin_uncertainty))
                uncertainty_scores.append(margin_uncertainty)
                report.methods_used.append("margin")

            # Aggregate
            if uncertainty_scores:
                if self.config.aggregation == "max":
                    agg = np.max(np.stack(uncertainty_scores, axis=1), axis=1)
                else:  # default: mean
                    agg = np.mean(np.stack(uncertainty_scores, axis=1), axis=1)

                report.mean_uncertainty = float(np.mean(agg))
                high_mask = agg > self.config.abstain_if_above
                report.pct_high_uncertainty = float(np.mean(high_mask))

                # Abstention check
                if report.mean_uncertainty > self.config.abstain_if_above:
                    report.abstention_triggered = True
                    report.abstention_reason = (
                        f"Mean uncertainty {report.mean_uncertainty:.3f} exceeds "
                        f"threshold {self.config.abstain_if_above}"
                    )

            # Conformal prediction
            if self.config.conformal_enabled and y_val is not None:
                self._run_conformal(report, proba_2d, y_val)

        except Exception as e:
            logger.warning(f"UncertaintyEstimator.estimate failed for {candidate_id}: {e}")

        return report

    def _compute_entropy(self, proba_2d: np.ndarray) -> np.ndarray:
        """Compute Shannon entropy per sample. Returns shape (n,)."""
        # Clip to avoid log(0)
        p = np.clip(proba_2d, 1e-12, 1.0)
        entropy = -np.sum(p * np.log(p), axis=1)
        # Normalize by log(K) to put in [0, 1]
        k = proba_2d.shape[1]
        if k > 1:
            entropy = entropy / np.log(k)
        return entropy

    def _compute_margin_uncertainty(self, proba_2d: np.ndarray) -> np.ndarray:
        """Compute margin uncertainty = 1 - (p_max - p_2nd_max). Returns shape (n,)."""
        if proba_2d.shape[1] == 1:
            return np.zeros(len(proba_2d))
        sorted_p = np.sort(proba_2d, axis=1)[:, ::-1]
        margin = sorted_p[:, 0] - sorted_p[:, 1]
        return 1.0 - margin

    def _run_conformal(
        self,
        report: UncertaintyReport,
        proba_2d: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Fit and evaluate conformal predictor. Updates report in-place."""
        try:
            predictor = SplitConformalPredictor(coverage=self.config.conformal_coverage)
            predictor.fit(proba_2d, y_val)

            sets = predictor.predict_sets(proba_2d)
            y_val_int = np.asarray(y_val, dtype=np.int64)

            # Empirical coverage
            covered = sum(1 for s, y in zip(sets, y_val_int) if int(y) in s)
            report.conformal_coverage_achieved = covered / len(y_val_int)

            # Efficiency (mean set size)
            report.mean_prediction_set_size = float(np.mean([len(s) for s in sets]))

            # % singleton sets
            singletons = sum(1 for s in sets if len(s) == 1)
            report.pct_singleton_sets = singletons / len(sets)

            report.methods_used.append("conformal")
        except Exception as e:
            logger.warning(f"Conformal prediction failed: {e}")

    def _ensure_2d(self, proba: np.ndarray) -> np.ndarray:
        """Convert (n,) binary proba to (n, 2) shape."""
        if proba.ndim == 1:
            return np.stack([1.0 - proba, proba], axis=1)
        return proba
