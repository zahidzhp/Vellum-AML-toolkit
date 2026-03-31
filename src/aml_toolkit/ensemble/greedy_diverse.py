"""Greedy diversity-aware ensemble member selection.

Implements forward greedy pruning — selects models that maximize both
performance gain AND complementary diversity, preventing redundant clones.

Why this beats independent weighting:
- Two identical high-confidence models get 0 diversity bonus (disagreement = 0)
- A lower-confidence model that disagrees on hard samples DOES improve ensemble
- Result: strictly better bias-variance tradeoff than soft_voting or weighted_avg
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import f1_score

from aml_toolkit.artifacts import EnsembleReport
from aml_toolkit.core.config import DynamicEnsembleConfig, ToolkitConfig
from aml_toolkit.ensemble.diversity_metrics import (
    ambiguity_decomposition,
    ensemble_diversity_score,
    pairwise_disagreement,
)

logger = logging.getLogger("aml_toolkit")


class GreedyDiverseEnsemble:
    """Forward greedy ensemble selection with diversity regularization.

    Algorithm:
    1. Start with the best individual model (by F1).
    2. For each remaining model, compute:
         gain = F1(current_ensemble + model) - F1(current_ensemble)
         diversity = pairwise_disagreement(model, current_ensemble_members)
         score = gain + diversity_weight * diversity
    3. Add model if score > 0 AND diversity > diversity_threshold.
    4. Stop when max_members reached or no model improves score.

    Usage:
        ensemble = GreedyDiverseEnsemble(config)
        result = ensemble.select(probas, y_val, candidate_scores)
    """

    def __init__(self, config: DynamicEnsembleConfig):
        self.config = config
        self.diversity_weight: float = 0.3  # lambda in score formula

    def select(
        self,
        probas: dict[str, np.ndarray],
        y_val: np.ndarray,
        candidate_scores: dict[str, float] | None = None,
    ) -> EnsembleReport:
        """Run greedy selection and return an EnsembleReport.

        Args:
            probas: candidate_id → probability array (n, K) or (n,).
            y_val: True labels (n,).
            candidate_scores: Optional pre-computed F1 scores (avoids recomputation).

        Returns:
            EnsembleReport with selected member IDs and diversity metrics.
        """
        if not probas:
            return EnsembleReport(
                rejection_reason="No candidates provided to GreedyDiverseEnsemble."
            )

        try:
            return self._greedy_select(probas, y_val, candidate_scores)
        except Exception as e:
            logger.warning(f"GreedyDiverseEnsemble.select failed: {e}")
            return EnsembleReport(rejection_reason=f"Greedy selection error: {e}")

    def _greedy_select(
        self,
        probas: dict[str, np.ndarray],
        y_val: np.ndarray,
        candidate_scores: dict[str, float] | None,
    ) -> EnsembleReport:
        # Ensure 2D probas
        probas_2d = {cid: self._ensure_2d(p) for cid, p in probas.items()}

        # Compute individual F1 if not provided
        if candidate_scores is None:
            candidate_scores = {}
            for cid, p in probas_2d.items():
                preds = np.argmax(p, axis=1)
                candidate_scores[cid] = float(
                    f1_score(y_val, preds, average="macro", zero_division=0)
                )

        # Sort by individual score (best first)
        sorted_ids = sorted(candidate_scores.keys(), key=lambda c: candidate_scores[c], reverse=True)

        # Greedy forward selection
        selected: list[str] = [sorted_ids[0]]
        remaining = sorted_ids[1:]

        current_f1 = candidate_scores[sorted_ids[0]]
        max_members = min(self.config.max_members, len(probas_2d))

        while remaining and len(selected) < max_members:
            best_score = 0.0
            best_cid = None

            ensemble_preds_current = [probas_2d[cid] for cid in selected]

            for cid in remaining:
                p_new = probas_2d[cid]

                # Candidate ensemble predictions
                candidate_probas = ensemble_preds_current + [p_new]
                ensemble_proba = np.mean(candidate_probas, axis=0)
                ensemble_preds = np.argmax(ensemble_proba, axis=1)
                new_f1 = float(f1_score(y_val, ensemble_preds, average="macro", zero_division=0))

                gain = new_f1 - current_f1

                # Diversity: mean disagreement with current selected members
                preds_new = np.argmax(p_new, axis=1)
                disagree_scores = []
                for sel_cid in selected:
                    preds_sel = np.argmax(probas_2d[sel_cid], axis=1)
                    disagree_scores.append(pairwise_disagreement(preds_new, preds_sel))
                diversity = float(np.mean(disagree_scores)) if disagree_scores else 0.0

                # Skip if diversity below threshold (clone detection)
                if diversity < self.config.diversity_threshold and len(selected) > 0:
                    continue

                score = gain + self.diversity_weight * diversity

                if score > best_score:
                    best_score = score
                    best_cid = cid

            if best_cid is None:
                break  # No beneficial addition found

            # Compute new ensemble F1 with this addition
            all_probas = [probas_2d[cid] for cid in selected] + [probas_2d[best_cid]]
            ensemble_proba = np.mean(all_probas, axis=0)
            ensemble_preds = np.argmax(ensemble_proba, axis=1)
            current_f1 = float(f1_score(y_val, ensemble_preds, average="macro", zero_division=0))

            selected.append(best_cid)
            remaining.remove(best_cid)

        # Compute final ensemble score
        final_probas = [probas_2d[cid] for cid in selected]
        if len(selected) > 1:
            final_proba = np.mean(final_probas, axis=0)
        else:
            final_proba = final_probas[0]
        final_preds = np.argmax(final_proba, axis=1)
        final_f1 = float(f1_score(y_val, final_preds, average="macro", zero_division=0))

        # Diversity metrics
        hard_preds = [np.argmax(probas_2d[cid], axis=1) for cid in selected]
        div_score = ensemble_diversity_score(hard_preds)
        amb_decomp = ambiguity_decomposition(final_probas, y_val)

        # Marginal gain vs best individual
        best_individual_f1 = max(candidate_scores.values())
        marginal_gain = final_f1 - best_individual_f1

        ensemble_selected = len(selected) > 1 and marginal_gain > 0

        return EnsembleReport(
            ensemble_selected=ensemble_selected,
            strategy="greedy_diverse",
            member_ids=selected,
            individual_scores=candidate_scores,
            ensemble_score=final_f1,
            marginal_gain=marginal_gain,
            diversity_score=div_score,
            ambiguity_decomposition=amb_decomp,
            notes=[
                f"Selected {len(selected)} members via greedy diversity pruning.",
                f"Diversity score: {div_score:.3f}",
                f"Marginal gain: {marginal_gain:+.4f}",
            ],
        )

    def _ensure_2d(self, proba: np.ndarray) -> np.ndarray:
        """Convert (n,) binary proba to (n, 2) shape."""
        proba = np.asarray(proba, dtype=np.float64)
        if proba.ndim == 1:
            return np.stack([1.0 - proba, proba], axis=1)
        return proba
