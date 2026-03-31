"""Experiment planner — rule-based proposals with optional LLM enhancement."""

from __future__ import annotations

import logging
import os
from typing import Any

from aml_toolkit.artifacts.experiment_plan import ExperimentPlan, ExperimentProposal
from aml_toolkit.artifacts.run_history import RunHistoryRecord
from aml_toolkit.core.config import AgenticPlannerConfig
from aml_toolkit.planning.rule_engine import RULES, PlannerRule, evaluate_rules

logger = logging.getLogger("aml_toolkit")


class ExperimentPlanner:
    """Generates experiment proposals from run state.

    Architecture:
    1. Always run rule engine (no external deps, deterministic).
    2. Optionally enhance with LLM (Claude API, opt-in via llm_enhanced=True).
       - LLM can only ADD proposals, never override rule engine.
       - Requires ANTHROPIC_API_KEY in environment; silently skips if absent.
    3. Enforce hard constraints (config values always win).
    4. Sort by priority, return top max_suggestions.

    The planner is mode-aware: mode="propose_only" means proposals are logged
    but NOT automatically applied to config.
    """

    def __init__(
        self,
        config: AgenticPlannerConfig,
        extra_rules: list[PlannerRule] | None = None,
    ):
        self.config = config
        self.rules = RULES + (extra_rules or [])

    def plan(
        self,
        run_state: dict[str, Any],
        toolkit_config: Any = None,  # ToolkitConfig
        history: list[RunHistoryRecord] | None = None,
    ) -> ExperimentPlan:
        """Generate experiment proposals.

        Args:
            run_state: Current pipeline state (profile, calibration results, etc.).
            toolkit_config: Full ToolkitConfig for constraint checking.
            history: Past run records (used for context in LLM mode).

        Returns:
            ExperimentPlan with up to max_suggestions proposals.
        """
        history = history or []

        try:
            return self._plan(run_state, toolkit_config, history)
        except Exception as e:
            logger.warning(f"ExperimentPlanner.plan failed: {e}")
            return ExperimentPlan(notes=[f"Planner error: {e}"])

    def _plan(
        self,
        run_state: dict[str, Any],
        toolkit_config: Any,
        history: list[RunHistoryRecord],
    ) -> ExperimentPlan:
        # Step 1: Rule engine (always runs)
        rule_proposals = evaluate_rules(run_state, toolkit_config, self.rules)
        rules_triggered = len(rule_proposals)

        # Step 2: Optional LLM enhancement
        all_proposals = list(rule_proposals)
        if self.config.llm_enhanced:
            llm_proposals = self._llm_enhance(run_state, toolkit_config, rule_proposals)
            existing_actions = {p.action for p in rule_proposals}
            for p in llm_proposals:
                if p.action not in existing_actions:
                    all_proposals.append(p)

        # Step 3: Enforce hard constraints
        all_proposals = self._enforce_constraints(all_proposals, toolkit_config)

        # Step 4: Sort by priority, take top max_suggestions
        all_proposals.sort(key=lambda p: p.priority)
        final_proposals = all_proposals[: self.config.max_suggestions]

        return ExperimentPlan(
            proposals=final_proposals,
            mode=self.config.mode,
            history_records_used=len(history),
            rules_evaluated=len(self.rules),
            rules_triggered=rules_triggered,
            notes=[
                f"Rule engine: {rules_triggered}/{len(self.rules)} rules triggered.",
                f"Returning top {len(final_proposals)} proposals.",
            ],
        )

    def _llm_enhance(
        self,
        run_state: dict[str, Any],
        toolkit_config: Any,
        existing_proposals: list[ExperimentProposal],
    ) -> list[ExperimentProposal]:
        """Call Claude API for additional proposals.

        Returns [] silently if ANTHROPIC_API_KEY is not set or call fails.
        LLM proposals use source="llm" and cannot override rule engine results.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.debug("ANTHROPIC_API_KEY not set — skipping LLM enhancement.")
            return []

        try:
            import anthropic  # type: ignore
            client = anthropic.Anthropic(api_key=api_key)

            # Build a concise state summary for the prompt
            state_summary = self._build_state_summary(run_state, existing_proposals)

            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "You are an ML experiment advisor. Given the current pipeline state "
                            "and existing proposals, suggest 1-2 additional experiments "
                            "(not already proposed) that could improve model accuracy or reliability.\n\n"
                            f"Current state: {state_summary}\n\n"
                            "Existing proposals:\n"
                            + "\n".join(f"- {p.action}: {p.rationale}" for p in existing_proposals)
                            + "\n\nRespond with JSON array of proposals: "
                            '[{"action": "...", "rationale": "...", "priority": 3}]'
                        ),
                    }
                ],
            )

            import json
            text = message.content[0].text.strip()
            # Extract JSON from response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                proposals_data = json.loads(text[start:end])
                llm_proposals = []
                for p_data in proposals_data:
                    if isinstance(p_data, dict) and "action" in p_data:
                        llm_proposals.append(ExperimentProposal(
                            action=str(p_data.get("action", "")),
                            rationale=str(p_data.get("rationale", "")),
                            priority=int(p_data.get("priority", 3)),
                            source="llm",
                        ))
                return llm_proposals

        except ImportError:
            logger.debug("anthropic package not installed — skipping LLM enhancement.")
        except Exception as e:
            logger.warning(f"LLM enhancement failed (non-fatal): {e}")

        return []

    def _enforce_constraints(
        self,
        proposals: list[ExperimentProposal],
        toolkit_config: Any,
    ) -> list[ExperimentProposal]:
        """Ensure proposals never violate user config constraints.

        Currently enforces:
        - max_candidates in config_patch ≤ user's max_candidates
        - Only interventions from allowed_types are patched
        """
        if toolkit_config is None:
            return proposals

        cleaned = []
        for p in proposals:
            patch = p.config_patch

            # Enforce max_candidates limit
            if "candidates" in patch and "max_candidates" in patch["candidates"]:
                user_max = toolkit_config.candidates.max_candidates
                patched = patch["candidates"]["max_candidates"]
                if patched > user_max:
                    p = p.model_copy(update={
                        "config_patch": {
                            **patch,
                            "candidates": {**patch["candidates"], "max_candidates": user_max},
                        }
                    })

            # Enforce intervention types are in user's allowed list
            if "interventions" in patch and "allowed_types" in patch.get("interventions", {}):
                user_allowed = set(toolkit_config.interventions.allowed_types)
                patched_types = patch["interventions"]["allowed_types"]
                valid_types = [t for t in patched_types if t in user_allowed]
                if not valid_types:
                    logger.debug(f"Proposal '{p.action}' filtered: no allowed intervention types.")
                    continue
                p = p.model_copy(update={
                    "config_patch": {
                        **patch,
                        "interventions": {**patch["interventions"], "allowed_types": valid_types},
                    }
                })

            cleaned.append(p)

        return cleaned

    def _build_state_summary(
        self,
        run_state: dict[str, Any],
        existing_proposals: list[ExperimentProposal],
    ) -> str:
        """Build a concise, safe state summary for LLM prompt."""
        keys = [
            "modality", "n_samples", "n_features", "n_classes",
            "imbalance_severity", "has_label_noise", "has_ood_shift",
            "mean_uncertainty", "calibration_results",
        ]
        summary = {k: run_state.get(k) for k in keys if k in run_state}
        return str(summary)[:1000]  # Cap at 1000 chars for safety
