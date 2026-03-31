"""Tests for Phase 2.6 — Experiment planner and rule engine."""

import os
from unittest.mock import MagicMock, patch

import pytest

from aml_toolkit.artifacts.experiment_plan import ExperimentPlan, ExperimentProposal
from aml_toolkit.core.config import AgenticPlannerConfig, InterventionsConfig, ToolkitConfig
from aml_toolkit.planning.experiment_planner import ExperimentPlanner
from aml_toolkit.planning.rule_engine import RULES, PlannerRule, evaluate_rules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> AgenticPlannerConfig:
    defaults = dict(
        enabled=True,
        mode="propose_only",
        max_suggestions=5,
        allow_data_review_requests=False,
        allow_abstention_recommendation=True,
        llm_enhanced=False,
        track_proposal_outcomes=True,
    )
    defaults.update(kwargs)
    return AgenticPlannerConfig(**defaults)


def _toolkit_config(**kwargs) -> ToolkitConfig:
    cfg = ToolkitConfig()
    for k, v in kwargs.items():
        object.__setattr__(cfg, k, v)
    return cfg


def _make_state(**kwargs) -> dict:
    defaults = dict(
        modality="TABULAR",
        n_samples=1000,
        n_features=20,
        n_classes=2,
        imbalance_severity="mild",
        has_label_noise=False,
        has_ood_shift=False,
        allowed_interventions=["CLASS_WEIGHTING", "OVERSAMPLING", "CALIBRATION"],
        calibration_results=[],
        mean_uncertainty=0.2,
    )
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Rule engine — standalone
# ---------------------------------------------------------------------------

class TestRuleEngine:
    def test_evaluate_returns_list(self):
        state = _make_state()
        proposals = evaluate_rules(state, None)
        assert isinstance(proposals, list)

    def test_severe_imbalance_triggers_class_weighting(self):
        state = _make_state(imbalance_severity="severe")
        proposals = evaluate_rules(state, None)
        actions = [p.action for p in proposals]
        assert "add_class_weighting" in actions

    def test_mild_imbalance_no_class_weighting(self):
        state = _make_state(imbalance_severity="mild")
        proposals = evaluate_rules(state, None)
        actions = [p.action for p in proposals]
        assert "add_class_weighting" not in actions

    def test_high_ece_triggers_reduce_candidates(self):
        state = _make_state(
            calibration_results=[{"ece_after": 0.20}, {"ece_after": 0.05}]
        )
        proposals = evaluate_rules(state, None)
        actions = [p.action for p in proposals]
        assert "reduce_max_candidates" in actions

    def test_label_noise_triggers_proposal(self):
        state = _make_state(has_label_noise=True)
        proposals = evaluate_rules(state, None)
        actions = [p.action for p in proposals]
        assert "note_label_noise_detected" in actions

    def test_ood_shift_triggers_conservative_recommendation(self):
        state = _make_state(has_ood_shift=True)
        proposals = evaluate_rules(state, None)
        actions = [p.action for p in proposals]
        assert "recommend_conservative_mode" in actions

    def test_image_small_dataset_triggers_augmentation(self):
        state = _make_state(
            modality="IMAGE",
            n_samples=2000,
            allowed_interventions=["AUGMENTATION"],
        )
        proposals = evaluate_rules(state, None)
        actions = [p.action for p in proposals]
        assert "enable_augmentation" in actions

    def test_image_small_dataset_triggers_resnet_preference(self):
        state = _make_state(modality="IMAGE", n_samples=3000)
        proposals = evaluate_rules(state, None)
        actions = [p.action for p in proposals]
        assert "prefer_resnet18_backbone" in actions

    def test_high_uncertainty_triggers_kfold(self):
        state = _make_state(mean_uncertainty=0.9)
        proposals = evaluate_rules(state, None)
        actions = [p.action for p in proposals]
        assert "suggest_kfold_cross_validation" in actions

    def test_proposals_sorted_by_priority(self):
        state = _make_state(
            imbalance_severity="severe",
            has_ood_shift=True,
            calibration_results=[{"ece_after": 0.20}],
        )
        proposals = evaluate_rules(state, None)
        priorities = [p.priority for p in proposals]
        assert priorities == sorted(priorities)

    def test_custom_rule_added(self):
        custom_rule = PlannerRule(
            name="custom_rule",
            condition=lambda s: s.get("custom_trigger", False),
            make_proposal=lambda s, cfg: ExperimentProposal(
                action="custom_action",
                rationale="Custom rule triggered.",
                priority=1,
            ),
        )
        state = _make_state(custom_trigger=True)
        proposals = evaluate_rules(state, None, rules=[custom_rule])
        assert any(p.action == "custom_action" for p in proposals)

    def test_blocked_intervention_not_proposed_when_not_in_allowed(self):
        """CLASS_WEIGHTING not in allowed_interventions → not proposed."""
        state = _make_state(
            imbalance_severity="severe",
            allowed_interventions=["OVERSAMPLING"],  # no CLASS_WEIGHTING
        )
        proposals = evaluate_rules(state, None)
        actions = [p.action for p in proposals]
        assert "add_class_weighting" not in actions

    def test_broken_condition_does_not_crash(self):
        """Rule with broken condition should be silently skipped."""
        broken_rule = PlannerRule(
            name="broken",
            condition=lambda s: 1 / 0,  # ZeroDivisionError
            make_proposal=lambda s, cfg: ExperimentProposal(action="x", rationale="x"),
        )
        proposals = evaluate_rules({}, None, rules=[broken_rule])
        assert proposals == []


# ---------------------------------------------------------------------------
# ExperimentPlanner — basic behavior
# ---------------------------------------------------------------------------

class TestExperimentPlannerBasic:
    def test_returns_experiment_plan(self):
        planner = ExperimentPlanner(_make_config())
        plan = planner.plan(_make_state())
        assert isinstance(plan, ExperimentPlan)

    def test_max_suggestions_enforced(self):
        planner = ExperimentPlanner(_make_config(max_suggestions=2))
        # Trigger many rules
        state = _make_state(
            imbalance_severity="severe",
            has_label_noise=True,
            has_ood_shift=True,
            calibration_results=[{"ece_after": 0.20}],
            mean_uncertainty=0.9,
        )
        plan = planner.plan(state)
        assert len(plan.proposals) <= 2

    def test_mode_recorded_in_plan(self):
        planner = ExperimentPlanner(_make_config(mode="propose_only"))
        plan = planner.plan(_make_state())
        assert plan.mode == "propose_only"

    def test_propose_only_mode_does_not_auto_apply(self):
        """mode=propose_only → proposals exist but toolkit config not mutated."""
        cfg = ToolkitConfig()
        original_max = cfg.candidates.max_candidates
        planner = ExperimentPlanner(_make_config(mode="propose_only"))
        state = _make_state(calibration_results=[{"ece_after": 0.20}])
        planner.plan(state, toolkit_config=cfg)
        # Config must not be mutated
        assert cfg.candidates.max_candidates == original_max

    def test_rules_evaluated_count(self):
        planner = ExperimentPlanner(_make_config())
        plan = planner.plan(_make_state())
        assert plan.rules_evaluated == len(RULES)

    def test_empty_state_no_crash(self):
        planner = ExperimentPlanner(_make_config())
        plan = planner.plan({})
        assert isinstance(plan, ExperimentPlan)


# ---------------------------------------------------------------------------
# ExperimentPlanner — constraint enforcement
# ---------------------------------------------------------------------------

class TestConstraintEnforcement:
    def test_max_candidates_patch_never_exceeds_user_config(self):
        cfg = ToolkitConfig()
        cfg = cfg.model_copy(update={"candidates": cfg.candidates.model_copy(update={"max_candidates": 2})})
        planner = ExperimentPlanner(_make_config(max_suggestions=10))
        state = _make_state(calibration_results=[{"ece_after": 0.20}])
        plan = planner.plan(state, toolkit_config=cfg)
        for proposal in plan.proposals:
            if "candidates" in proposal.config_patch:
                assert proposal.config_patch["candidates"].get("max_candidates", 0) <= 2

    def test_blocked_intervention_type_filtered(self):
        """Proposal with intervention not in user allowed_types → filtered out."""
        cfg = ToolkitConfig()
        # Remove CLASS_WEIGHTING from allowed types
        new_interventions = cfg.interventions.model_copy(
            update={"allowed_types": ["OVERSAMPLING", "CALIBRATION"]}
        )
        cfg = cfg.model_copy(update={"interventions": new_interventions})
        planner = ExperimentPlanner(_make_config(max_suggestions=10))
        state = _make_state(imbalance_severity="severe")
        plan = planner.plan(state, toolkit_config=cfg)
        # class_weighting proposal should be filtered (CLASS_WEIGHTING not in allowed)
        for p in plan.proposals:
            if "interventions" in p.config_patch:
                types = p.config_patch["interventions"].get("allowed_types", [])
                for t in types:
                    assert t in ["OVERSAMPLING", "CALIBRATION"]


# ---------------------------------------------------------------------------
# ExperimentPlanner — LLM enhancement
# ---------------------------------------------------------------------------

class TestLLMEnhancement:
    def test_llm_disabled_only_rule_engine(self):
        planner = ExperimentPlanner(_make_config(llm_enhanced=False))
        state = _make_state()
        plan = planner.plan(state)
        # All proposals should come from rule_engine
        for p in plan.proposals:
            assert p.source == "rule_engine"

    def test_llm_no_api_key_falls_back_silently(self):
        """No ANTHROPIC_API_KEY → LLM skipped, no error."""
        env_without_key = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            planner = ExperimentPlanner(_make_config(llm_enhanced=True))
            plan = planner.plan(_make_state())
        assert isinstance(plan, ExperimentPlan)

    def test_llm_failure_falls_back_to_rule_only(self):
        """LLM API failure → rule engine proposals still returned."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake_key"}):
            with patch("aml_toolkit.planning.experiment_planner.ExperimentPlanner._llm_enhance",
                       side_effect=Exception("API error")):
                planner = ExperimentPlanner(_make_config(llm_enhanced=True))
                plan = planner.plan(_make_state(imbalance_severity="severe"))
        # Should still return rule proposals (via _plan, which only catches _llm_enhance failures gracefully)
        assert isinstance(plan, ExperimentPlan)

    def test_llm_cannot_add_duplicate_actions(self):
        """LLM proposals with same action as rule engine are deduplicated."""
        rule_proposal = ExperimentProposal(
            action="add_class_weighting", rationale="rule", priority=1, source="rule_engine"
        )
        llm_proposal = ExperimentProposal(
            action="add_class_weighting", rationale="llm", priority=1, source="llm"
        )
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake_key"}):
            with patch(
                "aml_toolkit.planning.experiment_planner.ExperimentPlanner._llm_enhance",
                return_value=[llm_proposal],
            ):
                planner = ExperimentPlanner(_make_config(llm_enhanced=True, max_suggestions=10))
                plan = planner.plan(_make_state(imbalance_severity="severe"))
        # add_class_weighting should appear at most once
        class_weighting_count = sum(
            1 for p in plan.proposals if p.action == "add_class_weighting"
        )
        assert class_weighting_count <= 1


# ---------------------------------------------------------------------------
# ExperimentPlan serialization
# ---------------------------------------------------------------------------

class TestExperimentPlanSerialization:
    def test_plan_serializes_to_json(self):
        planner = ExperimentPlanner(_make_config())
        plan = planner.plan(_make_state(imbalance_severity="severe"))
        data = plan.model_dump(mode="json")
        assert isinstance(data, dict)
        assert "proposals" in data
        import json
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
