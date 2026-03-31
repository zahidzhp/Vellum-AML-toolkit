"""Training executor: controlled serial execution of candidate models with audit gating and resource guard."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aml_toolkit.artifacts import CandidatePortfolio, InterventionPlan, SplitAuditReport
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import InterventionType
from aml_toolkit.core.exceptions import ResourceAbstentionError, SplitIntegrityError
from aml_toolkit.interfaces.candidate_model import CandidateModel
from aml_toolkit.interfaces.model_metadata import ModelRegistry
from aml_toolkit.utils.resource_guard import ResourceGuard

logger = logging.getLogger("aml_toolkit")


@dataclass
class CandidateExecutionTrace:
    """Execution trace for a single candidate."""

    candidate_id: str
    model_family: str
    backbone: str | None = None
    status: str = "pending"  # pending, completed, failed, abstained
    metrics: dict[str, float] = field(default_factory=dict)
    training_trace: dict[str, list[float]] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error_message: str | None = None


@dataclass
class ExecutionResult:
    """Result of the full training execution pass."""

    traces: list[CandidateExecutionTrace] = field(default_factory=list)
    trained_models: dict[str, CandidateModel] = field(default_factory=dict)
    abstentions: list[str] = field(default_factory=list)


class TrainingLifecycleHook(ABC):
    """Hook interface for observing training lifecycle events.

    Implement any subset of methods to observe execution.
    All methods have no-op defaults so subclasses only override what they need.
    """

    def on_execution_start(self, portfolio: CandidatePortfolio, config: ToolkitConfig) -> None:
        """Called once before the training loop begins."""

    def on_candidate_start(self, candidate_id: str, model_family: str) -> None:
        """Called before each candidate begins training."""

    def on_candidate_end(self, trace: CandidateExecutionTrace) -> None:
        """Called after each candidate finishes (success, failure, or abstention)."""

    def on_execution_end(self, result: ExecutionResult) -> None:
        """Called once after all candidates have been processed."""


def run_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    portfolio: CandidatePortfolio,
    audit_report: SplitAuditReport,
    config: ToolkitConfig,
    registry: ModelRegistry | None = None,
    intervention_plan: InterventionPlan | None = None,
    hooks: list[TrainingLifecycleHook] | None = None,
    raw_data: dict[str, Any] | None = None,
) -> ExecutionResult:
    """Execute training for all candidates in the portfolio serially.

    Steps per candidate:
    1. Check audit gate (refuse if audit has blocking issues).
    2. Instantiate adapter from registry.
    3. Apply class_weight if intervention plan requires it.
    4. Train within resource guard.
    5. Evaluate on validation set.
    6. Record trace.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        portfolio: CandidatePortfolio from Phase 8.
        audit_report: SplitAuditReport (must pass for training to proceed).
        config: Toolkit configuration.
        registry: Model registry. If None, uses default.
        intervention_plan: Optional intervention plan for class_weight.
        hooks: Optional lifecycle hooks called during execution.

    Returns:
        ExecutionResult with traces and trained models.

    Raises:
        SplitIntegrityError: If audit has blocking issues.
    """
    hooks = hooks or []

    # Audit gate
    if not audit_report.passed:
        raise SplitIntegrityError(
            "Training executor refused: split audit has blocking issues. "
            f"Blocking issues: {audit_report.blocking_issues}"
        )

    if registry is None:
        from aml_toolkit.models.registry import create_default_registry
        registry = create_default_registry()

    resource_guard = ResourceGuard(config)
    resource_guard.start_timer()

    # Determine if class_weight should be used
    use_class_weight = _should_use_class_weight(intervention_plan)

    result = ExecutionResult()

    for hook in hooks:
        hook.on_execution_start(portfolio, config)

    for candidate in portfolio.candidate_models:
        trace = CandidateExecutionTrace(
            candidate_id=candidate.candidate_id,
            model_family=candidate.model_family,
        )

        logger.info(f"Training candidate: {candidate.candidate_id} ({candidate.model_family})")

        for hook in hooks:
            hook.on_candidate_start(candidate.candidate_id, candidate.model_family)

        try:
            # Check global time budget
            resource_guard.check_time_budget(candidate.candidate_id)

            # Instantiate adapter
            adapter_class = registry.get_adapter(candidate.model_family)
            metadata = registry.get_metadata(candidate.model_family)

            # Pass class_weight if applicable and adapter supports it
            init_kwargs: dict[str, Any] = {"seed": config.seed}
            if use_class_weight and not metadata.is_neural:
                init_kwargs["class_weight"] = "balanced"

            try:
                adapter = adapter_class(**init_kwargs)
            except TypeError:
                # Adapter may not accept class_weight
                adapter = adapter_class(seed=config.seed)

            # Determine X/y for this candidate — image-native models use raw paths
            from aml_toolkit.core.enums import ModalityType

            fit_X_train, fit_y_train = X_train, y_train
            fit_X_val, fit_y_val = X_val, y_val
            if (
                raw_data
                and "image_paths_train" in raw_data
                and ModalityType.IMAGE in metadata.supported_modalities
            ):
                fit_X_train = raw_data["image_paths_train"]
                fit_X_val = raw_data["image_paths_val"]

            # Train within resource guard
            with resource_guard.guarded_execution(candidate.candidate_id):
                adapter.fit(fit_X_train, fit_y_train, fit_X_val, fit_y_val, config)

            # Evaluate
            eval_metrics = adapter.evaluate(
                fit_X_val, fit_y_val, ["macro_f1", "accuracy"]
            )

            trace.status = "completed"
            trace.metrics = eval_metrics
            trace.training_trace = adapter.get_training_trace()
            trace.backbone = adapter.get_backbone()
            trace.elapsed_seconds = resource_guard.elapsed_seconds()

            result.trained_models[candidate.candidate_id] = adapter

            logger.info(
                f"Candidate {candidate.candidate_id} completed: "
                f"val metrics = {eval_metrics}"
            )

        except ResourceAbstentionError as e:
            trace.status = "abstained"
            trace.error_message = str(e)
            trace.elapsed_seconds = resource_guard.elapsed_seconds()
            result.abstentions.append(candidate.candidate_id)
            logger.warning(
                f"Candidate {candidate.candidate_id} abstained: {e}"
            )

        except NotImplementedError as e:
            trace.status = "failed"
            trace.error_message = f"Not implemented: {e}"
            trace.elapsed_seconds = resource_guard.elapsed_seconds()
            logger.warning(
                f"Candidate {candidate.candidate_id} not implemented: {e}"
            )

        except Exception as e:
            trace.status = "failed"
            trace.error_message = str(e)
            trace.elapsed_seconds = resource_guard.elapsed_seconds()
            logger.error(
                f"Candidate {candidate.candidate_id} failed: {e}"
            )

        for hook in hooks:
            hook.on_candidate_end(trace)

        result.traces.append(trace)

    for hook in hooks:
        hook.on_execution_end(result)

    return result


def _should_use_class_weight(intervention_plan: InterventionPlan | None) -> bool:
    """Check if the intervention plan includes class weighting."""
    if intervention_plan is None:
        return False
    return any(
        e.intervention_type == InterventionType.CLASS_WEIGHTING and e.selected
        for e in intervention_plan.selected_interventions
    )
