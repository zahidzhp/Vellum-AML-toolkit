"""Tests for Phase 9: Training Executor and Resource Guard.

Required tests:
1. Executor runs a minimal candidate.
2. Audit-blocked execution test.
3. Resource abstention test (OOM + time budget).
4. Execution trace generation test.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aml_toolkit.artifacts import (
    CandidateEntry,
    CandidatePortfolio,
    InterventionEntry,
    InterventionPlan,
    SplitAuditReport,
)
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import InterventionType
from aml_toolkit.core.exceptions import ResourceAbstentionError, SplitIntegrityError
from aml_toolkit.runtime.training_executor import (
    CandidateExecutionTrace,
    ExecutionResult,
    TrainingLifecycleHook,
    run_training,
)
from aml_toolkit.utils.resource_guard import ResourceGuard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_data():
    """Create minimal train/val data for testing."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(60, 4)
    y_train = np.array([0] * 30 + [1] * 30)
    X_val = rng.randn(20, 4)
    y_val = np.array([0] * 10 + [1] * 10)
    return X_train, y_train, X_val, y_val


@pytest.fixture()
def passing_audit():
    return SplitAuditReport(passed=True)


@pytest.fixture()
def failing_audit():
    return SplitAuditReport(
        passed=False,
        blocking_issues=["Duplicate leakage detected between train and test."],
    )


@pytest.fixture()
def single_logistic_portfolio():
    return CandidatePortfolio(
        candidate_models=[
            CandidateEntry(
                candidate_id="logistic_001",
                model_family="logistic",
                model_name="Logistic Regression",
                warmup_epochs=1,
                budget_allocation=1.0,
            )
        ],
        selected_families=["logistic"],
        budget_allocations={"logistic": 1.0},
        warmup_rules={"logistic": 1},
    )


@pytest.fixture()
def multi_candidate_portfolio():
    return CandidatePortfolio(
        candidate_models=[
            CandidateEntry(
                candidate_id="logistic_001",
                model_family="logistic",
                model_name="Logistic Regression",
                warmup_epochs=1,
                budget_allocation=0.5,
            ),
            CandidateEntry(
                candidate_id="rf_001",
                model_family="rf",
                model_name="Random Forest",
                warmup_epochs=1,
                budget_allocation=0.5,
            ),
        ],
        selected_families=["logistic", "rf"],
        budget_allocations={"logistic": 0.5, "rf": 0.5},
        warmup_rules={"logistic": 1, "rf": 1},
    )


@pytest.fixture()
def config():
    return ToolkitConfig()


# ---------------------------------------------------------------------------
# Test 1: Executor runs a minimal candidate
# ---------------------------------------------------------------------------

class TestMinimalCandidateExecution:
    """Verify executor can train a single candidate end-to-end."""

    def test_single_candidate_completes(self, simple_data, passing_audit, single_logistic_portfolio, config):
        X_train, y_train, X_val, y_val = simple_data
        result = run_training(
            X_train, y_train, X_val, y_val,
            portfolio=single_logistic_portfolio,
            audit_report=passing_audit,
            config=config,
        )

        assert isinstance(result, ExecutionResult)
        assert len(result.traces) == 1
        assert result.traces[0].status == "completed"
        assert "logistic_001" in result.trained_models
        assert len(result.abstentions) == 0

    def test_multiple_candidates_complete(self, simple_data, passing_audit, multi_candidate_portfolio, config):
        X_train, y_train, X_val, y_val = simple_data
        result = run_training(
            X_train, y_train, X_val, y_val,
            portfolio=multi_candidate_portfolio,
            audit_report=passing_audit,
            config=config,
        )

        assert len(result.traces) == 2
        completed = [t for t in result.traces if t.status == "completed"]
        assert len(completed) == 2
        assert "logistic_001" in result.trained_models
        assert "rf_001" in result.trained_models

    def test_class_weight_applied_when_intervention_plan_requests(
        self, simple_data, passing_audit, single_logistic_portfolio, config
    ):
        X_train, y_train, X_val, y_val = simple_data
        plan = InterventionPlan(
            selected_interventions=[
                InterventionEntry(
                    intervention_type=InterventionType.CLASS_WEIGHTING,
                    selected=True,
                    rationale="Imbalanced data.",
                )
            ]
        )
        result = run_training(
            X_train, y_train, X_val, y_val,
            portfolio=single_logistic_portfolio,
            audit_report=passing_audit,
            config=config,
            intervention_plan=plan,
        )
        assert result.traces[0].status == "completed"


# ---------------------------------------------------------------------------
# Test 2: Audit-blocked execution
# ---------------------------------------------------------------------------

class TestAuditGate:
    """Verify executor refuses to run when audit has blocking issues."""

    def test_raises_on_failed_audit(self, simple_data, failing_audit, single_logistic_portfolio, config):
        X_train, y_train, X_val, y_val = simple_data
        with pytest.raises(SplitIntegrityError, match="split audit has blocking issues"):
            run_training(
                X_train, y_train, X_val, y_val,
                portfolio=single_logistic_portfolio,
                audit_report=failing_audit,
                config=config,
            )

    def test_error_message_includes_blocking_issues(self, simple_data, failing_audit, single_logistic_portfolio, config):
        X_train, y_train, X_val, y_val = simple_data
        with pytest.raises(SplitIntegrityError) as exc_info:
            run_training(
                X_train, y_train, X_val, y_val,
                portfolio=single_logistic_portfolio,
                audit_report=failing_audit,
                config=config,
            )
        assert "Duplicate leakage" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 3: Resource abstention (OOM + time budget)
# ---------------------------------------------------------------------------

class TestResourceAbstention:
    """Verify that OOM and time budget violations produce structured abstention."""

    def test_oom_produces_abstention(self, simple_data, passing_audit, single_logistic_portfolio, config):
        """Simulate MemoryError during training -> abstention trace."""
        X_train, y_train, X_val, y_val = simple_data

        with patch(
            "aml_toolkit.runtime.training_executor.ResourceGuard"
        ) as MockGuard:
            guard_instance = MagicMock()
            MockGuard.return_value = guard_instance
            guard_instance.elapsed_seconds.return_value = 1.0

            # guarded_execution raises ResourceAbstentionError (simulating OOM conversion)
            guard_instance.guarded_execution.return_value.__enter__ = MagicMock(
                side_effect=ResourceAbstentionError(
                    "Out of memory during training of logistic_001.",
                    resource_type="memory",
                )
            )
            guard_instance.guarded_execution.return_value.__exit__ = MagicMock(return_value=False)

            result = run_training(
                X_train, y_train, X_val, y_val,
                portfolio=single_logistic_portfolio,
                audit_report=passing_audit,
                config=config,
            )

        assert len(result.traces) == 1
        assert result.traces[0].status == "abstained"
        assert "logistic_001" in result.abstentions
        assert result.traces[0].error_message is not None
        assert "Out of memory" in result.traces[0].error_message

    def test_time_budget_exceeded_produces_abstention(
        self, simple_data, passing_audit, single_logistic_portfolio, config
    ):
        """Simulate time budget exceeded -> abstention trace."""
        X_train, y_train, X_val, y_val = simple_data

        with patch(
            "aml_toolkit.runtime.training_executor.ResourceGuard"
        ) as MockGuard:
            guard_instance = MagicMock()
            MockGuard.return_value = guard_instance
            guard_instance.elapsed_seconds.return_value = 9999.0

            # check_time_budget raises ResourceAbstentionError
            guard_instance.check_time_budget.side_effect = ResourceAbstentionError(
                "Training time budget exceeded for logistic_001: 9999.0s > 3600s limit.",
                resource_type="time",
            )

            result = run_training(
                X_train, y_train, X_val, y_val,
                portfolio=single_logistic_portfolio,
                audit_report=passing_audit,
                config=config,
            )

        assert result.traces[0].status == "abstained"
        assert "logistic_001" in result.abstentions
        assert "time budget" in result.traces[0].error_message.lower()

    def test_resource_guard_catches_memory_error(self):
        """Unit test: ResourceGuard converts MemoryError to ResourceAbstentionError."""
        config = ToolkitConfig()
        guard = ResourceGuard(config)
        guard.start_timer()

        with pytest.raises(ResourceAbstentionError) as exc_info:
            with guard.guarded_execution("test_candidate"):
                raise MemoryError("simulated OOM")

        assert exc_info.value.resource_type == "memory"

    def test_resource_guard_catches_cuda_oom(self):
        """Unit test: ResourceGuard converts CUDA OOM RuntimeError to ResourceAbstentionError."""
        config = ToolkitConfig()
        guard = ResourceGuard(config)
        guard.start_timer()

        with pytest.raises(ResourceAbstentionError) as exc_info:
            with guard.guarded_execution("test_candidate"):
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        assert exc_info.value.resource_type == "gpu_memory"

    def test_resource_guard_reraises_non_oom_runtime_error(self):
        """Non-OOM RuntimeError should propagate unchanged."""
        config = ToolkitConfig()
        guard = ResourceGuard(config)
        guard.start_timer()

        with pytest.raises(RuntimeError, match="something else"):
            with guard.guarded_execution("test_candidate"):
                raise RuntimeError("something else entirely")

    def test_resource_guard_time_budget_check(self):
        """Unit test: check_time_budget raises after exceeding limit."""
        config = ToolkitConfig(compute={"max_training_time_seconds": 0})
        guard = ResourceGuard(config)
        guard.start_timer()

        with pytest.raises(ResourceAbstentionError) as exc_info:
            guard.check_time_budget("test_candidate")

        assert exc_info.value.resource_type == "time"


# ---------------------------------------------------------------------------
# Test 4: Execution trace generation
# ---------------------------------------------------------------------------

class TestExecutionTraceGeneration:
    """Verify execution traces contain the required fields."""

    def test_trace_fields_populated_on_success(self, simple_data, passing_audit, single_logistic_portfolio, config):
        X_train, y_train, X_val, y_val = simple_data
        result = run_training(
            X_train, y_train, X_val, y_val,
            portfolio=single_logistic_portfolio,
            audit_report=passing_audit,
            config=config,
        )

        trace = result.traces[0]
        assert trace.candidate_id == "logistic_001"
        assert trace.model_family == "logistic"
        assert trace.status == "completed"
        assert "macro_f1" in trace.metrics
        assert "accuracy" in trace.metrics
        assert trace.elapsed_seconds > 0.0
        assert trace.error_message is None
        assert isinstance(trace.training_trace, dict)

    def test_trace_fields_populated_on_failure(self, simple_data, passing_audit, config):
        """Force a failure by registering a broken adapter."""
        from aml_toolkit.interfaces.model_metadata import ModelFamilyMetadata, ModelRegistry

        class BrokenAdapter:
            def __init__(self, seed=42):
                pass

            def fit(self, *args, **kwargs):
                raise ValueError("intentional training failure")

        registry = ModelRegistry()
        registry.register(
            "broken",
            BrokenAdapter,
            ModelFamilyMetadata(
                family_name="broken",
                display_name="Broken",
                supported_modalities=[],
                is_neural=False,
            ),
        )

        portfolio = CandidatePortfolio(
            candidate_models=[
                CandidateEntry(
                    candidate_id="broken_001",
                    model_family="broken",
                    model_name="Broken",
                )
            ],
            selected_families=["broken"],
        )

        X_train, y_train, X_val, y_val = simple_data
        result = run_training(
            X_train, y_train, X_val, y_val,
            portfolio=portfolio,
            audit_report=SplitAuditReport(passed=True),
            config=config,
            registry=registry,
        )

        trace = result.traces[0]
        assert trace.status == "failed"
        assert trace.error_message is not None
        assert "intentional training failure" in trace.error_message
        assert trace.elapsed_seconds >= 0.0

    def test_multi_candidate_traces_in_order(self, simple_data, passing_audit, multi_candidate_portfolio, config):
        X_train, y_train, X_val, y_val = simple_data
        result = run_training(
            X_train, y_train, X_val, y_val,
            portfolio=multi_candidate_portfolio,
            audit_report=passing_audit,
            config=config,
        )

        assert len(result.traces) == 2
        assert result.traces[0].candidate_id == "logistic_001"
        assert result.traces[1].candidate_id == "rf_001"


# ---------------------------------------------------------------------------
# Test 5: Lifecycle hooks
# ---------------------------------------------------------------------------

class TestLifecycleHooks:
    """Verify lifecycle hooks are called at the correct points."""

    def test_hooks_called_in_order(self, simple_data, passing_audit, single_logistic_portfolio, config):
        calls: list[str] = []

        class RecordingHook(TrainingLifecycleHook):
            def on_execution_start(self, portfolio, cfg):
                calls.append("execution_start")

            def on_candidate_start(self, candidate_id, model_family):
                calls.append(f"candidate_start:{candidate_id}")

            def on_candidate_end(self, trace):
                calls.append(f"candidate_end:{trace.candidate_id}:{trace.status}")

            def on_execution_end(self, result):
                calls.append("execution_end")

        X_train, y_train, X_val, y_val = simple_data
        run_training(
            X_train, y_train, X_val, y_val,
            portfolio=single_logistic_portfolio,
            audit_report=passing_audit,
            config=config,
            hooks=[RecordingHook()],
        )

        assert calls == [
            "execution_start",
            "candidate_start:logistic_001",
            "candidate_end:logistic_001:completed",
            "execution_end",
        ]

    def test_hooks_called_on_abstention(self, simple_data, passing_audit, single_logistic_portfolio, config):
        calls: list[str] = []

        class RecordingHook(TrainingLifecycleHook):
            def on_candidate_end(self, trace):
                calls.append(f"candidate_end:{trace.status}")

        X_train, y_train, X_val, y_val = simple_data

        with patch(
            "aml_toolkit.runtime.training_executor.ResourceGuard"
        ) as MockGuard:
            guard_instance = MagicMock()
            MockGuard.return_value = guard_instance
            guard_instance.elapsed_seconds.return_value = 1.0
            guard_instance.check_time_budget.side_effect = ResourceAbstentionError(
                "time exceeded", resource_type="time",
            )

            run_training(
                X_train, y_train, X_val, y_val,
                portfolio=single_logistic_portfolio,
                audit_report=passing_audit,
                config=config,
                hooks=[RecordingHook()],
            )

        assert calls == ["candidate_end:abstained"]
