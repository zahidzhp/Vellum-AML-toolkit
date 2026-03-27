"""Smoke tests for all enum types."""

from aml_toolkit.core.enums import (
    AbstentionReason,
    DecisionType,
    InterventionType,
    ModalityType,
    OperatingMode,
    PipelineStage,
    RiskFlag,
    SplitStrategy,
    TaskType,
)


def test_modality_type_members():
    assert ModalityType.TABULAR.value == "TABULAR"
    assert ModalityType.IMAGE.value == "IMAGE"
    assert ModalityType.EMBEDDING.value == "EMBEDDING"
    assert len(ModalityType) == 3


def test_task_type_members():
    assert TaskType.BINARY.value == "BINARY"
    assert TaskType.MULTICLASS.value == "MULTICLASS"
    assert TaskType.MULTILABEL.value == "MULTILABEL"
    assert len(TaskType) == 3


def test_operating_mode_members():
    assert len(OperatingMode) == 4
    assert OperatingMode.CONSERVATIVE.value == "CONSERVATIVE"


def test_decision_type_members():
    assert len(DecisionType) == 5
    assert DecisionType.ABSTAIN.value == "ABSTAIN"


def test_intervention_type_members():
    assert len(InterventionType) == 7
    assert InterventionType.FOCAL_LOSS.value == "FOCAL_LOSS"


def test_risk_flag_members():
    assert len(RiskFlag) == 7
    assert RiskFlag.LEAKAGE.value == "LEAKAGE"


def test_pipeline_stage_members():
    assert PipelineStage.INIT.value == "INIT"
    assert PipelineStage.EXPLAINED.value == "EXPLAINED"
    assert PipelineStage.COMPLETED.value == "COMPLETED"
    assert PipelineStage.ABSTAINED.value == "ABSTAINED"
    assert len(PipelineStage) == 12


def test_abstention_reason_members():
    assert len(AbstentionReason) == 5
    assert AbstentionReason.RESOURCE_EXHAUSTED.value == "RESOURCE_EXHAUSTED"


def test_split_strategy_members():
    assert len(SplitStrategy) == 4
    assert SplitStrategy.TEMPORAL.value == "TEMPORAL"


def test_enums_are_string_enums():
    """All enums should be str enums for JSON serialization."""
    assert isinstance(ModalityType.TABULAR, str)
    assert isinstance(TaskType.BINARY, str)
    assert isinstance(DecisionType.CONTINUE, str)
