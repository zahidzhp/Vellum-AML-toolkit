"""Enumerations for modality, task type, operating mode, decisions, interventions, and risk flags."""

from enum import Enum


class ModalityType(str, Enum):
    """Supported input data modalities."""

    TABULAR = "TABULAR"
    IMAGE = "IMAGE"
    EMBEDDING = "EMBEDDING"


class TaskType(str, Enum):
    """Supported classification task types."""

    BINARY = "BINARY"
    MULTICLASS = "MULTICLASS"
    MULTILABEL = "MULTILABEL"


class OperatingMode(str, Enum):
    """Toolkit operating modes that control policy thresholds."""

    CONSERVATIVE = "CONSERVATIVE"
    BALANCED = "BALANCED"
    AGGRESSIVE = "AGGRESSIVE"
    INTERPRETABLE = "INTERPRETABLE"


class DecisionType(str, Enum):
    """Runtime decision engine outcomes."""

    CONTINUE = "CONTINUE"
    STOP = "STOP"
    EXPAND = "EXPAND"
    ABSTAIN = "ABSTAIN"
    FAIL = "FAIL"


class InterventionType(str, Enum):
    """Available data intervention strategies."""

    CLASS_WEIGHTING = "CLASS_WEIGHTING"
    OVERSAMPLING = "OVERSAMPLING"
    UNDERSAMPLING = "UNDERSAMPLING"
    AUGMENTATION = "AUGMENTATION"
    FOCAL_LOSS = "FOCAL_LOSS"
    THRESHOLDING = "THRESHOLDING"
    CALIBRATION = "CALIBRATION"


class RiskFlag(str, Enum):
    """Risk flags raised during profiling and auditing."""

    LEAKAGE = "LEAKAGE"
    LABEL_NOISE = "LABEL_NOISE"
    CLASS_IMBALANCE = "CLASS_IMBALANCE"
    OOD_SHIFT = "OOD_SHIFT"
    LABEL_CONFLICT = "LABEL_CONFLICT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    CALIBRATION_FAILURE = "CALIBRATION_FAILURE"


class PipelineStage(str, Enum):
    """Pipeline execution stages for state tracking."""

    INIT = "INIT"
    DATA_VALIDATED = "DATA_VALIDATED"
    PROFILED = "PROFILED"
    PROBED = "PROBED"
    INTERVENTION_SELECTED = "INTERVENTION_SELECTED"
    TRAINING_ACTIVE = "TRAINING_ACTIVE"
    MODEL_SELECTED = "MODEL_SELECTED"
    CALIBRATED = "CALIBRATED"
    ENSEMBLED = "ENSEMBLED"
    EXPLAINED = "EXPLAINED"
    COMPLETED = "COMPLETED"
    ABSTAINED = "ABSTAINED"


class AbstentionReason(str, Enum):
    """Typed reasons for pipeline abstention."""

    LEAKAGE_BLOCKED = "LEAKAGE_BLOCKED"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    NO_ROBUST_MODEL = "NO_ROBUST_MODEL"
    CRITICAL_FAILURE = "CRITICAL_FAILURE"


class SplitStrategy(str, Enum):
    """Supported data splitting strategies."""

    STRATIFIED = "STRATIFIED"
    GROUPED = "GROUPED"
    TEMPORAL = "TEMPORAL"
    PROVIDED = "PROVIDED"
