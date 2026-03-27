"""Custom exception taxonomy for structured error handling across the pipeline."""


class ToolkitError(Exception):
    """Base exception for all toolkit errors."""


class SchemaValidationError(ToolkitError):
    """Raised when the input dataset schema is invalid or incomplete."""


class UnsupportedModalityError(ToolkitError):
    """Raised when the detected or requested modality is not supported."""


class SplitIntegrityError(ToolkitError):
    """Raised when split validation detects a blocking integrity issue."""


class LeakageDetectedError(SplitIntegrityError):
    """Raised when data leakage is detected across splits."""


class ResourceAbstentionError(ToolkitError):
    """Raised when a resource limit (OOM, GPU, time) forces abstention."""

    def __init__(self, message: str, resource_type: str = "unknown") -> None:
        super().__init__(message)
        self.resource_type = resource_type


class CalibrationFailureError(ToolkitError):
    """Raised when calibration cannot be completed for a candidate."""


class ExplainabilityFailureWarning(UserWarning):
    """Warning when an explainability method fails or is unsupported for a model.

    This is a warning, not an exception — explainability failures should degrade
    gracefully, not crash the pipeline.
    """


class AbstentionTriggeredError(ToolkitError):
    """Raised when the pipeline decides to abstain from producing a final model."""

    def __init__(self, message: str, reason: str = "unknown") -> None:
        super().__init__(message)
        self.reason = reason
