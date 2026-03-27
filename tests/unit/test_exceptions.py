"""Smoke tests for custom exception hierarchy."""

import warnings

import pytest

from aml_toolkit.core.exceptions import (
    AbstentionTriggeredError,
    CalibrationFailureError,
    ExplainabilityFailureWarning,
    LeakageDetectedError,
    ResourceAbstentionError,
    SchemaValidationError,
    SplitIntegrityError,
    ToolkitError,
    UnsupportedModalityError,
)


def test_toolkit_error_is_base():
    with pytest.raises(ToolkitError):
        raise ToolkitError("base error")


def test_schema_validation_error():
    with pytest.raises(SchemaValidationError):
        raise SchemaValidationError("missing target column")


def test_unsupported_modality_error():
    with pytest.raises(UnsupportedModalityError):
        raise UnsupportedModalityError("audio not supported")


def test_split_integrity_error():
    with pytest.raises(SplitIntegrityError):
        raise SplitIntegrityError("split is invalid")


def test_leakage_detected_inherits_split_integrity():
    with pytest.raises(SplitIntegrityError):
        raise LeakageDetectedError("duplicate overlap found")


def test_resource_abstention_error_has_resource_type():
    err = ResourceAbstentionError("OOM during training", resource_type="gpu_memory")
    assert err.resource_type == "gpu_memory"
    assert "OOM" in str(err)


def test_calibration_failure_error():
    with pytest.raises(CalibrationFailureError):
        raise CalibrationFailureError("temperature scaling failed")


def test_explainability_failure_is_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.warn("SHAP unavailable", ExplainabilityFailureWarning)
        assert len(w) == 1
        assert issubclass(w[0].category, ExplainabilityFailureWarning)


def test_abstention_triggered_error_has_reason():
    err = AbstentionTriggeredError("no robust model", reason="NO_ROBUST_MODEL")
    assert err.reason == "NO_ROBUST_MODEL"


def test_all_exceptions_inherit_from_toolkit_error():
    for exc_class in [
        SchemaValidationError,
        UnsupportedModalityError,
        SplitIntegrityError,
        LeakageDetectedError,
        ResourceAbstentionError,
        CalibrationFailureError,
        AbstentionTriggeredError,
    ]:
        assert issubclass(exc_class, ToolkitError)
