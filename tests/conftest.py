"""Shared test fixtures and helpers for the Autonomous ML Toolkit test suite.

Provides reusable fixtures for synthetic datasets, configs, and common
test data patterns used across unit, integration, and regression tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aml_toolkit.artifacts import (
    CalibrationReport,
    CandidateEntry,
    CandidatePortfolio,
    DataProfile,
    DatasetManifest,
    EnsembleReport,
    ExplainabilityReport,
    InterventionPlan,
    ProbeResultSet,
    RuntimeDecisionLog,
    SplitAuditReport,
)
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import PipelineStage


# ---------------------------------------------------------------------------
# Synthetic tabular datasets
# ---------------------------------------------------------------------------

@pytest.fixture()
def binary_csv(tmp_path: Path) -> Path:
    """Create a balanced binary classification CSV (100 samples, 3 features)."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "f1": rng.randn(100),
        "f2": rng.randn(100),
        "f3": rng.randn(100),
        "label": np.array([0] * 50 + [1] * 50),
    })
    path = tmp_path / "binary.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def imbalanced_csv(tmp_path: Path) -> Path:
    """Create a severely imbalanced binary CSV (95:5 ratio)."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "label": np.array([0] * 190 + [1] * 10),
    })
    path = tmp_path / "imbalanced.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def multiclass_csv(tmp_path: Path) -> Path:
    """Create a 3-class CSV (150 samples)."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "f1": rng.randn(150),
        "f2": rng.randn(150),
        "label": np.array(["cat"] * 50 + ["dog"] * 50 + ["bird"] * 50),
    })
    path = tmp_path / "multiclass.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def grouped_csv(tmp_path: Path) -> Path:
    """Create a CSV with a group column for grouped leakage testing."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "group_id": [f"patient_{i // 4}" for i in range(n)],
        "label": np.array([0] * 100 + [1] * 100),
    })
    path = tmp_path / "grouped.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def temporal_csv(tmp_path: Path) -> Path:
    """Create a CSV with a datetime column for temporal leakage testing."""
    rng = np.random.RandomState(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "date": dates,
        "label": np.array([0] * 50 + [1] * 50),
    })
    path = tmp_path / "temporal.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def duplicate_csv(tmp_path: Path) -> Path:
    """Create a CSV with exact duplicate rows across splits."""
    rng = np.random.RandomState(42)
    base = pd.DataFrame({
        "f1": rng.randn(80),
        "f2": rng.randn(80),
        "label": np.array([0] * 40 + [1] * 40),
    })
    # Duplicate the first 10 rows
    duplicates = base.iloc[:10].copy()
    df = pd.concat([base, duplicates], ignore_index=True)
    path = tmp_path / "duplicates.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# NumPy data arrays
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_train_val():
    """Minimal train/val numpy arrays for model testing."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(60, 4)
    y_train = np.array([0] * 30 + [1] * 30)
    X_val = rng.randn(20, 4)
    y_val = np.array([0] * 10 + [1] * 10)
    return X_train, y_train, X_val, y_val


@pytest.fixture()
def simple_train_val_test():
    """Minimal train/val/test numpy arrays for full pipeline testing."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(60, 4)
    y_train = np.array([0] * 30 + [1] * 30)
    X_val = rng.randn(20, 4)
    y_val = np.array([0] * 10 + [1] * 10)
    X_test = rng.randn(20, 4)
    y_test = np.array([0] * 10 + [1] * 10)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_config() -> ToolkitConfig:
    """Default ToolkitConfig."""
    return ToolkitConfig()


@pytest.fixture()
def minimal_config(tmp_path: Path) -> ToolkitConfig:
    """Minimal config for fast tests — logistic only, small budget."""
    return ToolkitConfig(
        dataset={"path": "", "target_column": "label"},
        reporting={"output_dir": str(tmp_path / "outputs")},
        candidates={"allowed_families": ["logistic"], "max_candidates": 1},
        compute={"max_training_time_seconds": 60},
    )


# ---------------------------------------------------------------------------
# Artifact stubs
# ---------------------------------------------------------------------------

@pytest.fixture()
def passing_audit() -> SplitAuditReport:
    return SplitAuditReport(passed=True)


@pytest.fixture()
def failing_audit() -> SplitAuditReport:
    return SplitAuditReport(
        passed=False,
        blocking_issues=["Duplicate leakage detected between train and test."],
    )


@pytest.fixture()
def complete_artifacts() -> dict:
    """Full set of stage artifacts simulating a completed pipeline run."""
    return {
        "run_id": "test_run_001",
        "final_status": PipelineStage.COMPLETED,
        "stages_completed": [
            PipelineStage.INIT,
            PipelineStage.DATA_VALIDATED,
            PipelineStage.PROFILED,
            PipelineStage.PROBED,
            PipelineStage.INTERVENTION_SELECTED,
            PipelineStage.TRAINING_ACTIVE,
            PipelineStage.MODEL_SELECTED,
            PipelineStage.CALIBRATED,
            PipelineStage.ENSEMBLED,
            PipelineStage.EXPLAINED,
            PipelineStage.COMPLETED,
        ],
        "best_candidate_id": "logistic_001",
        "dataset_manifest": DatasetManifest(
            dataset_id="test", modality="TABULAR", task_type="BINARY",
            split_strategy="STRATIFIED",
        ),
        "split_audit_report": SplitAuditReport(passed=True),
        "data_profile": DataProfile(),
        "probe_results": ProbeResultSet(),
        "intervention_plan": InterventionPlan(),
        "candidate_portfolio": CandidatePortfolio(),
        "runtime_decision_log": RuntimeDecisionLog(),
        "calibration_report": CalibrationReport(),
        "ensemble_report": EnsembleReport(),
        "explainability_report": ExplainabilityReport(),
        "warnings": [],
    }
