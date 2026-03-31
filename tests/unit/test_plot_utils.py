"""Tests for reporting/plot_utils.py — all plot functions."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from aml_toolkit.reporting.plot_utils import (
    plot_calibration_diagram,
    plot_classification_report,
    plot_feature_importance,
    plot_learning_curves,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_vs_metric,
)


@pytest.fixture()
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture()
def binary_data():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    y_score = rng.uniform(0, 1, size=100)
    y_pred = (y_score >= 0.5).astype(int)
    return y_true, y_score, y_pred


# ---------------------------------------------------------------------------
# plot_learning_curves
# ---------------------------------------------------------------------------

def test_learning_curves_returns_path(tmp_dir):
    traces = {
        "train_loss": [1.0, 0.8, 0.6],
        "val_loss": [1.1, 0.9, 0.7],
        "val_macro_f1": [0.5, 0.6, 0.7],
    }
    path = plot_learning_curves(traces, tmp_dir / "lc.png")
    assert path is not None
    assert Path(path).exists()


def test_learning_curves_loss_only(tmp_dir):
    traces = {"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9]}
    path = plot_learning_curves(traces, tmp_dir / "lc_loss.png")
    assert path is not None


def test_learning_curves_empty_traces_returns_none(tmp_dir):
    path = plot_learning_curves({}, tmp_dir / "lc_empty.png")
    assert path is None


def test_learning_curves_bad_input_returns_none(tmp_dir):
    path = plot_learning_curves("not_a_dict", tmp_dir / "lc_bad.png")  # type: ignore
    assert path is None


# ---------------------------------------------------------------------------
# plot_classification_report
# ---------------------------------------------------------------------------

def test_classification_report_returns_path(tmp_dir, binary_data):
    y_true, _, y_pred = binary_data
    path = plot_classification_report(y_true, y_pred, None, tmp_dir / "cr.png")
    assert path is not None
    assert Path(path).exists()


def test_classification_report_with_class_names(tmp_dir, binary_data):
    y_true, _, y_pred = binary_data
    path = plot_classification_report(y_true, y_pred, ["neg", "pos"], tmp_dir / "cr_named.png")
    assert path is not None


def test_classification_report_bad_input_returns_none(tmp_dir):
    path = plot_classification_report(
        np.array([0, 1]), np.array([2, 3, 4]), None, tmp_dir / "cr_bad.png"
    )
    assert path is None


# ---------------------------------------------------------------------------
# plot_roc_curve
# ---------------------------------------------------------------------------

def test_roc_curve_returns_path(tmp_dir, binary_data):
    y_true, y_score, _ = binary_data
    path = plot_roc_curve(y_true, y_score, tmp_dir / "roc.png")
    assert path is not None
    assert Path(path).exists()


def test_roc_curve_mismatched_lengths_returns_none(tmp_dir):
    path = plot_roc_curve(np.array([0, 1, 0]), np.array([0.1, 0.9]), tmp_dir / "roc_bad.png")
    assert path is None


# ---------------------------------------------------------------------------
# plot_precision_recall_curve
# ---------------------------------------------------------------------------

def test_pr_curve_returns_path(tmp_dir, binary_data):
    y_true, y_score, _ = binary_data
    path = plot_precision_recall_curve(y_true, y_score, tmp_dir / "pr.png")
    assert path is not None
    assert Path(path).exists()


def test_pr_curve_mismatched_lengths_returns_none(tmp_dir):
    path = plot_precision_recall_curve(np.array([0, 1]), np.array([0.1, 0.9, 0.5]), tmp_dir / "pr_bad.png")
    assert path is None


# ---------------------------------------------------------------------------
# plot_calibration_diagram
# ---------------------------------------------------------------------------

def test_calibration_diagram_returns_path(tmp_dir, binary_data):
    y_true, y_score, _ = binary_data
    proba_after = np.clip(y_score + np.random.default_rng(1).uniform(-0.1, 0.1, len(y_score)), 0, 1)
    path = plot_calibration_diagram(y_true, y_score, proba_after, 10, tmp_dir / "cal.png")
    assert path is not None
    assert Path(path).exists()


def test_calibration_diagram_mismatched_lengths_returns_none(tmp_dir):
    path = plot_calibration_diagram(
        np.array([0, 1, 0]), np.array([0.1, 0.9]), np.array([0.2, 0.8]), 10, tmp_dir / "cal_bad.png"
    )
    assert path is None


# ---------------------------------------------------------------------------
# plot_feature_importance
# ---------------------------------------------------------------------------

def test_feature_importance_returns_path(tmp_dir):
    importances = np.array([0.3, 0.1, 0.4, 0.05, 0.15])
    path = plot_feature_importance(importances, None, 5, tmp_dir / "fi.png")
    assert path is not None
    assert Path(path).exists()


def test_feature_importance_with_names(tmp_dir):
    importances = np.array([0.3, 0.1, 0.4])
    path = plot_feature_importance(importances, ["a", "b", "c"], 3, tmp_dir / "fi_named.png")
    assert path is not None


def test_feature_importance_empty_returns_none(tmp_dir):
    path = plot_feature_importance(np.array([]), None, 5, tmp_dir / "fi_empty.png")
    assert path is None


# ---------------------------------------------------------------------------
# plot_threshold_vs_metric
# ---------------------------------------------------------------------------

def test_threshold_vs_metric_returns_path(tmp_dir, binary_data):
    y_true, y_score, _ = binary_data
    path = plot_threshold_vs_metric(y_true, y_score, "f1", tmp_dir / "thresh.png")
    assert path is not None
    assert Path(path).exists()


def test_threshold_vs_metric_bad_input_returns_none(tmp_dir):
    path = plot_threshold_vs_metric(
        np.array([0, 1]), np.array([0.5, 0.6, 0.7]), "f1", tmp_dir / "thresh_bad.png"
    )
    assert path is None
