"""Tabular shallow probe wrappers with lightweight intervention branches."""

import logging
import time
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from aml_toolkit.artifacts import ProbeResult

logger = logging.getLogger("aml_toolkit")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metrics: list[str]) -> dict[str, float]:
    """Compute requested metrics."""
    results: dict[str, float] = {}
    for m in metrics:
        if m == "accuracy":
            results[m] = float(accuracy_score(y_true, y_pred))
        elif m == "macro_f1":
            results[m] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        elif m == "weighted_f1":
            results[m] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return results


def _apply_intervention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    branch: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply a lightweight intervention branch to training data.

    Returns transformed X, y, and extra fit kwargs (e.g. class_weight).
    """
    fit_kwargs: dict[str, Any] = {}

    if branch == "none":
        pass
    elif branch == "class_weighting":
        fit_kwargs["class_weight"] = "balanced"
    elif branch == "oversampling":
        try:
            from imblearn.over_sampling import SMOTE

            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        except Exception as e:
            logger.warning(f"SMOTE failed, falling back to no intervention: {e}")
    elif branch == "undersampling":
        try:
            from imblearn.under_sampling import RandomUnderSampler

            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)
        except Exception as e:
            logger.warning(f"Undersampling failed, falling back to no intervention: {e}")

    return X_train, y_train, fit_kwargs


def run_tabular_probe(
    probe_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metrics: list[str],
    intervention_branch: str = "none",
    seed: int = 42,
) -> ProbeResult:
    """Run a single tabular probe with optional intervention branch.

    Args:
        probe_name: One of 'logistic', 'rf', 'xgb'.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        metrics: Metric names to compute.
        intervention_branch: Intervention to apply ('none', 'class_weighting', 'oversampling', 'undersampling').
        seed: Random seed.

    Returns:
        ProbeResult with train/val metrics.
    """
    X_tr, y_tr, fit_kwargs = _apply_intervention(X_train.copy(), y_train.copy(), intervention_branch)
    notes: list[str] = []

    start = time.time()

    if probe_name == "logistic":
        cw = fit_kwargs.get("class_weight", None)
        model = LogisticRegression(
            max_iter=500, random_state=seed, class_weight=cw, solver="lbfgs"
        )
        model.fit(X_tr, y_tr)
    elif probe_name == "rf":
        cw = fit_kwargs.get("class_weight", None)
        model = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=seed, class_weight=cw
        )
        model.fit(X_tr, y_tr)
    elif probe_name == "xgb":
        try:
            from xgboost import XGBClassifier

            n_classes = len(np.unique(y_tr))
            model = XGBClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=seed,
                use_label_encoder=False,
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                verbosity=0,
            )
            model.fit(X_tr, y_tr)
        except Exception as e:
            notes.append(f"XGBoost not available ({type(e).__name__}), skipping probe.")
            return ProbeResult(
                model_name=probe_name,
                intervention_branch=intervention_branch,
                notes=notes,
            )
    else:
        notes.append(f"Unknown probe name: {probe_name}")
        return ProbeResult(
            model_name=probe_name,
            intervention_branch=intervention_branch,
            notes=notes,
        )

    fit_time = time.time() - start

    train_pred = model.predict(X_tr)
    val_pred = model.predict(X_val)

    train_metrics = _compute_metrics(y_tr, train_pred, metrics)
    val_metrics = _compute_metrics(y_val, val_pred, metrics)

    return ProbeResult(
        model_name=probe_name,
        intervention_branch=intervention_branch,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        fit_time_seconds=fit_time,
        modality="TABULAR",
        notes=notes,
    )
