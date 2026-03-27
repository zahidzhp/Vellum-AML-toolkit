"""Image embedding probe: logistic/MLP head on pre-computed embeddings.

This module handles embedding-based probing for both IMAGE and EMBEDDING modalities.
For IMAGE modality, embeddings are expected to be pre-extracted (e.g., from a frozen
pretrained backbone). Actual feature extraction from raw images is deferred to later phases.
"""

import logging
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from aml_toolkit.artifacts import ProbeResult

logger = logging.getLogger("aml_toolkit")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metrics: list[str]) -> dict[str, float]:
    results: dict[str, float] = {}
    for m in metrics:
        if m == "accuracy":
            results[m] = float(accuracy_score(y_true, y_pred))
        elif m == "macro_f1":
            results[m] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        elif m == "weighted_f1":
            results[m] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return results


def run_embedding_probe(
    probe_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metrics: list[str],
    seed: int = 42,
) -> ProbeResult:
    """Run a probe on pre-computed embeddings.

    Args:
        probe_name: One of 'embedding_logistic', 'embedding_mlp'.
        X_train: Training embeddings.
        y_train: Training labels.
        X_val: Validation embeddings.
        y_val: Validation labels.
        metrics: Metric names to compute.
        seed: Random seed.

    Returns:
        ProbeResult with train/val metrics.
    """
    notes: list[str] = []
    start = time.time()

    if probe_name == "embedding_logistic":
        model = LogisticRegression(max_iter=500, random_state=seed, solver="lbfgs")
        model.fit(X_train, y_train)
    elif probe_name == "embedding_mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=200,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
        model.fit(X_train, y_train)
    else:
        notes.append(f"Unknown embedding probe: {probe_name}")
        return ProbeResult(
            model_name=probe_name,
            intervention_branch="none",
            modality="EMBEDDING",
            notes=notes,
        )

    fit_time = time.time() - start

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    return ProbeResult(
        model_name=probe_name,
        intervention_branch="none",
        train_metrics=_compute_metrics(y_train, train_pred, metrics),
        val_metrics=_compute_metrics(y_val, val_pred, metrics),
        fit_time_seconds=fit_time,
        modality="EMBEDDING",
        notes=notes,
    )
