"""Baseline probe models: majority class and stratified random."""

import time
from collections import Counter

import numpy as np

from aml_toolkit.artifacts import ProbeResult


class MajorityBaseline:
    """Predicts the most frequent class in the training set."""

    def __init__(self) -> None:
        self._majority_class: int | str | None = None
        self._n_classes: int = 2
        self._fit_time: float = 0.0

    def fit(self, y_train: np.ndarray) -> None:
        start = time.time()
        counter = Counter(y_train.tolist())
        self._majority_class = counter.most_common(1)[0][0]
        self._n_classes = len(counter)
        self._fit_time = time.time() - start

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._majority_class)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, metrics: list[str]) -> dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score

        results: dict[str, float] = {}
        for m in metrics:
            if m == "accuracy":
                results[m] = float(accuracy_score(y_true, y_pred))
            elif m == "macro_f1":
                results[m] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            elif m == "weighted_f1":
                results[m] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        return results

    def to_probe_result(self, y_train: np.ndarray, y_val: np.ndarray, metrics: list[str]) -> ProbeResult:
        train_pred = self.predict(np.zeros(len(y_train)))
        val_pred = self.predict(np.zeros(len(y_val)))
        return ProbeResult(
            model_name="majority_baseline",
            intervention_branch="none",
            train_metrics=self.evaluate(y_train, train_pred, metrics),
            val_metrics=self.evaluate(y_val, val_pred, metrics),
            fit_time_seconds=self._fit_time,
            notes=[f"Majority class: {self._majority_class}"],
        )


class StratifiedBaseline:
    """Predicts classes proportionally to training set distribution."""

    def __init__(self) -> None:
        self._classes: np.ndarray = np.array([])
        self._probs: np.ndarray = np.array([])
        self._fit_time: float = 0.0
        self._seed: int = 42

    def fit(self, y_train: np.ndarray, seed: int = 42) -> None:
        start = time.time()
        self._seed = seed
        classes, counts = np.unique(y_train, return_counts=True)
        self._classes = classes
        self._probs = counts / counts.sum()
        self._fit_time = time.time() - start

    def predict(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self._seed)
        return rng.choice(self._classes, size=len(X), p=self._probs)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, metrics: list[str]) -> dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score

        results: dict[str, float] = {}
        for m in metrics:
            if m == "accuracy":
                results[m] = float(accuracy_score(y_true, y_pred))
            elif m == "macro_f1":
                results[m] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            elif m == "weighted_f1":
                results[m] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        return results

    def to_probe_result(self, y_train: np.ndarray, y_val: np.ndarray, metrics: list[str]) -> ProbeResult:
        train_pred = self.predict(np.zeros(len(y_train)))
        val_pred = self.predict(np.zeros(len(y_val)))
        return ProbeResult(
            model_name="stratified_baseline",
            intervention_branch="none",
            train_metrics=self.evaluate(y_train, train_pred, metrics),
            val_metrics=self.evaluate(y_val, val_pred, metrics),
            fit_time_seconds=self._fit_time,
            notes=[f"Class distribution: {dict(zip(self._classes.tolist(), self._probs.tolist()))}"],
        )
