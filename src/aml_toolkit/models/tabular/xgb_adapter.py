"""XGBoost model adapter scaffold."""

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.interfaces.candidate_model import CandidateModel


class XGBAdapter(CandidateModel):
    """Adapter wrapping XGBClassifier. Gracefully handles missing XGBoost."""

    def __init__(self, seed: int = 42) -> None:
        self._model: Any = None
        self._seed = seed
        self._trace: dict[str, list[float]] = {}
        self._available = True
        try:
            from xgboost import XGBClassifier
        except Exception:
            self._available = False

    def fit(self, X_train: Any, y_train: Any, X_val: Any, y_val: Any, config: ToolkitConfig) -> None:
        if not self._available:
            return
        from xgboost import XGBClassifier

        n_classes = len(np.unique(y_train))
        self._model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=self._seed,
            use_label_encoder=False,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            verbosity=0,
        )
        self._model.fit(X_train, y_train)
        val_pred = self._model.predict(X_val)
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        self._trace = {"val_macro_f1": [val_f1]}

    def predict(self, X: Any) -> np.ndarray:
        if self._model is None:
            return np.zeros(len(X))
        return self._model.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray | None:
        if self._model is None:
            return None
        return self._model.predict_proba(X)

    def evaluate(self, X: Any, y: Any, metrics: list[str]) -> dict[str, float]:
        pred = self.predict(X)
        results: dict[str, float] = {}
        for m in metrics:
            if m == "accuracy":
                results[m] = float(accuracy_score(y, pred))
            elif m == "macro_f1":
                results[m] = float(f1_score(y, pred, average="macro", zero_division=0))
        return results

    def get_training_trace(self) -> dict[str, list[float]]:
        return self._trace

    def get_model_family(self) -> str:
        return "xgb"

    def is_probabilistic(self) -> bool:
        return True

    def serialize(self, path: Any) -> None:
        if self._model is not None:
            self._model.save_model(str(path))
