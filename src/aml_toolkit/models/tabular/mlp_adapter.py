"""MLP (sklearn) model adapter scaffold."""

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.interfaces.candidate_model import CandidateModel


class MLPAdapter(CandidateModel):
    """Adapter wrapping sklearn MLPClassifier."""

    def __init__(self, seed: int = 42) -> None:
        self._model: MLPClassifier | None = None
        self._seed = seed
        self._trace: dict[str, list[float]] = {}

    def fit(self, X_train: Any, y_train: Any, X_val: Any, y_val: Any, config: ToolkitConfig) -> None:
        self._model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            random_state=self._seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self._model.fit(X_train, y_train)
        val_pred = self._model.predict(X_val)
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        self._trace = {"val_macro_f1": [val_f1], "train_loss": list(self._model.loss_curve_)}

    def predict(self, X: Any) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray | None:
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
        return "mlp"

    def is_probabilistic(self) -> bool:
        return True

    def serialize(self, path: Any) -> None:
        import joblib
        joblib.dump(self._model, Path(path))
