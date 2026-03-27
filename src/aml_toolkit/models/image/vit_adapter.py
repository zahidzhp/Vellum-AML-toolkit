"""ViT model adapter scaffold (placeholder for Phase 9+ full training)."""

from typing import Any

import numpy as np

from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.interfaces.candidate_model import CandidateModel


class ViTAdapter(CandidateModel):
    """Placeholder adapter for ViT-based image classification.

    Full training logic deferred to Phase 9. This scaffold establishes
    the interface contract and registry entry.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._trace: dict[str, list[float]] = {}

    def fit(self, X_train: Any, y_train: Any, X_val: Any, y_val: Any, config: ToolkitConfig) -> None:
        raise NotImplementedError("ViT training deferred to Phase 9.")

    def predict(self, X: Any) -> np.ndarray:
        raise NotImplementedError("ViT predict deferred to Phase 9.")

    def predict_proba(self, X: Any) -> np.ndarray | None:
        raise NotImplementedError("ViT predict_proba deferred to Phase 9.")

    def evaluate(self, X: Any, y: Any, metrics: list[str]) -> dict[str, float]:
        raise NotImplementedError("ViT evaluate deferred to Phase 9.")

    def get_training_trace(self) -> dict[str, list[float]]:
        return self._trace

    def get_model_family(self) -> str:
        return "vit"

    def is_probabilistic(self) -> bool:
        return True

    def serialize(self, path: Any) -> None:
        raise NotImplementedError("ViT serialize deferred to Phase 9.")
