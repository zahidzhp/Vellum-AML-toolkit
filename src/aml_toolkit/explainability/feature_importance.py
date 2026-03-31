"""Feature importance explainability for tree-based and linear models."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from aml_toolkit.artifacts.explainability_report import ExplainabilityOutput
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.interfaces.explainability import ExplainabilityStrategy

logger = logging.getLogger("aml_toolkit")


class FeatureImportanceStrategy(ExplainabilityStrategy):
    """Extract feature importance from models that expose it natively.

    Supports sklearn tree models (feature_importances_), linear models (coef_),
    and XGBoost.
    """

    def explain(
        self,
        model: Any,
        X: Any,
        y: Any,
        output_dir: Path,
        config: ToolkitConfig,
    ) -> ExplainabilityOutput:
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_model = self._extract_raw_model(model)
        importances = self._get_importances(raw_model, X)

        if importances is None:
            return ExplainabilityOutput(
                method=self.method_name(),
                candidate_id="",
                supported=False,
                fallback_reason="Model does not expose feature importances.",
            )

        # Save
        npy_path = output_dir / "feature_importances.npy"
        np.save(npy_path, importances)

        # Top features
        n_features = len(importances)
        top_k = min(10, n_features)
        top_indices = np.argsort(np.abs(importances))[::-1][:top_k]

        summary = {
            "n_features": n_features,
            "top_features": [
                {"index": int(i), "importance": float(importances[i])}
                for i in top_indices
            ],
        }

        artifact_paths = [str(npy_path)]

        # Bar chart
        from aml_toolkit.reporting.plot_utils import plot_feature_importance
        chart_path = plot_feature_importance(
            importances, None, top_k, output_dir / "feature_importance_chart.png"
        )
        if chart_path:
            artifact_paths.append(chart_path)

        return ExplainabilityOutput(
            method=self.method_name(),
            candidate_id="",
            artifact_paths=artifact_paths,
            summary=summary,
        )

    def supports_model(self, model: Any) -> bool:
        raw = self._extract_raw_model(model)
        return (
            hasattr(raw, "feature_importances_")
            or hasattr(raw, "coef_")
        )

    def method_name(self) -> str:
        return "feature_importance"

    def _extract_raw_model(self, model: Any) -> Any:
        """Try to get the underlying sklearn/xgb model from an adapter."""
        if hasattr(model, "_model"):
            return model._model
        return model

    def _get_importances(self, raw_model: Any, X: Any) -> np.ndarray | None:
        if hasattr(raw_model, "feature_importances_"):
            return np.asarray(raw_model.feature_importances_)
        if hasattr(raw_model, "coef_"):
            coef = np.asarray(raw_model.coef_)
            if coef.ndim > 1:
                return np.abs(coef).mean(axis=0)
            return np.abs(coef)
        return None
