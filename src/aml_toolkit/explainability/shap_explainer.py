"""SHAP-based explainability for tabular models."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from aml_toolkit.artifacts.explainability_report import ExplainabilityOutput
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.interfaces.explainability import ExplainabilityStrategy

logger = logging.getLogger("aml_toolkit")


class ShapExplainerStrategy(ExplainabilityStrategy):
    """SHAP value computation for tabular models.

    Uses the shap library to compute feature contributions.
    Fails gracefully if shap is not installed or the model is unsupported.
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

        try:
            import shap
        except ImportError:
            return ExplainabilityOutput(
                method=self.method_name(),
                candidate_id="",
                supported=False,
                fallback_reason="shap library is not installed.",
            )

        raw_model = self._extract_raw_model(model)

        try:
            # Use a small background sample for speed
            n_background = min(50, len(X))
            background = X[:n_background]

            predict_fn = self._get_predict_fn(model)
            explainer = shap.Explainer(predict_fn, background)
            shap_values = explainer(X[:min(100, len(X))])

            # Save shap values
            vals = shap_values.values
            npy_path = output_dir / "shap_values.npy"
            np.save(npy_path, vals)

            # Mean absolute SHAP per feature
            if vals.ndim == 3:
                mean_abs = np.abs(vals).mean(axis=(0, 2))
            else:
                mean_abs = np.abs(vals).mean(axis=0)

            top_k = min(10, len(mean_abs))
            top_indices = np.argsort(mean_abs)[::-1][:top_k]

            summary = {
                "n_samples_explained": vals.shape[0],
                "n_features": vals.shape[1] if vals.ndim >= 2 else 0,
                "top_features": [
                    {"index": int(i), "mean_abs_shap": float(mean_abs[i])}
                    for i in top_indices
                ],
            }

            return ExplainabilityOutput(
                method=self.method_name(),
                candidate_id="",
                artifact_paths=[str(npy_path)],
                summary=summary,
            )

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return ExplainabilityOutput(
                method=self.method_name(),
                candidate_id="",
                supported=False,
                fallback_reason=f"SHAP computation failed: {e}",
            )

    def supports_model(self, model: Any) -> bool:
        # SHAP supports most sklearn-compatible models
        raw = self._extract_raw_model(model)
        return hasattr(raw, "predict") or hasattr(model, "predict")

    def method_name(self) -> str:
        return "shap"

    def _extract_raw_model(self, model: Any) -> Any:
        if hasattr(model, "_model"):
            return model._model
        return model

    def _get_predict_fn(self, model: Any) -> Any:
        """Get a predict function suitable for SHAP."""
        if hasattr(model, "predict_proba"):
            proba_fn = model.predict_proba
            # Test if it works
            return proba_fn
        return model.predict
