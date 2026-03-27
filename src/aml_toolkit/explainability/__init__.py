"""Explainability module: heatmaps, feature importance, SHAP, Grad-CAM, and faithfulness."""

from aml_toolkit.explainability.confusion_heatmap import ConfusionHeatmapStrategy
from aml_toolkit.explainability.explainability_manager import run_explainability
from aml_toolkit.explainability.faithfulness import feature_removal_faithfulness
from aml_toolkit.explainability.feature_importance import FeatureImportanceStrategy
from aml_toolkit.explainability.gradcam import GradCAMStrategy
from aml_toolkit.explainability.shap_explainer import ShapExplainerStrategy

__all__ = [
    "ConfusionHeatmapStrategy",
    "FeatureImportanceStrategy",
    "GradCAMStrategy",
    "ShapExplainerStrategy",
    "feature_removal_faithfulness",
    "run_explainability",
]
