"""Explainability manager: orchestrates all explainability methods with graceful fallback."""

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from aml_toolkit.artifacts.explainability_report import ExplainabilityOutput, ExplainabilityReport
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import ModalityType
from aml_toolkit.core.exceptions import ExplainabilityFailureWarning
from aml_toolkit.explainability.confusion_heatmap import ConfusionHeatmapStrategy
from aml_toolkit.explainability.faithfulness import feature_removal_faithfulness
from aml_toolkit.explainability.feature_importance import FeatureImportanceStrategy
from aml_toolkit.explainability.gradcam import GradCAMStrategy
from aml_toolkit.explainability.shap_explainer import ShapExplainerStrategy
from aml_toolkit.interfaces.candidate_model import CandidateModel
from aml_toolkit.interfaces.explainability import ExplainabilityStrategy

logger = logging.getLogger("aml_toolkit")

_TABULAR_STRATEGIES: dict[str, type[ExplainabilityStrategy]] = {
    "feature_importance": FeatureImportanceStrategy,
    "shap": ShapExplainerStrategy,
}

_IMAGE_STRATEGIES: dict[str, type[ExplainabilityStrategy]] = {
    "gradcam": GradCAMStrategy,
}

# Always run confusion heatmap regardless of modality
_UNIVERSAL_STRATEGIES: dict[str, type[ExplainabilityStrategy]] = {
    "confusion_heatmap": ConfusionHeatmapStrategy,
}


def run_explainability(
    trained_models: dict[str, CandidateModel],
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ToolkitConfig,
    output_dir: Path,
    modality: ModalityType = ModalityType.TABULAR,
) -> ExplainabilityReport:
    """Run all configured explainability methods on trained candidates.

    For each candidate and method:
    1. Check if the method supports the model.
    2. If yes, run it and record the output.
    3. If no, issue a warning and record the fallback reason.
    4. Optionally compute faithfulness scores.

    Unsupported methods warn but never crash the pipeline.

    Args:
        trained_models: Dict of candidate_id -> trained CandidateModel.
        X_val: Validation features.
        y_val: Validation labels.
        config: Toolkit configuration.
        output_dir: Base directory for saving artifacts.
        modality: Data modality (determines which methods to attempt).

    Returns:
        ExplainabilityReport with all outputs and caveats.
    """
    exp_config = config.explainability
    report = ExplainabilityReport()

    # Determine which strategies to run
    strategies: dict[str, ExplainabilityStrategy] = {}

    # Universal (always)
    for name, cls in _UNIVERSAL_STRATEGIES.items():
        strategies[name] = cls()

    # Modality-specific
    if modality in (ModalityType.TABULAR, ModalityType.EMBEDDING):
        for name in exp_config.tabular_methods:
            if name in _TABULAR_STRATEGIES:
                strategies[name] = _TABULAR_STRATEGIES[name]()
            else:
                logger.warning(f"Unknown tabular explainability method: {name}")

    if modality == ModalityType.IMAGE:
        for name in exp_config.image_methods:
            if name in _IMAGE_STRATEGIES:
                strategies[name] = _IMAGE_STRATEGIES[name]()
            else:
                logger.warning(f"Unknown image explainability method: {name}")

    # Run per candidate
    for candidate_id, model in trained_models.items():
        candidate_dir = Path(output_dir) / candidate_id

        for method_name, strategy in strategies.items():
            report.methods_attempted.append(f"{candidate_id}:{method_name}")

            if not strategy.supports_model(model):
                fallback = ExplainabilityOutput(
                    method=method_name,
                    candidate_id=candidate_id,
                    supported=False,
                    fallback_reason=f"Method '{method_name}' does not support this model.",
                )
                report.outputs.append(fallback)
                _record_unsupported(report, candidate_id, method_name, strategy)
                continue

            try:
                output = strategy.explain(model, X_val, y_val, candidate_dir / method_name, config)
                output.candidate_id = candidate_id

                if not output.supported:
                    _record_unsupported(report, candidate_id, method_name, strategy, output.fallback_reason)
                    report.outputs.append(output)
                    continue

                # Faithfulness check for feature-based methods
                if (
                    exp_config.faithfulness_enabled
                    and method_name == "feature_importance"
                    and output.supported
                    and output.summary.get("top_features")
                ):
                    importances = _reconstruct_importances(output)
                    if importances is not None:
                        score = feature_removal_faithfulness(
                            model, X_val, y_val, importances, top_k=min(5, len(importances))
                        )
                        output.faithfulness_score = score

                report.outputs.append(output)
                report.methods_succeeded.append(f"{candidate_id}:{method_name}")

            except Exception as e:
                warnings.warn(
                    f"Explainability method '{method_name}' failed for {candidate_id}: {e}",
                    ExplainabilityFailureWarning,
                    stacklevel=2,
                )
                fallback = ExplainabilityOutput(
                    method=method_name,
                    candidate_id=candidate_id,
                    supported=False,
                    fallback_reason=f"Unexpected error: {e}",
                )
                report.outputs.append(fallback)
                report.methods_failed.append(f"{candidate_id}:{method_name}")

    # Add caveats
    report.caveats.append(
        "Explainability outputs are approximations of model behavior, not ground truth. "
        "They should be interpreted with caution."
    )
    if any(not o.supported for o in report.outputs):
        report.caveats.append(
            "Some explainability methods were not supported for certain models. "
            "See individual output entries for details."
        )
    if exp_config.faithfulness_enabled:
        report.caveats.append(
            "Faithfulness scores measure explanation fidelity via feature removal. "
            "A low score may indicate the explanation does not reflect true model reliance."
        )

    return report


def _record_unsupported(
    report: ExplainabilityReport,
    candidate_id: str,
    method_name: str,
    strategy: ExplainabilityStrategy,
    reason: str | None = None,
) -> None:
    """Record that a method is unsupported without crashing."""
    msg = reason or f"Method '{method_name}' does not support this model."
    warnings.warn(
        f"Explainability: {msg} (candidate={candidate_id})",
        ExplainabilityFailureWarning,
        stacklevel=3,
    )
    report.methods_failed.append(f"{candidate_id}:{method_name}")


def _reconstruct_importances(output: ExplainabilityOutput) -> np.ndarray | None:
    """Reconstruct importance array from summary for faithfulness check."""
    n_features = output.summary.get("n_features", 0)
    top_features = output.summary.get("top_features", [])
    if n_features == 0 or not top_features:
        return None
    importances = np.zeros(n_features)
    for entry in top_features:
        idx = entry.get("index", 0)
        imp = entry.get("importance", 0.0)
        if idx < n_features:
            importances[idx] = imp
    return importances
