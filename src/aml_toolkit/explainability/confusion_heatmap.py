"""Confusion matrix heatmap generation."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix

from aml_toolkit.artifacts.explainability_report import ExplainabilityOutput
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.interfaces.explainability import ExplainabilityStrategy

logger = logging.getLogger("aml_toolkit")


class ConfusionHeatmapStrategy(ExplainabilityStrategy):
    """Generate a confusion matrix heatmap for any classifier."""

    def explain(
        self,
        model: Any,
        X: Any,
        y: Any,
        output_dir: Path,
        config: ToolkitConfig,
    ) -> ExplainabilityOutput:
        output_dir.mkdir(parents=True, exist_ok=True)

        preds = model.predict(X)
        cm = confusion_matrix(y, preds)
        labels = sorted(set(y))

        # Save as numpy for downstream consumers (matplotlib rendering is optional)
        cm_path = output_dir / "confusion_matrix.npy"
        np.save(cm_path, cm)

        artifact_paths = [str(cm_path)]

        # Try to render a matplotlib heatmap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")

            fig.colorbar(im)
            fig.tight_layout()
            png_path = output_dir / "confusion_heatmap.png"
            fig.savefig(png_path, dpi=100)
            plt.close(fig)
            artifact_paths.append(str(png_path))
        except Exception as e:
            logger.warning(f"Matplotlib heatmap generation failed: {e}")

        summary = {
            "matrix": cm.tolist(),
            "labels": [int(l) if isinstance(l, (int, np.integer)) else str(l) for l in labels],
            "accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0,
        }

        return ExplainabilityOutput(
            method=self.method_name(),
            candidate_id="",  # filled by caller
            artifact_paths=artifact_paths,
            summary=summary,
        )

    def supports_model(self, model: Any) -> bool:
        return hasattr(model, "predict")

    def method_name(self) -> str:
        return "confusion_heatmap"
