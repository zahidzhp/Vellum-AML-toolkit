"""Visualization utilities: reusable plot functions for pipeline output.

Every plot function follows the same contract:
- Uses matplotlib Agg backend (headless, no display)
- Saves a PNG to the given path
- Returns the path as a string on success, None on failure
- Never raises — wrapped in @safe_plot decorator that logs warnings
"""

import functools
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("aml_toolkit")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_agg() -> None:
    import matplotlib
    matplotlib.use("Agg")


def save_fig(fig: Any, path: Path, dpi: int = 150) -> str:
    """Save a matplotlib figure to PNG, close it, return path string."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(path)


def safe_plot(func):
    """Decorator: catch all exceptions, log warning, return None."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Plot '{func.__name__}' failed: {e}")
            return None
    return wrapper


# ---------------------------------------------------------------------------
# 1. Learning Curves
# ---------------------------------------------------------------------------

@safe_plot
def plot_learning_curves(
    traces: dict[str, list[float]],
    output_path: Path,
) -> str | None:
    """Plot training loss, validation loss, and validation F1 over epochs."""
    _ensure_agg()
    import matplotlib.pyplot as plt

    has_loss = "train_loss" in traces or "val_loss" in traces
    has_f1 = "val_macro_f1" in traces

    n_panels = int(has_loss) + int(has_f1)
    if n_panels == 0:
        return None

    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4 * n_panels), squeeze=False)
    panel = 0

    if has_loss:
        ax = axes[panel, 0]
        if "train_loss" in traces:
            ax.plot(traces["train_loss"], label="Train Loss", color="#2196F3")
        if "val_loss" in traces:
            ax.plot(traces["val_loss"], label="Val Loss", color="#FF5722")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        panel += 1

    if has_f1:
        ax = axes[panel, 0]
        ax.plot(traces["val_macro_f1"], label="Val Macro F1", color="#4CAF50", marker="o", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro F1")
        ax.set_title("Validation F1 Over Epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Learning Curves", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 2. Classification Report Table
# ---------------------------------------------------------------------------

@safe_plot
def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None,
    output_path: Path,
) -> str | None:
    """Render a classification report (precision/recall/F1) as a table image."""
    _ensure_agg()
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Build table data
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rows = []
    for key, metrics in report.items():
        if isinstance(metrics, dict):
            label = key
            if class_names and key.isdigit():
                idx = int(key)
                if idx < len(class_names):
                    label = class_names[idx]
            rows.append([
                label,
                f"{metrics.get('precision', 0):.3f}",
                f"{metrics.get('recall', 0):.3f}",
                f"{metrics.get('f1-score', 0):.3f}",
                str(int(metrics.get('support', 0))),
            ])

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.5))
    ax.axis("off")
    ax.set_title("Classification Report", fontsize=14, fontweight="bold", pad=20)

    colors = []
    for row in rows:
        f1 = float(row[3])
        if row[0] in ("macro avg", "weighted avg", "accuracy"):
            colors.append(["#E3F2FD"] * 5)
        elif f1 >= 0.8:
            colors.append(["#E8F5E9"] * 5)
        elif f1 >= 0.5:
            colors.append(["#FFF3E0"] * 5)
        else:
            colors.append(["#FFEBEE"] * 5)

    table = ax.table(
        cellText=rows, colLabels=headers, cellColours=colors,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    return save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 3. ROC Curve
# ---------------------------------------------------------------------------

@safe_plot
def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
) -> str | None:
    """Plot ROC curve with AUC annotation (binary classification)."""
    _ensure_agg()
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#1565C0", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1565C0")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    return save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 4. Precision-Recall Curve
# ---------------------------------------------------------------------------

@safe_plot
def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
) -> str | None:
    """Plot Precision-Recall curve with AP annotation (binary classification)."""
    _ensure_agg()
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(recall, precision, color="#4CAF50", lw=2, label=f"PR (AP = {ap:.3f})")
    ax.fill_between(recall, precision, alpha=0.1, color="#4CAF50")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    return save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 5. Calibration Reliability Diagram
# ---------------------------------------------------------------------------

@safe_plot
def plot_calibration_diagram(
    y_true: np.ndarray,
    proba_before: np.ndarray,
    proba_after: np.ndarray,
    n_bins: int,
    output_path: Path,
) -> str | None:
    """Plot reliability diagram: expected vs observed probability, before and after calibration."""
    _ensure_agg()
    import matplotlib.pyplot as plt

    def _bin_data(y: np.ndarray, proba: np.ndarray, n_bins: int):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_mids = []
        bin_freqs = []
        for i in range(n_bins):
            mask = (proba >= bins[i]) & (proba < bins[i + 1])
            if mask.sum() > 0:
                bin_mids.append((bins[i] + bins[i + 1]) / 2)
                bin_freqs.append(y[mask].mean())
        return np.array(bin_mids), np.array(bin_freqs)

    y = np.asarray(y_true, dtype=float)
    mid_before, freq_before = _bin_data(y, proba_before, n_bins)
    mid_after, freq_after = _bin_data(y, proba_after, n_bins)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Perfectly Calibrated")
    if len(mid_before) > 0:
        ax.plot(mid_before, freq_before, "o-", color="#FF5722", lw=2, label="Before Calibration")
    if len(mid_after) > 0:
        ax.plot(mid_after, freq_after, "s-", color="#1565C0", lw=2, label="After Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Reliability Diagram", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    return save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 6. Feature Importance Bar Chart
# ---------------------------------------------------------------------------

@safe_plot
def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str] | None,
    top_k: int,
    output_path: Path,
) -> str | None:
    """Horizontal bar chart of top-K feature importances."""
    _ensure_agg()
    import matplotlib.pyplot as plt

    n = len(importances)
    if n == 0:
        return None

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n)]

    # Sort by absolute importance
    indices = np.argsort(np.abs(importances))[::-1][:top_k]
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals = importances[indices]

    # Colors: positive = blue, negative = red
    colors = ["#1565C0" if v >= 0 else "#D32F2F" for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * top_k + 1)))
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_vals, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Top Features)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    return save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 7. Threshold vs Metric Curve
# ---------------------------------------------------------------------------

@safe_plot
def plot_threshold_vs_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str,
    output_path: Path,
) -> str | None:
    """Plot threshold sweep curve showing metric vs decision threshold."""
    _ensure_agg()
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score

    thresholds = np.linspace(0.01, 0.99, 99)
    scores = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        if len(np.unique(preds)) == 1 and np.unique(preds)[0] == 0:
            scores.append(0.0)
        else:
            scores.append(float(f1_score(y_true, preds, average="binary", zero_division=0)))
    scores = np.array(scores)

    best_idx = np.argmax(scores)
    best_thresh = thresholds[best_idx]
    best_score = scores[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, scores, color="#4CAF50", lw=2)
    ax.axvline(best_thresh, color="#D32F2F", linestyle="--", lw=1.5,
               label=f"Optimal: {best_thresh:.3f} ({metric_name}={best_score:.3f})")
    ax.scatter([best_thresh], [best_score], color="#D32F2F", s=80, zorder=5)
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f"Threshold vs {metric_name.upper()}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    return save_fig(fig, output_path)
