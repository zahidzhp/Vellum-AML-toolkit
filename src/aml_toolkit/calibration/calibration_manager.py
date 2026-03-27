"""Calibration manager: orchestrates calibration and threshold optimization for trained candidates."""

import logging
from typing import Any

import numpy as np

from aml_toolkit.artifacts import CalibrationReport, CalibrationResult
from aml_toolkit.calibration.isotonic import IsotonicCalibrator
from aml_toolkit.calibration.temperature_scaling import TemperatureScalingCalibrator
from aml_toolkit.calibration.threshold_optimizer import ThresholdOptimizer
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.exceptions import CalibrationFailureError
from aml_toolkit.interfaces.calibrator import Calibrator
from aml_toolkit.interfaces.candidate_model import CandidateModel

logger = logging.getLogger("aml_toolkit")

_CALIBRATOR_REGISTRY: dict[str, type[Calibrator]] = {
    "temperature_scaling": TemperatureScalingCalibrator,
    "isotonic": IsotonicCalibrator,
}


def run_calibration(
    trained_models: dict[str, CandidateModel],
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ToolkitConfig,
) -> CalibrationReport:
    """Run calibration and threshold optimization on all trained candidates.

    For each candidate:
    1. Check if the model is probabilistic. If not, log a warning and skip.
    2. Get raw probabilities on the validation set.
    3. For each enabled calibration method, fit and evaluate.
    4. Select the method that best optimizes the primary objective.
    5. Optimize the decision threshold on calibrated probabilities.

    Args:
        trained_models: Dict of candidate_id -> trained CandidateModel.
        X_val: Validation features.
        y_val: Validation labels.
        config: Toolkit configuration.

    Returns:
        CalibrationReport with per-candidate results.
    """
    cal_config = config.calibration
    report = CalibrationReport(primary_objective=cal_config.primary_objective)

    for candidate_id, model in trained_models.items():
        try:
            result = _calibrate_candidate(
                candidate_id, model, X_val, y_val, cal_config.enabled_methods,
                cal_config.primary_objective,
            )
            report.results.append(result)
        except CalibrationFailureError as e:
            logger.warning(f"Calibration failed for {candidate_id}: {e}")
            report.warnings.append(f"{candidate_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected calibration error for {candidate_id}: {e}")
            report.warnings.append(f"{candidate_id}: unexpected error: {e}")

    return report


def _calibrate_candidate(
    candidate_id: str,
    model: CandidateModel,
    X_val: np.ndarray,
    y_val: np.ndarray,
    enabled_methods: list[str],
    primary_objective: str,
) -> CalibrationResult:
    """Calibrate a single candidate.

    Returns the best calibration result based on the primary objective.
    """
    # Check probabilistic
    if not model.is_probabilistic():
        return CalibrationResult(
            candidate_id=candidate_id,
            method="none",
            objective_metric=primary_objective,
            notes=["Non-probabilistic model; calibration skipped."],
        )

    raw_proba = model.predict_proba(X_val)
    if raw_proba is None:
        return CalibrationResult(
            candidate_id=candidate_id,
            method="none",
            objective_metric=primary_objective,
            notes=["predict_proba returned None; calibration skipped."],
        )

    # Extract positive-class probabilities for binary case
    proba_pos = _extract_positive_class_proba(raw_proba)

    best_result: CalibrationResult | None = None
    best_objective_value: float = float("inf")

    for method_name in enabled_methods:
        if method_name not in _CALIBRATOR_REGISTRY:
            logger.warning(f"Unknown calibration method '{method_name}'; skipping.")
            continue

        calibrator = _CALIBRATOR_REGISTRY[method_name]()

        try:
            calibrator.fit(proba_pos, y_val)
            calibrated = calibrator.calibrate(proba_pos)
            result = calibrator.evaluate(proba_pos, calibrated, y_val)
            result.candidate_id = candidate_id
            result.objective_metric = primary_objective

            # Select best by primary objective (lower is better for ECE and Brier)
            obj_value = _get_objective_after(result, primary_objective)
            if obj_value < best_objective_value:
                best_objective_value = obj_value
                best_result = result

        except Exception as e:
            logger.warning(f"Calibration method '{method_name}' failed for {candidate_id}: {e}")

    if best_result is None:
        raise CalibrationFailureError(
            f"All calibration methods failed for {candidate_id}."
        )

    # Threshold optimization on calibrated probabilities
    try:
        optimizer = ThresholdOptimizer(metric="f1")
        # Re-calibrate with the winning method to get calibrated proba
        winning_calibrator = _CALIBRATOR_REGISTRY[best_result.method]()
        winning_calibrator.fit(proba_pos, y_val)
        calibrated_for_threshold = winning_calibrator.calibrate(proba_pos)

        best_thresh = optimizer.optimize(y_val, calibrated_for_threshold)
        best_result.threshold_optimized = best_thresh
        best_result.notes.append(
            f"Threshold optimized to {best_thresh:.4f} (F1={optimizer.best_score:.4f})."
        )
    except Exception as e:
        logger.warning(f"Threshold optimization failed for {candidate_id}: {e}")
        best_result.notes.append(f"Threshold optimization failed: {e}")

    return best_result


def _extract_positive_class_proba(proba: np.ndarray) -> np.ndarray:
    """Extract positive-class probabilities from a probability array.

    Handles both (n,) shape (already positive class) and (n, k) shape
    (take column 1 for binary, or max for multiclass).
    """
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 2:
        return proba[:, 1]
    # Multiclass: return the full array (caller must handle)
    return proba[:, 1] if proba.shape[1] >= 2 else proba.ravel()


def _get_objective_after(result: CalibrationResult, objective: str) -> float:
    """Get the post-calibration objective value (lower is better)."""
    if objective == "ece":
        return result.ece_after if result.ece_after is not None else float("inf")
    if objective == "brier":
        return result.brier_after if result.brier_after is not None else float("inf")
    return float("inf")
