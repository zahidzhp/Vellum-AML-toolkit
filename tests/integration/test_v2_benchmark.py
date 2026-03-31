"""V2 benchmark tests — verify V2 features don't degrade accuracy.

These tests use sklearn's synthetic datasets to run lightweight end-to-end
checks without touching real ML training.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from aml_toolkit.ensemble.greedy_diverse import GreedyDiverseEnsemble
from aml_toolkit.core.config import DynamicEnsembleConfig
from aml_toolkit.uncertainty.estimator import UncertaintyEstimator
from aml_toolkit.core.config import UncertaintyConfig
from aml_toolkit.uncertainty.conformal import SplitConformalPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def imbalanced_binary():
    """Imbalanced binary dataset: 90/10 class split."""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        weights=[0.9, 0.1],
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def balanced_multiclass():
    """Balanced 3-class dataset."""
    X, y = make_classification(
        n_samples=600,
        n_features=20,
        n_informative=10,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Conformal coverage guarantee
# ---------------------------------------------------------------------------

class TestConformalCoverageOnRealData:
    def test_coverage_guarantee_binary(self, imbalanced_binary):
        X_train, X_test, y_train, y_test = imbalanced_binary
        model = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
        model.fit(X_train, y_train)

        proba_cal = model.predict_proba(X_train)
        proba_test = model.predict_proba(X_test)

        predictor = SplitConformalPredictor(coverage=0.9)
        predictor.fit(proba_cal, y_train)
        coverage = predictor.empirical_coverage(proba_test, y_test)

        assert coverage >= 0.85, f"Coverage {coverage:.3f} too low on binary"

    def test_coverage_guarantee_multiclass(self, balanced_multiclass):
        X_train, X_test, y_train, y_test = balanced_multiclass
        model = LogisticRegression(max_iter=500, random_state=0)
        model.fit(X_train, y_train)

        proba_cal = model.predict_proba(X_train)
        proba_test = model.predict_proba(X_test)

        predictor = SplitConformalPredictor(coverage=0.9)
        predictor.fit(proba_cal, y_train)
        coverage = predictor.empirical_coverage(proba_test, y_test)

        assert coverage >= 0.85, f"Coverage {coverage:.3f} too low on multiclass"


# ---------------------------------------------------------------------------
# Greedy diverse ensemble vs single best model
# ---------------------------------------------------------------------------

class TestGreedyDiverseOnRealData:
    def test_diverse_ensemble_no_regression(self, imbalanced_binary):
        """Greedy diverse ensemble should not be worse than best individual by >5%."""
        X_train, X_test, y_train, y_test = imbalanced_binary

        # Train multiple models with different hyperparameters
        models = {
            "lr_balanced": LogisticRegression(class_weight="balanced", max_iter=500, random_state=0),
            "lr_none": LogisticRegression(max_iter=500, random_state=1),
            "lr_c01": LogisticRegression(C=0.1, max_iter=500, random_state=2),
        }
        for m in models.values():
            m.fit(X_train, y_train)

        probas = {name: m.predict_proba(X_train) for name, m in models.items()}
        individual_f1s = {
            name: f1_score(y_train, np.argmax(p, axis=1), average="macro", zero_division=0)
            for name, p in probas.items()
        }

        config = DynamicEnsembleConfig(max_members=3, diversity_threshold=0.02)
        ensemble = GreedyDiverseEnsemble(config)
        report = ensemble.select(probas, y_train, candidate_scores=individual_f1s)

        best_individual = max(individual_f1s.values())
        # Ensemble score should not regress by more than 5%
        assert report.ensemble_score is not None
        assert report.ensemble_score >= best_individual - 0.05, (
            f"Ensemble F1 {report.ensemble_score:.3f} < best individual {best_individual:.3f} - 0.05"
        )

    def test_single_model_selected_when_one_candidate(self, imbalanced_binary):
        X_train, _, y_train, _ = imbalanced_binary
        model = LogisticRegression(class_weight="balanced", max_iter=500, random_state=0)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_train)

        config = DynamicEnsembleConfig(max_members=3)
        ensemble = GreedyDiverseEnsemble(config)
        report = ensemble.select({"only_model": proba}, y_train)
        assert "only_model" in report.member_ids


# ---------------------------------------------------------------------------
# Uncertainty estimation on real data
# ---------------------------------------------------------------------------

class TestUncertaintyOnRealData:
    def test_calibrated_model_lower_uncertainty_than_random(self, imbalanced_binary):
        X_train, X_test, y_train, y_test = imbalanced_binary

        # Calibrated model: trained on data
        model = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        proba_calibrated = model.predict_proba(X_test)

        # Random model: random probabilities
        rng = np.random.default_rng(0)
        proba_random = rng.dirichlet([1, 1], size=len(X_test))

        config = UncertaintyConfig(enabled=True, methods=["entropy"])
        estimator = UncertaintyEstimator(config)

        report_calibrated = estimator.estimate("calibrated", proba_calibrated)
        report_random = estimator.estimate("random", proba_random)

        # A trained model should be more confident (lower entropy) than random
        assert report_calibrated.mean_uncertainty < report_random.mean_uncertainty, (
            f"Trained model uncertainty {report_calibrated.mean_uncertainty:.3f} >= "
            f"random {report_random.mean_uncertainty:.3f}"
        )

    def test_uncertainty_report_fields_populated(self, balanced_multiclass):
        X_train, X_test, y_train, y_test = balanced_multiclass
        model = LogisticRegression(max_iter=500, random_state=0)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)

        config = UncertaintyConfig(
            enabled=True,
            methods=["entropy", "margin"],
            conformal_enabled=True,
            conformal_coverage=0.9,
        )
        estimator = UncertaintyEstimator(config)
        report = estimator.estimate("lr", proba, y_val=y_test)

        assert report.entropy_mean is not None
        assert report.margin_mean is not None
        assert report.mean_prediction_set_size is not None
        assert report.sample_count == len(X_test)


# ---------------------------------------------------------------------------
# V2 all-off behavior matches V1 (coordinator level)
# ---------------------------------------------------------------------------

class TestV2OffMatchesV1Behavior:
    def test_coordinator_v2_off_returns_no_modifications(self, imbalanced_binary):
        """With all V2 features disabled, coordinator should return empty results."""
        from aml_toolkit.adaptive.coordinator import AdaptiveIntelligenceCoordinator
        from aml_toolkit.core.config import ToolkitConfig

        cfg = ToolkitConfig()  # all advanced off
        coordinator = AdaptiveIntelligenceCoordinator(cfg)

        # Pre-training: no reordering
        rec = coordinator.pre_training_recommendations({}, {})
        assert rec.candidate_order == []
        assert rec.compute_budget_fractions == {}

        # Post-calibration: no uncertainty
        from aml_toolkit.artifacts.calibration_report import CalibrationReport
        cal = CalibrationReport(primary_objective="ece")
        X_train, X_test, y_train, y_test = imbalanced_binary
        result = coordinator.post_calibration_analysis(cal, {}, None, y_test)
        assert result.uncertainty_reports == {}
        assert not result.abstention_triggered
