"""Tests for model registry, adapter scaffolding, and candidate portfolio builder."""

import numpy as np
import pytest

from aml_toolkit.artifacts import CandidatePortfolio, InterventionEntry, InterventionPlan
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import InterventionType, ModalityType
from aml_toolkit.interfaces.model_metadata import ModelFamilyMetadata
from aml_toolkit.models.registry import build_candidate_portfolio, create_default_registry


class TestDefaultRegistry:
    def test_registry_has_all_families(self):
        registry = create_default_registry()
        families = registry.list_families()
        assert "logistic" in families
        assert "rf" in families
        assert "xgb" in families
        assert "mlp" in families
        assert "embedding_head" in families
        assert "cnn" in families
        assert "vit" in families

    def test_tabular_families(self):
        registry = create_default_registry()
        tabular = registry.list_families_for_modality(ModalityType.TABULAR)
        assert "logistic" in tabular
        assert "rf" in tabular
        assert "xgb" in tabular
        assert "mlp" in tabular
        assert "cnn" not in tabular

    def test_image_families(self):
        registry = create_default_registry()
        image = registry.list_families_for_modality(ModalityType.IMAGE)
        assert "cnn" in image
        assert "vit" in image
        assert "embedding_head" in image
        assert "logistic" not in image

    def test_embedding_families(self):
        registry = create_default_registry()
        emb = registry.list_families_for_modality(ModalityType.EMBEDDING)
        assert "logistic" in emb
        assert "rf" in emb
        assert "embedding_head" in emb

    def test_metadata_neural_flag(self):
        registry = create_default_registry()
        assert registry.get_metadata("logistic").is_neural is False
        assert registry.get_metadata("mlp").is_neural is True
        assert registry.get_metadata("cnn").is_neural is True

    def test_metadata_gradcam_support(self):
        registry = create_default_registry()
        assert registry.get_metadata("cnn").supports_gradcam is True
        assert registry.get_metadata("logistic").supports_gradcam is False


class TestAdapterInstantiation:
    def test_logistic_adapter_interface(self):
        from aml_toolkit.models.tabular.logistic_adapter import LogisticAdapter

        adapter = LogisticAdapter()
        assert adapter.get_model_family() == "logistic"
        assert adapter.is_probabilistic() is True

    def test_rf_adapter_interface(self):
        from aml_toolkit.models.tabular.rf_adapter import RandomForestAdapter

        adapter = RandomForestAdapter()
        assert adapter.get_model_family() == "rf"

    def test_mlp_adapter_interface(self):
        from aml_toolkit.models.tabular.mlp_adapter import MLPAdapter

        adapter = MLPAdapter()
        assert adapter.get_model_family() == "mlp"

    def test_cnn_adapter_properties(self):
        from aml_toolkit.models.image.cnn_adapter import CNNAdapter

        adapter = CNNAdapter()
        assert adapter.get_model_family() == "cnn"
        assert adapter.is_probabilistic() is True
        assert adapter._supports_gradcam is True

    def test_vit_adapter_properties(self):
        from aml_toolkit.models.image.vit_adapter import ViTAdapter

        adapter = ViTAdapter()
        assert adapter.get_model_family() == "vit"
        assert adapter.is_probabilistic() is True
        assert adapter._supports_gradcam is False

    def test_logistic_adapter_fit_predict(self):
        from aml_toolkit.models.tabular.logistic_adapter import LogisticAdapter

        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        y_train = (X_train[:, 0] > 0).astype(int)
        X_val = np.random.randn(20, 3)
        y_val = (X_val[:, 0] > 0).astype(int)

        adapter = LogisticAdapter()
        adapter.fit(X_train, y_train, X_val, y_val, ToolkitConfig())
        preds = adapter.predict(X_val)
        assert len(preds) == 20
        proba = adapter.predict_proba(X_val)
        assert proba.shape == (20, 2)
        trace = adapter.get_training_trace()
        assert "val_macro_f1" in trace


class TestCandidatePortfolioBuilder:
    def test_tabular_portfolio(self):
        config = ToolkitConfig(
            candidates={"allowed_families": ["logistic", "rf", "xgb"], "max_candidates": 5}
        )
        portfolio = build_candidate_portfolio(ModalityType.TABULAR, config)
        assert isinstance(portfolio, CandidatePortfolio)
        assert len(portfolio.candidate_models) == 3
        families = [c.model_family for c in portfolio.candidate_models]
        assert "logistic" in families
        assert "rf" in families
        assert "xgb" in families

    def test_config_whitelist_filters(self):
        config = ToolkitConfig(
            candidates={"allowed_families": ["logistic"], "max_candidates": 5}
        )
        portfolio = build_candidate_portfolio(ModalityType.TABULAR, config)
        assert len(portfolio.candidate_models) == 1
        assert portfolio.candidate_models[0].model_family == "logistic"

    def test_max_candidates_caps(self):
        config = ToolkitConfig(
            candidates={"allowed_families": ["logistic", "rf", "xgb", "mlp"], "max_candidates": 2}
        )
        portfolio = build_candidate_portfolio(ModalityType.TABULAR, config)
        assert len(portfolio.candidate_models) == 2

    def test_rejection_reasons_recorded(self):
        config = ToolkitConfig(
            candidates={"allowed_families": ["logistic"], "max_candidates": 5}
        )
        portfolio = build_candidate_portfolio(ModalityType.TABULAR, config)
        # rf, xgb, mlp should be rejected (not in allowed_families)
        assert len(portfolio.rejection_reasons) > 0

    def test_warmup_rules_populated(self):
        config = ToolkitConfig(
            candidates={"allowed_families": ["logistic", "mlp"], "max_candidates": 5}
        )
        portfolio = build_candidate_portfolio(ModalityType.TABULAR, config)
        assert "logistic" in portfolio.warmup_rules
        assert "mlp" in portfolio.warmup_rules
        # MLP is neural, should have higher warmup
        assert portfolio.warmup_rules["mlp"] >= portfolio.warmup_rules["logistic"]

    def test_budget_allocations_sum_to_one(self):
        config = ToolkitConfig(
            candidates={"allowed_families": ["logistic", "rf", "xgb"], "max_candidates": 5}
        )
        portfolio = build_candidate_portfolio(ModalityType.TABULAR, config)
        total_budget = sum(portfolio.budget_allocations.values())
        assert abs(total_budget - 1.0) < 0.001

    def test_image_portfolio(self):
        config = ToolkitConfig(
            candidates={"allowed_families": ["cnn", "vit", "embedding_head"], "max_candidates": 5}
        )
        portfolio = build_candidate_portfolio(ModalityType.IMAGE, config)
        families = [c.model_family for c in portfolio.candidate_models]
        assert "cnn" in families or "vit" in families or "embedding_head" in families

    def test_embedding_portfolio(self):
        config = ToolkitConfig(
            candidates={"allowed_families": ["logistic", "rf", "embedding_head"], "max_candidates": 5}
        )
        portfolio = build_candidate_portfolio(ModalityType.EMBEDDING, config)
        assert len(portfolio.candidate_models) >= 1

    def test_portfolio_serializes(self, tmp_path):
        config = ToolkitConfig(
            candidates={"allowed_families": ["logistic", "rf"], "max_candidates": 5}
        )
        portfolio = build_candidate_portfolio(ModalityType.TABULAR, config)

        from aml_toolkit.utils.serialization import save_artifact_json, load_artifact_json

        path = tmp_path / "portfolio.json"
        save_artifact_json(portfolio, path)
        loaded = load_artifact_json(CandidatePortfolio, path)
        assert len(loaded.candidate_models) == len(portfolio.candidate_models)

    def test_empty_allowed_families(self):
        config = ToolkitConfig(
            candidates={"allowed_families": ["nonexistent"], "max_candidates": 5}
        )
        portfolio = build_candidate_portfolio(ModalityType.TABULAR, config)
        assert len(portfolio.candidate_models) == 0
