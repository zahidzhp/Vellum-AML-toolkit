"""Default model registry populated with all built-in adapters, plus candidate portfolio builder."""

import logging
from typing import Any

from aml_toolkit.artifacts import CandidateEntry, CandidatePortfolio, InterventionPlan
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import InterventionType, ModalityType
from aml_toolkit.interfaces.model_metadata import ModelFamilyMetadata, ModelRegistry

logger = logging.getLogger("aml_toolkit")


def create_default_registry() -> ModelRegistry:
    """Create and populate the default model registry with all built-in adapters."""
    from aml_toolkit.models.image.cnn_adapter import CNNAdapter
    from aml_toolkit.models.image.embedding_head_adapter import EmbeddingHeadAdapter
    from aml_toolkit.models.image.vit_adapter import ViTAdapter
    from aml_toolkit.models.tabular.logistic_adapter import LogisticAdapter
    from aml_toolkit.models.tabular.mlp_adapter import MLPAdapter
    from aml_toolkit.models.tabular.rf_adapter import RandomForestAdapter
    from aml_toolkit.models.tabular.xgb_adapter import XGBAdapter

    registry = ModelRegistry()

    registry.register(
        "logistic",
        LogisticAdapter,
        ModelFamilyMetadata(
            family_name="logistic",
            display_name="Logistic Regression",
            supported_modalities=[ModalityType.TABULAR, ModalityType.EMBEDDING],
            is_neural=False,
            default_warmup_epochs=1,
            is_probabilistic=True,
        ),
    )

    registry.register(
        "rf",
        RandomForestAdapter,
        ModelFamilyMetadata(
            family_name="rf",
            display_name="Random Forest",
            supported_modalities=[ModalityType.TABULAR, ModalityType.EMBEDDING],
            is_neural=False,
            default_warmup_epochs=1,
            is_probabilistic=True,
        ),
    )

    registry.register(
        "xgb",
        XGBAdapter,
        ModelFamilyMetadata(
            family_name="xgb",
            display_name="XGBoost",
            supported_modalities=[ModalityType.TABULAR, ModalityType.EMBEDDING],
            is_neural=False,
            default_warmup_epochs=1,
            is_probabilistic=True,
        ),
    )

    registry.register(
        "mlp",
        MLPAdapter,
        ModelFamilyMetadata(
            family_name="mlp",
            display_name="MLP (sklearn)",
            supported_modalities=[ModalityType.TABULAR, ModalityType.EMBEDDING],
            is_neural=True,
            default_warmup_epochs=10,
            is_probabilistic=True,
        ),
    )

    registry.register(
        "embedding_head",
        EmbeddingHeadAdapter,
        ModelFamilyMetadata(
            family_name="embedding_head",
            display_name="Embedding Head (Logistic)",
            supported_modalities=[ModalityType.EMBEDDING, ModalityType.IMAGE],
            is_neural=False,
            default_warmup_epochs=1,
            is_probabilistic=True,
        ),
    )

    registry.register(
        "cnn",
        CNNAdapter,
        ModelFamilyMetadata(
            family_name="cnn",
            display_name="CNN (pretrained)",
            supported_modalities=[ModalityType.IMAGE],
            is_neural=True,
            default_warmup_epochs=10,
            is_probabilistic=True,
            supports_gradcam=True,
        ),
    )

    registry.register(
        "vit",
        ViTAdapter,
        ModelFamilyMetadata(
            family_name="vit",
            display_name="ViT (pretrained)",
            supported_modalities=[ModalityType.IMAGE],
            is_neural=True,
            default_warmup_epochs=10,
            is_probabilistic=True,
        ),
    )

    return registry


def build_candidate_portfolio(
    modality: ModalityType,
    config: ToolkitConfig,
    intervention_plan: InterventionPlan | None = None,
    registry: ModelRegistry | None = None,
) -> CandidatePortfolio:
    """Select candidate models and build a portfolio.

    Selection logic:
    1. Filter registry to families that support the modality.
    2. Filter to families allowed by config.
    3. Cap at max_candidates.
    4. Assign warm-up epochs from registry metadata.
    5. Assign budget allocations based on config strategy.

    Args:
        modality: The dataset modality.
        config: Toolkit configuration.
        intervention_plan: Optional intervention plan (informs class_weight usage).
        registry: Model registry. If None, uses default.

    Returns:
        CandidatePortfolio with selected candidates.
    """
    if registry is None:
        registry = create_default_registry()

    allowed_families = set(config.candidates.allowed_families)
    max_candidates = config.candidates.max_candidates

    # Families that support this modality AND are allowed
    available = registry.list_families_for_modality(modality)
    eligible = [f for f in available if f in allowed_families]

    if not eligible:
        logger.warning(
            f"No eligible candidate families for modality {modality.value} "
            f"with allowed_families={list(allowed_families)}."
        )

    # Cap at max
    selected = eligible[:max_candidates]

    # Determine if class_weight should be passed
    use_class_weight = False
    if intervention_plan:
        for entry in intervention_plan.selected_interventions:
            if entry.intervention_type == InterventionType.CLASS_WEIGHTING:
                use_class_weight = True
                break

    # Build entries
    candidates: list[CandidateEntry] = []
    warmup_rules: dict[str, int] = {}
    budget_allocations: dict[str, float] = {}
    rejection_reasons: dict[str, str] = {}

    for family_name in selected:
        meta = registry.get_metadata(family_name)
        warmup = meta.default_warmup_epochs
        if meta.is_neural:
            warmup = max(warmup, config.runtime_decision.min_warmup_epochs_neural)
        else:
            warmup = max(warmup, config.runtime_decision.min_warmup_epochs_default)

        # Resolve backbone name for families that have one
        backbone: str | None = None
        if family_name == "cnn":
            backbone = config.candidates.cnn_backbone
        elif family_name == "vit":
            backbone = config.candidates.vit_backbone

        # Build candidate ID with backbone if available
        if backbone:
            candidate_id = f"{family_name}_{backbone}_001"
        else:
            candidate_id = f"{family_name}_001"

        candidates.append(
            CandidateEntry(
                candidate_id=candidate_id,
                model_family=family_name,
                model_name=meta.display_name,
                backbone=backbone,
                warmup_epochs=warmup,
                budget_allocation=1.0,  # will be normalized below
            )
        )
        warmup_rules[family_name] = warmup

    # Normalize budgets
    if candidates and config.candidates.budget_strategy == "equal":
        per_candidate = 1.0 / len(candidates)
        for c in candidates:
            c.budget_allocation = per_candidate
            budget_allocations[c.model_family] = per_candidate
    elif candidates:
        # Proportional or other — default to equal for now
        per_candidate = 1.0 / len(candidates)
        for c in candidates:
            c.budget_allocation = per_candidate
            budget_allocations[c.model_family] = per_candidate

    # Record rejected families
    for family_name in available:
        if family_name not in selected:
            if family_name not in allowed_families:
                rejection_reasons[family_name] = "Not in allowed_families config."
            else:
                rejection_reasons[family_name] = f"Exceeded max_candidates ({max_candidates})."

    return CandidatePortfolio(
        candidate_models=candidates,
        selected_families=[c.model_family for c in candidates],
        budget_allocations=budget_allocations,
        warmup_rules=warmup_rules,
        rejection_reasons=rejection_reasons,
    )
