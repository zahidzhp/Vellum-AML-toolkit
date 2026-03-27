"""Hierarchical YAML config system with mode overlays and CLI overrides."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from aml_toolkit.core.enums import OperatingMode

# Default config directory relative to package
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIGS_DIR = _PACKAGE_ROOT / "configs"


class DatasetConfig(BaseModel):
    """Dataset input settings."""

    path: str = ""
    modality_override: str | None = None
    target_column: str = "label"
    group_column: str | None = None
    time_column: str | None = None
    metadata_columns: list[str] = Field(default_factory=list)


class SplittingConfig(BaseModel):
    """Data splitting settings."""

    strategy: str = "STRATIFIED"
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    random_seed: int = 42


class ProfilingConfig(BaseModel):
    """Data profiling thresholds."""

    imbalance_ratio_warning: float = 5.0
    imbalance_ratio_severe: float = 20.0
    duplicate_check_enabled: bool = True
    ood_shift_enabled: bool = True


class ProbesConfig(BaseModel):
    """Probe engine settings."""

    enabled_probes: list[str] = Field(
        default_factory=lambda: ["majority", "stratified", "logistic", "rf", "xgb"]
    )
    intervention_branches: list[str] = Field(
        default_factory=lambda: ["none", "class_weighting", "oversampling", "undersampling"]
    )
    metric: str = "macro_f1"


class InterventionsConfig(BaseModel):
    """Intervention planner settings."""

    allowed_types: list[str] = Field(
        default_factory=lambda: [
            "CLASS_WEIGHTING",
            "OVERSAMPLING",
            "UNDERSAMPLING",
            "AUGMENTATION",
            "FOCAL_LOSS",
            "THRESHOLDING",
            "CALIBRATION",
        ]
    )
    oversampling_noise_risk_threshold: float = 0.15
    require_calibration_when_imbalanced: bool = True


class CandidatesConfig(BaseModel):
    """Candidate model selection settings."""

    allowed_families: list[str] = Field(
        default_factory=lambda: ["logistic", "rf", "xgb", "mlp"]
    )
    max_candidates: int = 5
    budget_strategy: str = "equal"


class RuntimeDecisionConfig(BaseModel):
    """Runtime decision engine thresholds."""

    min_warmup_epochs_default: int = 5
    min_warmup_epochs_neural: int = 10
    improvement_slope_threshold: float = 0.001
    overfit_gap_limit: float = 0.15
    patience: int = 3


class CalibrationConfig(BaseModel):
    """Calibration settings."""

    enabled_methods: list[str] = Field(
        default_factory=lambda: ["temperature_scaling", "isotonic"]
    )
    primary_objective: str = "ece"


class EnsembleConfig(BaseModel):
    """Ensemble builder settings."""

    enabled_strategies: list[str] = Field(
        default_factory=lambda: ["soft_voting", "weighted_averaging"]
    )
    marginal_gain_threshold: float = 0.01
    max_ensemble_size: int = 3


class ExplainabilityConfig(BaseModel):
    """Explainability settings."""

    tabular_methods: list[str] = Field(
        default_factory=lambda: ["feature_importance", "shap"]
    )
    image_methods: list[str] = Field(default_factory=lambda: ["gradcam"])
    faithfulness_enabled: bool = True


class ReportingConfig(BaseModel):
    """Reporting and output settings."""

    output_dir: str = "outputs"
    formats: list[str] = Field(default_factory=lambda: ["json", "markdown"])
    verbosity: str = "normal"


class ComputeConfig(BaseModel):
    """Compute budget and resource settings."""

    max_training_time_seconds: int = 3600
    memory_limit_gb: float | None = None
    gpu_enabled: bool = True
    resource_abstention_on_oom: bool = True


class ToolkitConfig(BaseModel):
    """Top-level toolkit configuration combining all sections."""

    mode: OperatingMode = OperatingMode.BALANCED
    seed: int = 42
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    splitting: SplittingConfig = Field(default_factory=SplittingConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    probes: ProbesConfig = Field(default_factory=ProbesConfig)
    interventions: InterventionsConfig = Field(default_factory=InterventionsConfig)
    candidates: CandidatesConfig = Field(default_factory=CandidatesConfig)
    runtime_decision: RuntimeDecisionConfig = Field(default_factory=RuntimeDecisionConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_config(
    config_path: Path | None = None,
    mode: OperatingMode | str | None = None,
    overrides: dict[str, Any] | None = None,
    configs_dir: Path = CONFIGS_DIR,
) -> ToolkitConfig:
    """Load toolkit configuration with hierarchical merging.

    Loading order (later overrides earlier):
    1. configs/default.yaml
    2. configs/modes/<mode>.yaml
    3. User-provided config file
    4. Programmatic overrides dict

    Args:
        config_path: Optional path to a user config YAML file.
        mode: Operating mode name or enum. If None, uses default from config.
        overrides: Dict of overrides to apply last.
        configs_dir: Base directory for config files.

    Returns:
        Fully resolved ToolkitConfig instance.
    """
    # Layer 1: defaults
    default_path = configs_dir / "default.yaml"
    merged: dict[str, Any] = {}
    if default_path.exists():
        merged = load_yaml(default_path)

    # Layer 2: mode overlay
    if mode is not None:
        mode_str = mode.value if isinstance(mode, OperatingMode) else mode
    elif "mode" in merged:
        mode_str = merged["mode"]
    else:
        mode_str = OperatingMode.BALANCED.value

    mode_path = configs_dir / "modes" / f"{mode_str.lower()}.yaml"
    if mode_path.exists():
        mode_data = load_yaml(mode_path)
        merged = _deep_merge(merged, mode_data)

    # Layer 3: user config file
    if config_path is not None:
        user_data = load_yaml(config_path)
        merged = _deep_merge(merged, user_data)

    # Layer 4: programmatic overrides
    if overrides is not None:
        merged = _deep_merge(merged, overrides)

    # Ensure mode is set
    merged["mode"] = mode_str.upper()

    return ToolkitConfig.model_validate(merged)
