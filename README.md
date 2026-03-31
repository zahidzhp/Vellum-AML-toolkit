# Autonomous ML Toolkit (AML Toolkit)

A modular, policy-driven machine learning toolkit that autonomously handles classification tasks end-to-end — including data validation, profiling, training, calibration, explainability, and adaptive intelligence across tabular and image data.

Point it at a dataset and it validates, profiles, trains, calibrates, and explains — producing a full audit trail of every decision. The **V2 Adaptive Intelligence Layer** adds uncertainty quantification, diversity-aware ensemble pruning, historical meta-policy, and experiment planning on top of the core pipeline.

---

## What It Does

Given a dataset (tabular CSV, image folder, or embedding matrix), the toolkit autonomously:

1. **Ingests and validates** the data schema, detects modality and task type
2. **Audits split integrity** — catches duplicate leakage, grouped/entity leakage, temporal leakage, and class absence
3. **Profiles data health** — class imbalance severity, duplicates, label conflicts, outliers, and distribution shift
4. **Runs diagnostic probes** — low-cost models that estimate learnability and test intervention sensitivity
5. **Plans interventions** — selects class weighting, resampling, augmentation, or thresholding based on evidence (blocks unsafe interventions like oversampling when label noise is high)
6. **Trains candidate models** with runtime decision-making — warm-up gates prevent premature termination, underperformers are stopped early, and resource failures trigger structured abstention
7. **Calibrates outputs** via temperature scaling or isotonic regression, then optimizes decision thresholds
8. **Builds ensembles** — soft voting, weighted averaging, or diversity-aware greedy selection (V2)
9. **Generates explainability artifacts** — feature importance, SHAP values, GradCAM heatmaps — with faithfulness checks
10. **Produces reports and audit logs** — JSON and Markdown reports, visualizations, and a timestamped audit log of every pipeline event

If the toolkit determines it cannot produce a trustworthy result (leakage detected, all models fail, resource exhaustion, or high uncertainty), it **abstains** with a structured reason rather than producing a misleading output.

---

## Supported Inputs

| Modality | Format | Example |
|----------|--------|---------|
| **Tabular** | CSV file with feature columns and a target column | `data.csv` with columns `f1, f2, ..., label` |
| **Image** | Folder-per-class directory structure | `images/cat/*.jpg`, `images/dog/*.jpg` |
| **Embedding** | NumPy `.npz` with `embeddings` and `labels` arrays | Pre-computed CLIP or ViT embeddings |

Task types: binary classification, multiclass classification, multilabel classification.

### Image Classification

The toolkit supports image classification via folder-per-class directory structure:

```
images/
  cat/
    img_001.jpg
    img_002.jpg
  dog/
    img_003.jpg
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.

**How it works:**

1. **Auto-detection**: Detects image modality when the dataset path is a directory
2. **Feature extraction**: Images are converted to embeddings using a configurable pretrained backbone (default: ResNet18)
3. **CNN training** (aggressive mode): Transfer learning with any torchvision ResNet — freezes backbone, trains head, then fine-tunes last residual block
4. **ViT training** (aggressive mode): Transfer learning with any timm ViT variant
5. **GradCAM heatmaps**: Hook-based Grad-CAM for CNN models

**Configurable backbones:**

```yaml
candidates:
  cnn_backbone: resnet50              # Any torchvision ResNet variant
  vit_backbone: vit_base_patch16_224  # Any timm ViT variant
  feature_extractor_backbone: resnet18
```

**Mode availability:**

| Model Family | balanced | conservative | aggressive | interpretable |
|-------------|----------|-------------|------------|---------------|
| embedding_head | Yes | Yes | Yes | Yes |
| cnn | No | No | Yes | No |
| vit | No | No | Yes | No |

---

## V2 Adaptive Intelligence Layer

The V2 layer sits on top of the core pipeline and adds four opt-in adaptive capabilities. All features are **disabled by default**, modality-agnostic (tabular and image), and degrade gracefully — a V2 failure never aborts the pipeline.

### Run History Store

Persists a summary of each completed run to a JSONL file. Future runs query this store to find similar past datasets using **cosine similarity on a numerical feature vector** (not coarse categorical buckets).

```yaml
advanced:
  run_history:
    enabled: true
    store_path: ~/.aml_toolkit/run_history.jsonl
    max_records: 1000
```

Dataset signatures encode: log(n_samples), log(n_features), imbalance ratio, missingness %, OOD shift score, label noise score, and derived flags — all normalized to [0, 1] for cosine similarity.

### Uncertainty Quantification

Estimates predictive uncertainty on calibrated probabilities (zero extra inference — reuses `cal_report.plot_data`). Three methods:

- **Entropy**: normalized Shannon entropy H(p) / log(K)
- **Margin**: 1 − (p_max − p_2nd_max)
- **Conformal prediction sets**: mathematically guaranteed coverage (LAC algorithm, numpy-only)

```yaml
advanced:
  uncertainty:
    enabled: true
    methods: [entropy, margin]
    abstain_if_above: 0.8         # Trigger HIGH_UNCERTAINTY abstention
    use_calibrated_proba: true    # Use cal_report.plot_data — zero extra inference
    conformal_enabled: true       # Prediction sets with coverage guarantee
    conformal_coverage: 0.9       # P(y ∈ C(x)) ≥ 0.9
```

When `conformal_enabled: true`, each sample gets a **prediction set** C(x) with a provable guarantee that the true label is included at least 90% of the time. Smaller sets = more confident model.

### Dynamic Ensemble (Diversity-Aware)

Replaces soft voting and weighted averaging with **forward greedy pruning**. Scores candidates by:

```
score = F1_gain + λ × pairwise_disagreement
```

Clones (disagreement < `diversity_threshold`) are rejected even if their individual F1 is high. This prevents redundant models from diluting ensemble diversity.

```yaml
advanced:
  dynamic_ensemble:
    enabled: true
    diversity_threshold: 0.05   # Min pairwise disagreement to add a model
    max_members: 4
    use_uncertainty_weights: false
```

EnsembleReport gains `diversity_score` and `ambiguity_decomposition` (bias/variance/diversity/error breakdown).

### Meta-Policy Engine

Before training, queries run history to **reorder candidates** and **allocate compute budgets** based on which families worked best on similar datasets:

```
score = Σ (cosine_sim × recency_decay^days_ago × family_won)
budget_fraction = softmax(scores)  # sums to 1.0
```

```yaml
advanced:
  meta_policy:
    enabled: true
    compute_budget_aware: true  # Allocate compute fractions per candidate
    recency_decay: 0.9          # Weight recent history more
    similarity_method: cosine
```

Result: promising families get more training time; families that never won on similar data get less.

### Experiment Planner

After each run, generates actionable proposals for the next experiment using a rule engine (always runs) and an optional LLM enhancement (Claude API, opt-in):

```yaml
advanced:
  agentic_planner:
    enabled: true
    mode: propose_only           # propose_only (safe) or auto_apply
    max_suggestions: 3
    llm_enhanced: false          # Set true + ANTHROPIC_API_KEY to enable
    track_proposal_outcomes: true
```

Built-in rules cover: severe imbalance → class weighting, high ECE → reduce candidates, OOD shift → conservative mode, small image dataset → augmentation + ResNet18 preference, high uncertainty → k-fold cross-validation.

LLM proposals can only **add** suggestions — they never override the rule engine, and all proposals are constrained by user config.

### Efficiency: Zero Redundant Inference

All V2 features reuse already-computed artifacts:

```
Calibration stage → proba_before, proba_after stored in cal_report.plot_data
                              │
          ┌───────────────────┼────────────────────┐
          ▼                   ▼                    ▼
UncertaintyEstimator   EnsemblePruner      ConformalPredictor
(uses proba_after)     (uses proba_after)  (uses proba_after)
```

For expensive CNN/ViT inference on image datasets, the calibration pass is the only extra pass. All V2 analysis is free after that.

### Profile-Based Presets

Four ready-to-use profiles in `configs/profiles/`:

| Profile | V2 Features |
|---------|-------------|
| `conservative.yaml` | All V2 off |
| `balanced.yaml` | Run history + uncertainty |
| `advanced.yaml` | + Conformal + dynamic ensemble + meta-policy |
| `research.yaml` | All V2 on, k-fold uncertainty, LLM-ready |

```bash
aml-toolkit run data.csv --config configs/profiles/advanced.yaml
```

---

## Installation

Requires Python 3.11 or later.

```bash
git clone <repository-url>
cd Vellum

python3.11 -m venv .venv
source .venv/bin/activate

pip install -e .
pip install -e ".[dev]"   # for running tests
```

---

## Quick Start

```bash
# Run the full pipeline (V1 behavior — all V2 off)
aml-toolkit run data.csv

# Validate dataset without training
aml-toolkit validate data.csv

# Run with a config file
aml-toolkit run data.csv --config my_config.yaml

# Image classification (embedding_head, balanced)
aml-toolkit run images/

# Image with CNN transfer learning (aggressive)
aml-toolkit run images/ --mode aggressive

# Run with all V2 features enabled
aml-toolkit run data.csv --config configs/profiles/research.yaml
```

## CLI Reference

```
aml-toolkit run <DATASET> [OPTIONS]

Arguments:
  DATASET              Path to the input dataset (CSV or directory)

Options:
  -c, --config PATH      Path to a YAML config file
  -m, --mode TEXT        Operating mode: conservative, balanced, aggressive, interpretable
  -o, --output-dir TEXT  Override output directory
  --seed INTEGER         Random seed override
  -v, --verbose          Enable verbose (DEBUG-level) logging
  --help                 Show help and exit

aml-toolkit validate <DATASET> [OPTIONS]

Arguments:
  DATASET              Path to the input dataset

Options:
  -c, --config PATH    Path to a YAML config file
  --help               Show help and exit
```

---

## Configuration

Configuration loads in layers (later overrides earlier):

1. `configs/default.yaml` — baseline defaults
2. `configs/modes/<mode>.yaml` — mode-specific overrides
3. Your config file (`--config`)
4. CLI arguments (`--seed`, `--output-dir`, etc.)

### Operating Modes

| Mode | Description |
|------|-------------|
| **balanced** (default) | Balances thoroughness with compute efficiency |
| **conservative** | Tighter safety thresholds, fewer candidates, strict overfit limits |
| **aggressive** | Wider candidate pools (CNN, ViT), longer budgets, relaxed thresholds |
| **interpretable** | Restricts to inherently interpretable models |

### Core Configuration Sections

```yaml
# Dataset settings
dataset:
  target_column: label
  group_column: patient_id    # Optional: grouped splitting
  time_column: date           # Optional: temporal splitting
  modality_override: TABULAR  # Optional: force modality

# Splitting
splitting:
  strategy: STRATIFIED        # STRATIFIED, GROUPED, TEMPORAL, PROVIDED
  test_ratio: 0.2
  val_ratio: 0.1
  random_seed: 42

# Data profiling
profiling:
  imbalance_ratio_warning: 5.0
  imbalance_ratio_severe: 20.0
  duplicate_check_enabled: true
  ood_shift_enabled: true

# Probe engine
probes:
  enabled_probes: [majority, stratified, logistic, rf, xgb]
  intervention_branches: [none, class_weighting, oversampling, undersampling]
  metric: macro_f1

# Interventions
interventions:
  allowed_types:
    - CLASS_WEIGHTING
    - OVERSAMPLING
    - UNDERSAMPLING
    - AUGMENTATION
    - FOCAL_LOSS
    - THRESHOLDING
    - CALIBRATION
  oversampling_noise_risk_threshold: 0.15
  require_calibration_when_imbalanced: true

# Candidates
candidates:
  allowed_families: [logistic, rf, xgb, mlp, embedding_head]
  max_candidates: 5
  budget_strategy: equal
  cnn_backbone: resnet18
  vit_backbone: vit_small_patch16_224
  feature_extractor_backbone: resnet18

# Runtime decisions
runtime_decision:
  min_warmup_epochs_default: 5
  min_warmup_epochs_neural: 10
  improvement_slope_threshold: 0.001
  overfit_gap_limit: 0.15
  patience: 3

# Calibration
calibration:
  enabled_methods: [temperature_scaling, isotonic]
  primary_objective: ece     # ece or brier

# Ensemble
ensemble:
  enabled_strategies: [soft_voting, weighted_averaging]
  marginal_gain_threshold: 0.01
  max_ensemble_size: 3

# Explainability
explainability:
  tabular_methods: [feature_importance, shap]
  image_methods: [gradcam]
  faithfulness_enabled: true

# Reporting
reporting:
  output_dir: outputs
  formats: [json, markdown]
  verbosity: normal

# Compute budget
compute:
  max_training_time_seconds: 3600
  memory_limit_gb: null
  gpu_enabled: true
  resource_abstention_on_oom: true
```

### V2 Advanced Configuration

```yaml
advanced:
  # Persist run history for future similarity-based recommendations
  run_history:
    enabled: true
    store_path: ~/.aml_toolkit/run_history.jsonl
    max_records: 1000

  # Uncertainty estimation (uses calibrated probabilities — zero extra inference)
  uncertainty:
    enabled: true
    methods: [entropy, margin]
    aggregation: mean
    abstain_if_above: 0.8       # HIGH_UNCERTAINTY abstention threshold
    use_calibrated_proba: true
    conformal_enabled: true     # Prediction sets with coverage guarantee
    conformal_coverage: 0.9     # 1-α coverage
    use_cross_val: false        # k-fold uncertainty for small datasets
    cross_val_folds: 5

  # Diversity-aware ensemble selection
  dynamic_ensemble:
    enabled: true
    allowed_modes: [greedy_diverse]
    max_members: 4
    diversity_threshold: 0.05   # Minimum pairwise disagreement to admit a model
    use_uncertainty_weights: false

  # History-based candidate ordering and compute budget allocation
  meta_policy:
    enabled: true
    exploration_weight: 0.3     # Fraction of budget allocated uniformly
    compute_budget_aware: true
    similarity_method: cosine
    recency_decay: 0.9          # Per-day decay for historical records

  # Experiment proposal engine
  agentic_planner:
    enabled: true
    mode: propose_only          # propose_only is safe; auto_apply requires caution
    max_suggestions: 3
    llm_enhanced: false         # Requires ANTHROPIC_API_KEY
    track_proposal_outcomes: true
```

### Example: Medical Imaging (Aggressive + V2)

```yaml
mode: AGGRESSIVE

candidates:
  allowed_families: [embedding_head, cnn]
  max_candidates: 2
  cnn_backbone: resnet50

compute:
  max_training_time_seconds: 1200
  gpu_enabled: true

advanced:
  uncertainty:
    enabled: true
    conformal_enabled: true
    conformal_coverage: 0.95    # Higher coverage for medical use
  dynamic_ensemble:
    enabled: true
    diversity_threshold: 0.05
```

### Example: Tabular with Grouped Splitting

```yaml
mode: CONSERVATIVE

dataset:
  target_column: diagnosis
  group_column: patient_id

splitting:
  strategy: GROUPED

candidates:
  allowed_families: [logistic, rf]
  max_candidates: 2

compute:
  max_training_time_seconds: 600
  gpu_enabled: false
```

---

## Output Structure

```
outputs/
  20260328_143022_a1b2c3/
    intake/
    audit/
    profiling/
    probes/
    interventions/
    candidates/
    runtime/
    calibration/
    ensemble/
    explainability/
      heatmaps/
    reporting/
      final_report.json
      final_report.md
      plots/                     # Learning curves, ROC, PR, calibration diagrams
    logs/
      audit_log.json
```

### Final Report Fields

| Field | Description |
|-------|-------------|
| `run_id` | Unique run identifier |
| `final_status` | `COMPLETED` or `ABSTAINED` |
| `abstention_reason` | Structured reason (see table below) |
| `final_recommendation` | Recommended model or abstention explanation |
| `stages_completed` | Ordered list of completed stages |
| `dataset_summary` | Modality, task type, sample counts |
| `split_audit_summary` | Leakage check results |
| `profile_summary` | Data health flags and statistics |
| `probe_summary` | Probe model performance comparison |
| `intervention_summary` | Selected and rejected interventions with rationale |
| `candidate_summary` | Candidate models and their configurations |
| `runtime_decision_summary` | Per-candidate training decisions |
| `calibration_summary` | Calibration method, ECE before/after, optimized threshold |
| `ensemble_summary` | Strategy, members, diversity score (V2), ambiguity decomposition (V2) |
| `explainability_summary` | Methods used, faithfulness results |
| `plot_paths` | Paths to generated visualizations |
| `warnings` | Any issues encountered during the run |

### Abstention Reasons

| Reason | Trigger |
|--------|---------|
| `LEAKAGE_BLOCKED` | Split audit found train/test data contamination |
| `SCHEMA_INVALID` | Input data failed schema validation |
| `RESOURCE_EXHAUSTED` | OOM or training time budget exceeded |
| `NO_ROBUST_MODEL` | No candidate model passed quality thresholds |
| `CRITICAL_FAILURE` | Unexpected error during pipeline execution |
| `HIGH_UNCERTAINTY` | Mean predictive uncertainty exceeded configured threshold (V2) |

---

## Pipeline Stages

```
INIT → DATA_VALIDATED → PROFILED → PROBED → INTERVENTION_SELECTED
     → TRAINING_ACTIVE → MODEL_SELECTED → CALIBRATED → ENSEMBLED
     → EXPLAINED → COMPLETED
```

The pipeline transitions to `ABSTAINED` from any stage if a trustworthy result is not achievable.

---

## Programmatic Usage

### Basic

```python
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.orchestration.orchestrator import PipelineOrchestrator

config = ToolkitConfig(
    dataset={"path": "data.csv", "target_column": "label"},
    candidates={"allowed_families": ["logistic", "rf"], "max_candidates": 2},
)

report = PipelineOrchestrator(config).run("data.csv")
print(report.final_status)          # PipelineStage.COMPLETED
print(report.final_recommendation)  # "Recommended model: rf_001"
```

### With V2 Features

```python
from aml_toolkit.core.config import load_config
from aml_toolkit.orchestration.orchestrator import PipelineOrchestrator

# Load a pre-built profile
config = load_config(config_path="configs/profiles/advanced.yaml")
report = PipelineOrchestrator(config).run("data.csv")

# Or enable specific V2 features programmatically
config = load_config(overrides={
    "advanced": {
        "uncertainty": {"enabled": True, "conformal_enabled": True},
        "dynamic_ensemble": {"enabled": True, "diversity_threshold": 0.05},
        "meta_policy": {"enabled": True},
    }
})
```

### Image Classification

```python
# Balanced: embedding_head (logistic on ResNet features)
config = ToolkitConfig(
    dataset={"path": "images/"},
    candidates={"allowed_families": ["embedding_head"], "max_candidates": 1},
    compute={"gpu_enabled": False},
)
report = PipelineOrchestrator(config).run("images/")

# Aggressive: CNN transfer learning
config = ToolkitConfig(
    dataset={"path": "images/"},
    candidates={
        "allowed_families": ["cnn", "embedding_head"],
        "max_candidates": 2,
        "cnn_backbone": "resnet50",
    },
    compute={"gpu_enabled": True},
)
report = PipelineOrchestrator(config).run("images/")
```

### V2 Modules Standalone

```python
from aml_toolkit.uncertainty.estimator import UncertaintyEstimator
from aml_toolkit.uncertainty.conformal import SplitConformalPredictor
from aml_toolkit.ensemble.greedy_diverse import GreedyDiverseEnsemble
from aml_toolkit.core.config import UncertaintyConfig, DynamicEnsembleConfig

# Conformal prediction sets
predictor = SplitConformalPredictor(coverage=0.9)
predictor.fit(proba_cal, y_cal)
sets = predictor.predict_sets(proba_test)   # list[list[int]]
coverage = predictor.empirical_coverage(proba_test, y_test)

# Uncertainty estimation
estimator = UncertaintyEstimator(UncertaintyConfig(
    enabled=True, methods=["entropy", "margin"], conformal_enabled=True
))
report = estimator.estimate("my_model", proba, y_val=y_val)
print(report.mean_uncertainty, report.mean_prediction_set_size)

# Diversity-aware ensemble selection
ensemble = GreedyDiverseEnsemble(DynamicEnsembleConfig(max_members=3, diversity_threshold=0.05))
report = ensemble.select({"rf": proba_rf, "xgb": proba_xgb, "lr": proba_lr}, y_val)
print(report.member_ids, report.diversity_score)
```

### Individual Pipeline Stages

```python
from aml_toolkit.intake.dataset_intake_manager import run_intake
from aml_toolkit.profiling.profiler_engine import run_profiling
from aml_toolkit.audit.split_auditor import run_split_audit

config = ToolkitConfig(dataset={"path": "data.csv", "target_column": "label"})
intake = run_intake(config)

print(intake.manifest.modality)    # ModalityType.TABULAR
print(intake.manifest.task_type)   # TaskType.BINARY

audit = run_split_audit(
    data=intake.data,
    manifest=intake.manifest,
    split=intake.split_result,
    config=config,
)
print(audit.passed)                # True if no leakage detected

profile = run_profiling(
    data=intake.data,
    manifest=intake.manifest,
    split=intake.split_result,
    config=config,
)
print(profile.risk_flags)          # [RiskFlag.CLASS_IMBALANCE, ...]
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=aml_toolkit --cov-report=term-missing
```

The test suite contains **598 tests** across 32 files:

- **Unit tests** (28 files) — per-module contract verification including V2 modules (uncertainty, ensemble diversity, meta-policy, run history, experiment planner, config expansion)
- **Integration tests** (4 files) — multi-stage data flow, V2 sanity checks (all-off regression, constraint enforcement, graceful degradation), and benchmark tests (coverage guarantees, diversity vs single model)

---

## Project Structure

```
src/aml_toolkit/
  core/             Config, enums, exceptions, logging, seeds, paths
  artifacts/        Typed Pydantic models for every pipeline stage output
  interfaces/       Protocols/ABCs for all pluggable components
  intake/           Schema parsing, modality/task detection, split building
  audit/            Split auditing, leakage checks, augmentation guard
  profiling/        Class distribution, duplicates, label conflicts, OOD shift
  probes/           Baseline and shallow diagnostic models
  interventions/    Weighting, resampling, augmentation, thresholding planning
  models/           Model registry + adapters (logistic, RF, XGBoost, MLP, CNN, ViT)
  runtime/          Training executor, warm-up policies, runtime decision engine
  calibration/      Temperature scaling, isotonic regression, threshold optimization
  ensemble/         Soft voting, weighted averaging, diversity-aware greedy pruning (V2)
  explainability/   Feature importance, SHAP, GradCAM, confusion heatmaps, faithfulness
  reporting/        JSON and Markdown report builders, plot generation
  orchestration/    Pipeline orchestrator, state machine, audit logger
  api/              CLI entrypoint (Typer)
  utils/            Serialization, resource guard, image feature extraction

  — V2 Adaptive Intelligence —
  adaptive/         AdaptiveIntelligenceCoordinator (unified V2 entry point)
  history/          Run history store + dataset signature builder
  uncertainty/      Entropy, margin, split-conformal prediction sets
  meta_policy/      Cosine-similarity history-based candidate ordering
  planning/         Rule engine + experiment planner (optional LLM)

configs/
  default.yaml
  modes/            conservative, balanced, aggressive, interpretable
  profiles/         conservative, balanced, advanced, research (V2 presets)

knowledge/          Design docs, planning artifacts (gitignored)
tests/
  unit/             28 unit test files
  integration/      4 integration test files
```

---

## License

MIT
