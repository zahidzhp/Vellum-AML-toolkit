# Autonomous ML Toolkit (AML Toolkit)

A modular, policy-driven machine learning toolkit that autonomously handles classification tasks end-to-end. Point it at a dataset, and it will validate, profile, train, calibrate, and explain — producing a full audit trail of every decision made.

## What It Does

Given a dataset (tabular CSV, image folder, or embedding matrix), the toolkit autonomously:

1. **Ingests and validates** the data schema, detects modality and task type
2. **Audits split integrity** — catches duplicate leakage, grouped/entity leakage, temporal leakage, and class absence
3. **Profiles data health** — class imbalance severity, duplicates, label conflicts, outliers, and distribution shift
4. **Runs diagnostic probes** — low-cost models that estimate learnability and test intervention sensitivity
5. **Plans interventions** — selects class weighting, resampling, augmentation, or thresholding based on evidence (blocks unsafe interventions like oversampling when label noise is high)
6. **Trains candidate models** with runtime decision-making — warm-up gates prevent premature termination, underperformers are stopped early, and resource failures trigger structured abstention
7. **Calibrates outputs** via temperature scaling or isotonic regression, then optimizes decision thresholds
8. **Builds ensembles** only when the gain over the best single model exceeds a configurable threshold
9. **Generates explainability artifacts** — feature importance, SHAP values, GradCAM heatmaps — with faithfulness checks
10. **Produces reports and audit logs** — JSON and Markdown reports, plus a timestamped audit log of every pipeline event

If the toolkit determines it cannot produce a trustworthy result (leakage detected, all models fail, resource exhaustion), it **abstains** with a structured reason rather than producing a misleading output.

## Supported Inputs

| Modality | Format | Example |
|----------|--------|---------|
| **Tabular** | CSV file with feature columns and a target column | `data.csv` with columns `f1, f2, ..., label` |
| **Image** | Folder-per-class directory structure | `images/cat/*.jpg`, `images/dog/*.jpg` |
| **Embedding** | NumPy `.npz` with `embeddings` and `labels` arrays | Pre-computed CLIP or ViT embeddings |

Task types: binary classification, multiclass classification, multilabel classification.

## Installation

Requires Python 3.11 or later.

```bash
# Clone the repository
git clone <repository-url>
cd Vellum

# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install with dependencies
pip install -e .

# Install dev dependencies (for running tests)
pip install -e ".[dev]"
```

## Quick Start

### Run the full pipeline

```bash
aml-toolkit run data.csv
```

This will:
- Auto-detect that `data.csv` is tabular with a `label` column
- Run all 10 pipeline stages
- Write outputs to `outputs/<run_id>/`
- Print the final status and recommendation

### Validate a dataset without training

```bash
aml-toolkit validate data.csv
```

Runs intake and validation only — useful for checking your data before a full run.

### Specify a config file

```bash
aml-toolkit run data.csv --config my_config.yaml
```

### Override options from the command line

```bash
aml-toolkit run data.csv --mode conservative --seed 99 --output-dir results/ --verbose
```

## CLI Reference

```
aml-toolkit run <DATASET> [OPTIONS]

Arguments:
  DATASET              Path to the input dataset (CSV or directory)

Options:
  -c, --config PATH    Path to a YAML config file
  -m, --mode TEXT      Operating mode: conservative, balanced, aggressive, interpretable
  -o, --output-dir TEXT  Override output directory
  --seed INTEGER       Random seed override
  -v, --verbose        Enable verbose (DEBUG-level) logging
  --help               Show help and exit

aml-toolkit validate <DATASET> [OPTIONS]

Arguments:
  DATASET              Path to the input dataset

Options:
  -c, --config PATH    Path to a YAML config file
  --help               Show help and exit
```

## Configuration

All behavior is controlled through YAML config files. The toolkit loads configuration in layers, where each layer overrides the previous:

1. `configs/default.yaml` — baseline defaults
2. `configs/modes/<mode>.yaml` — mode-specific overrides
3. Your custom config file (`--config`)
4. CLI arguments (`--seed`, `--output-dir`, etc.)

### Operating Modes

| Mode | Description |
|------|-------------|
| **balanced** (default) | Balances thoroughness with compute efficiency |
| **conservative** | Tighter safety thresholds, fewer candidates, longer warm-up, strict overfit limits |
| **aggressive** | Wider candidate pools (including CNNs, ViTs), longer training budgets, relaxed thresholds |
| **interpretable** | Restricts to inherently interpretable models and explanations |

### Configuration Sections

A config file can override any of these sections. Only include the fields you want to change — everything else uses the default.

```yaml
# Dataset settings
dataset:
  target_column: label        # Name of the target/label column
  group_column: patient_id    # Column for grouped splitting (optional)
  time_column: date           # Column for temporal splitting (optional)
  modality_override: TABULAR  # Force a modality instead of auto-detect (optional)

# Splitting strategy
splitting:
  strategy: STRATIFIED         # STRATIFIED, GROUPED, TEMPORAL, or PROVIDED
  test_ratio: 0.2
  val_ratio: 0.1
  random_seed: 42

# Data profiling thresholds
profiling:
  imbalance_ratio_warning: 5.0   # Warn if minority/majority ratio exceeds this
  imbalance_ratio_severe: 20.0   # Flag as severe imbalance above this
  duplicate_check_enabled: true
  ood_shift_enabled: true         # Enable train/test distribution shift detection

# Probe engine
probes:
  enabled_probes:               # Which probe models to run
    - majority
    - stratified
    - logistic
    - rf
    - xgb
  intervention_branches:         # Test these intervention strategies during probing
    - none
    - class_weighting
    - oversampling
    - undersampling
  metric: macro_f1               # Primary metric for probe comparison

# Intervention planner
interventions:
  allowed_types:
    - CLASS_WEIGHTING
    - OVERSAMPLING
    - UNDERSAMPLING
    - AUGMENTATION
    - FOCAL_LOSS
    - THRESHOLDING
    - CALIBRATION
  oversampling_noise_risk_threshold: 0.15  # Block oversampling if label noise exceeds this
  require_calibration_when_imbalanced: true

# Candidate model selection
candidates:
  allowed_families:              # Model families to consider
    - logistic
    - rf
    - xgb
    - mlp
  max_candidates: 5
  budget_strategy: equal         # How to allocate compute across candidates

# Runtime decision engine
runtime_decision:
  min_warmup_epochs_default: 5   # Minimum epochs before stopping (non-neural)
  min_warmup_epochs_neural: 10   # Minimum epochs before stopping (neural)
  improvement_slope_threshold: 0.001  # Stop if improvement slope drops below this
  overfit_gap_limit: 0.15        # Stop if train-val gap exceeds this
  patience: 3                    # Epochs of no improvement before stopping

# Calibration
calibration:
  enabled_methods:
    - temperature_scaling
    - isotonic
  primary_objective: ece         # Optimize for ECE (expected calibration error) or brier

# Ensemble building
ensemble:
  enabled_strategies:
    - soft_voting
    - weighted_averaging
  marginal_gain_threshold: 0.01  # Only ensemble if gain over best single model exceeds this
  max_ensemble_size: 3

# Explainability
explainability:
  tabular_methods:
    - feature_importance
    - shap
  image_methods:
    - gradcam
  faithfulness_enabled: true     # Run faithfulness checks on explanations

# Reporting
reporting:
  output_dir: outputs            # Base output directory
  formats:
    - json
    - markdown
  verbosity: normal

# Compute budget
compute:
  max_training_time_seconds: 3600  # Total training time budget
  memory_limit_gb: null            # Memory limit (null = no limit)
  gpu_enabled: true
  resource_abstention_on_oom: true # Abstain instead of crashing on OOM
```

### Example: Custom Config for Medical Imaging

```yaml
# medical_config.yaml
mode: CONSERVATIVE

dataset:
  target_column: diagnosis
  group_column: patient_id  # Prevent same patient appearing in train and test

splitting:
  strategy: GROUPED

candidates:
  allowed_families:
    - logistic
    - rf
  max_candidates: 2

compute:
  max_training_time_seconds: 600
  gpu_enabled: false
```

```bash
aml-toolkit run patient_data.csv --config medical_config.yaml
```

## Output Structure

Each run creates a timestamped directory under the output path:

```
outputs/
  20260328_143022_a1b2c3/      # <date>_<time>_<hash>
    intake/                     # Raw intake artifacts
    audit/                      # Split audit results
    profiling/                  # Data health profile
    probes/                     # Probe model results
    interventions/              # Intervention plan
    candidates/                 # Candidate model artifacts
    runtime/                    # Runtime decision log
    calibration/                # Calibration results
    ensemble/                   # Ensemble evaluation
    explainability/             # Explanations and heatmaps
      heatmaps/                 # GradCAM or confusion heatmaps
    reporting/
      final_report.json         # Machine-readable report
      final_report.md           # Human-readable report
    logs/
      audit_log.json            # Timestamped audit trail of every pipeline event
```

### Final Report

The JSON report (`final_report.json`) includes:

| Field | Description |
|-------|-------------|
| `run_id` | Unique run identifier |
| `final_status` | `COMPLETED` or `ABSTAINED` |
| `abstention_reason` | Why the pipeline abstained (if applicable) |
| `final_recommendation` | Recommended model or abstention explanation |
| `stages_completed` | Ordered list of completed pipeline stages |
| `dataset_summary` | Modality, task type, sample counts |
| `split_audit_summary` | Leakage check results |
| `profile_summary` | Data health flags and statistics |
| `probe_summary` | Probe model performance comparison |
| `intervention_summary` | Selected and rejected interventions with rationale |
| `candidate_summary` | Candidate models and their configurations |
| `runtime_decision_summary` | Per-candidate training decisions |
| `calibration_summary` | Calibration method, ECE before/after, optimized threshold |
| `ensemble_summary` | Whether ensemble was selected, gain over best single model |
| `explainability_summary` | Explanation methods used, faithfulness results |
| `warnings` | Any issues encountered during the run |

### Audit Log

The audit log (`audit_log.json`) is a timestamped sequence of every pipeline event:

```json
[
  {"timestamp": "2026-03-28T14:30:22Z", "stage": "INIT", "event": "pipeline_start", "detail": {"run_id": "...", "dataset": "data.csv"}},
  {"timestamp": "2026-03-28T14:30:23Z", "stage": "DATA_VALIDATED", "event": "intake_complete", "detail": {"modality": "TABULAR", "task_type": "BINARY"}},
  {"timestamp": "2026-03-28T14:30:23Z", "stage": "DATA_VALIDATED", "event": "audit_passed", "detail": {}},
  ...
]
```

## Pipeline Stages

The pipeline follows a strict stage order enforced by an internal state machine. No stage can be skipped, and the pipeline can transition to `ABSTAINED` from any stage.

```
INIT -> DATA_VALIDATED -> PROFILED -> PROBED -> INTERVENTION_SELECTED
     -> TRAINING_ACTIVE -> MODEL_SELECTED -> CALIBRATED -> ENSEMBLED
     -> EXPLAINED -> COMPLETED
```

At any point, the pipeline may transition to `ABSTAINED` if it determines a trustworthy result is not achievable. Abstention reasons include:

| Reason | Trigger |
|--------|---------|
| `LEAKAGE_BLOCKED` | Split audit found train/test data contamination |
| `SCHEMA_INVALID` | Input data failed schema validation |
| `RESOURCE_EXHAUSTED` | OOM or training time budget exceeded |
| `NO_ROBUST_MODEL` | No candidate model passed quality thresholds |
| `CRITICAL_FAILURE` | Unexpected error during pipeline execution |

## Programmatic Usage

```python
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.orchestration.orchestrator import PipelineOrchestrator

config = ToolkitConfig(
    dataset={"path": "data.csv", "target_column": "label"},
    candidates={"allowed_families": ["logistic", "rf"], "max_candidates": 2},
)

orchestrator = PipelineOrchestrator(config)
report = orchestrator.run("data.csv")

print(report.final_status)          # PipelineStage.COMPLETED
print(report.final_recommendation)  # "Recommended model: logistic_001"
print(report.warnings)              # Any issues encountered
```

### Using Individual Stages

Each pipeline stage can be used independently:

```python
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.intake.dataset_intake_manager import run_intake
from aml_toolkit.profiling.profiler_engine import run_profiling
from aml_toolkit.audit.split_auditor import run_split_audit

config = ToolkitConfig(dataset={"path": "data.csv", "target_column": "label"})

# Run intake only
intake = run_intake(config)
print(intake.manifest.modality)    # ModalityType.TABULAR
print(intake.manifest.task_type)   # TaskType.BINARY

# Run audit on the splits
audit = run_split_audit(
    data=intake.data,
    manifest=intake.manifest,
    split=intake.split_result,
    config=config,
)
print(audit.passed)                # True if no leakage detected

# Profile the data
profile = run_profiling(
    data=intake.data,
    manifest=intake.manifest,
    split=intake.split_result,
    config=config,
)
print(profile.risk_flags)          # [RiskFlag.CLASS_IMBALANCE, ...]
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run unit tests only
python -m pytest tests/unit/ -v

# Run integration tests only
python -m pytest tests/integration/ -v

# Run with coverage
python -m pytest tests/ --cov=aml_toolkit --cov-report=term-missing
```

The test suite includes:
- **Unit tests** (20 files, 350+ tests) — per-module contract verification
- **Regression tests** — guard against integration regressions across all phases
- **Integration tests** — multi-stage data flow on real synthetic datasets
- **End-to-end sanity suite** — 15 mandatory edge cases including leakage detection, resource abstention, noise-blocked oversampling, and augmentation guard

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
  ensemble/         Soft voting, weighted averaging, marginal gain evaluation
  explainability/   Feature importance, SHAP, GradCAM, confusion heatmaps, faithfulness
  reporting/        JSON and Markdown report builders
  orchestration/    Pipeline orchestrator, state machine, audit logger
  api/              CLI entrypoint
  utils/            Serialization, resource guard
```

## Design Documents

- `plan.md` — Phase-by-phase execution playbook (15 implementation phases)
- `system_design.md` — System architecture, component design, data flow, and state machine
- `CONTRIBUTING.md` — Coding conventions, naming rules, and development guidelines
- `ARTIFACTS.md` — Artifact directory conventions and output standards

## License

MIT
