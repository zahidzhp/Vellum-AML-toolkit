# Configuration Philosophy

This directory holds the YAML configuration files that control all toolkit behavior.

---

## Core Principle

**Config-first**: Every threshold, policy choice, and behavior that affects ML decisions is editable through these files. The toolkit should never silently hardcode values that a user might need to change.

---

## Hierarchical Loading Order

Config values are resolved in this order, where later layers override earlier ones:

1. **`configs/default.yaml`** — Global defaults for all settings
2. **`configs/modes/<mode>.yaml`** — Operating mode overrides (conservative, balanced, aggressive, interpretable)
3. **Dataset/task-specific overrides** — Passed via config file path or inline
4. **CLI argument overrides** — Command-line flags take highest precedence

---

## Config Sections

Each section maps to a pipeline stage or cross-cutting concern:

| Section | Controls |
|---------|----------|
| `dataset` | Input path, modality override, target column, metadata columns |
| `splitting` | Strategy (stratified, grouped, temporal), test/val ratios, group column, time column |
| `profiling` | Imbalance severity thresholds, duplicate detection settings, OOD shift sensitivity |
| `probes` | Enabled probe models, intervention branches to test, metric selection |
| `interventions` | Allowed intervention types, oversampling noise-risk threshold, calibration requirement rules |
| `candidates` | Allowed model families, budget allocation strategy, warm-up epoch thresholds per family |
| `runtime_decision` | Minimum warm-up epochs, improvement slope threshold, overfit gap limit, patience |
| `calibration` | Enabled methods (temperature scaling, isotonic), primary objective (ECE, Brier Score) |
| `ensemble` | Enabled strategies, marginal gain threshold, max ensemble size |
| `explainability` | Enabled methods per modality, faithfulness metric toggle |
| `reporting` | Output directory, report formats (JSON, Markdown), verbosity level |
| `compute` | Max training time, memory limit, GPU preference, resource abstention policy |

---

## Operating Modes

Each mode file overrides specific defaults to shift the toolkit's behavior:

- **`conservative.yaml`** — Strict leakage checks, lower compute budgets, prefers interpretable models, requires calibration
- **`balanced.yaml`** — Default trade-off between thoroughness and speed
- **`aggressive.yaml`** — Wider candidate pools, longer training budgets, allows more complex ensembles
- **`interpretable.yaml`** — Restricts to inherently interpretable models (logistic, RF, shallow trees), disables deep learning candidates

---

## Adding New Config Values

When adding a new configurable behavior:

1. Add the default value to `configs/default.yaml`
2. Add mode-specific overrides if the behavior differs by mode
3. Ensure the config key is documented in this file
4. Wire the value through `core/config.py` so it's accessible via the config object
