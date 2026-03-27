# System Design Document
## Autonomous Classification and Heatmap Generation Toolkit

---

## 1. Overview

### 1.1 Objective
This document defines the system architecture, module design, data flow, interfaces, and execution logic for the Autonomous Classification and Heatmap Generation Toolkit (the "Toolkit").

The goal is to define the system architecture for the Autonomous Classification and Heatmap Generation Toolkit and provide a concrete, implementable system blueprint.

### 1.2 Design Principles
- Policy-driven, not brute-force AutoML
- Modular and extensible architecture
- Safety-first (leakage, noise, calibration)
- Budget-aware execution
- Explainability + auditability as first-class citizens
- Family-aware runtime intelligence

---

## 1.3 Functional Requirement Reference

FR codes used in this document are defined here. All behaviors are also described in prose at the point of use.

| Code | Requirement |
|------|-------------|
| FR-063A | Candidates must not be terminated before an architecture-specific minimum warm-up threshold has elapsed, unless a hard failure occurs (OOM, NaN loss, crash). |
| FR-092A | Image explainability outputs must include at least one faithfulness-oriented metric (e.g. deletion/insertion score) for supported backbones. |
| FR-128 | Augmentation must not occur before split finalization. Any augmentation pipeline must operate only on the training split after leakage checks pass. |
| FR-129 | Resource failures (OOM, GPU unavailable, memory limit exceeded) must not crash uncontrolled. They must produce a structured abstention event with reason and metrics recorded. |
| FR-130 | The system must detect label inconsistencies: identical or near-identical inputs with conflicting labels must be flagged as a risk before training. |
| FR-131 | The system must detect basic train-vs-test distribution shift (OOD signals) and include a summary in the DataProfile risk flags. |

---

## 2. High-Level Architecture

### 2.1 Architectural Style
The system follows a **pipeline + decision engine hybrid architecture**:

- Sequential staged pipeline
- Decision checkpoints between stages
- Pluggable modules per stage
- Central orchestration controller

### 2.2 Core Layers

1. **Input & Standardization Layer**
2. **Data Intelligence Layer**
3. **Probe & Diagnosis Layer**
4. **Policy & Intervention Layer**
5. **Model Orchestration Layer**
6. **Runtime Decision Engine**
7. **Post-processing Layer (Calibration + Thresholding)**
8. **Ensemble Layer**
9. **Explainability & Heatmap Layer**
10. **Governance & Reporting Layer**

---

## 3. System Components

## 3.1 Orchestrator (Core Controller)

### Responsibilities
- Controls execution flow
- Maintains system state
- Invokes modules in sequence
- Applies decision policies
- Tracks experiment lifecycle

### Inputs
- User config
- Dataset manifest

### Outputs
- Final recommendation
- Execution logs

### Internal State
- Stage status
- Active candidates
- Intervention plan
- Budget tracker

---

## 3.2 Dataset Intake Manager

### Responsibilities
- Load dataset
- Detect modality (tabular/image/embedding)
- Detect task type (binary/multiclass/multilabel)
- Validate schema

### Key Submodules
- Schema Parser
- Modality Detector
- Split Handler

### Output
- DatasetManifest object

---

## 3.3 Split Auditor

### Responsibilities
- Validate split integrity
- Detect leakage
- Enforce augmentation isolation (FR-128)

### Checks
- Duplicate overlap
- Entity leakage
- Temporal leakage
- Class absence

### Output
- SplitAuditReport

---

## 3.4 Data Profiler

### Responsibilities
- Compute dataset statistics
- Detect imbalance
- Detect label inconsistencies (FR-130)
- Detect OOD signals (FR-131)

### Key Algorithms
- Distribution comparison
- Duplicate clustering
- Feature statistics

### Output
- DataProfile

---

## 3.5 Probe Engine

### Responsibilities
- Run baseline models
- Run intervention variants
- Estimate learnability

### Pipeline
1. Baselines
2. Shallow models
3. Intervention branches

### Output
- ProbeResultSet

---

## 3.6 Intervention Planner

### Responsibilities
- Decide weighting vs resampling
- Apply safety constraints
- Generate intervention plan

### Logic Type
- Rule-based engine
- Optional meta-policy scoring

### Output
- InterventionPlan

---

## 3.7 Candidate Model Manager

### Responsibilities
- Select candidate models
- Allocate training budget
- Manage training lifecycle

### Candidate Pools
- Tabular: XGBoost, RF, Linear, MLP
- Image: CNN, ViT
- Embedding: Linear head, MLP

### Output
- CandidatePortfolio

---

## 3.8 Runtime Decision Engine

### Responsibilities
- Monitor early training signals
- Decide continue/stop/expand

### Inputs
- Training metrics stream

### Signals
- Loss trend
- Metric trend
- Calibration proxy
- Gradient stability

### Key Constraint
- Enforce warm-up threshold (FR-063A)

### Output
- RuntimeDecisionLog

---

## 3.9 Calibration & Threshold Optimizer

### Responsibilities
- Calibrate probabilities
- Optimize thresholds

### Methods
- Temperature scaling
- Isotonic regression

### Metrics
- ECE
- Brier Score

### Output
- CalibrationReport

---

## 3.10 Ensemble Builder

### Responsibilities
- Combine candidate models
- Evaluate ensemble benefit

### Methods
- Voting (initial scope)
- Averaging (initial scope)
- Stacking (stretch goal — deferred, do not implement by default)

### Constraint
- Must show complementary errors and meet a measurable gain threshold before ensemble is selected

### Output
- EnsembleReport

---

## 3.11 Explainability & Heatmap Engine

### Responsibilities
- Generate explainability outputs
- Validate faithfulness (FR-092A)

### Outputs
- Confusion heatmaps
- Feature heatmaps
- Grad-CAM
- Faithfulness scores

---

## 3.12 Governance Layer

### Responsibilities
- Log decisions
- Ensure reproducibility
- Store artifacts

### Output
- FinalReport
- Audit logs

---

## 4. Data Flow

### Stage Flow

1. Intake → DatasetManifest
2. Split Audit → SplitAuditReport
3. Profiling → DataProfile
4. Probe → ProbeResultSet
5. Intervention → InterventionPlan
6. Candidate Selection → CandidatePortfolio
7. Training + Runtime Decisions → RuntimeDecisionLog
8. Calibration → CalibrationReport
9. Ensemble → EnsembleReport
10. Explainability → ExplainabilityReport
11. Reporting → FinalReport

---

## 5. State Machine

### States
- INIT
- DATA_VALIDATED
- PROFILED
- PROBED
- INTERVENTION_SELECTED
- TRAINING_ACTIVE
- MODEL_SELECTED
- CALIBRATED
- ENSEMBLED
- EXPLAINED
- COMPLETED
- ABSTAINED

### Happy Path Transitions
```
INIT → DATA_VALIDATED → PROFILED → PROBED → INTERVENTION_SELECTED
  → TRAINING_ACTIVE → MODEL_SELECTED → CALIBRATED → ENSEMBLED
  → EXPLAINED → COMPLETED
```

### ABSTAINED Transitions
Any stage may transition to ABSTAINED. The abstention event must carry a typed reason:

| Trigger Stage | Reason |
|---------------|--------|
| Split Audit | `LEAKAGE_BLOCKED` — blocking leakage or integrity failure detected |
| Schema / Intake | `SCHEMA_INVALID` — schema cannot be parsed or modality unsupported |
| Training / Runtime | `RESOURCE_EXHAUSTED` — OOM or resource failure (FR-129) |
| Runtime Decision | `NO_ROBUST_MODEL` — no candidate met quality threshold after full training |
| Any stage | `CRITICAL_FAILURE` — unrecoverable error not covered above |

---

## 6. Configuration Schema (Conceptual)

### Key Configs
- objective_metric
- compute_budget
- mode (conservative/balanced/aggressive)
- allowed_interventions
- model_whitelist
- early_stop_policy

---

## 7. Runtime Decision Logic (Pseudo)

IF resource_failure OR critical_instability (NaN loss, crash):
    ABSTAIN

IF epoch < min_epoch_threshold:
    CONTINUE   # warm-up gate — never terminate before this (FR-063A)
ELSE:
    IF underperforming AND stable (no improvement slope):
        STOP
    IF improving:
        CONTINUE
    IF uncertain (ambiguous trend, high variance):
        EXPAND (add candidate to pool)

# Note: gradient stability is a valid signal only for neural models.
# For tabular candidates (XGBoost, RF, logistic), use metric trend
# and generalization gap (val_loss - train_loss) only.

---

## 8. Error Handling

### Categories
- Data errors
- Split errors
- Resource errors (FR-129)
- Training instability

### Strategy
- Fail-safe logging
- Abstain when necessary

---

## 9. Scalability Design

- Parallel probe execution
- Parallel candidate training
- Async logging
- Modular execution

---

## 10. Deployment Architecture

### Modes
- Local CLI
- Batch pipeline
- Service API

### Components
- Orchestrator service
- Worker nodes
- Artifact store

---

## 11. Observability

- Metrics logging
- Decision trace logs
- Model comparison summary in FinalReport (JSON + markdown; no external dashboard dependency in v1)

---

## 12. Security Considerations

- Data masking in reports
- Access control
- Artifact isolation

---

## 13. Future Enhancements

- Meta-learning policy
- Online learning
- Drift detection
- Active learning loop

---

## 14. Conclusion

This system design defines a modular, policy-driven, autonomous ML system capable of handling classification and heatmap tasks with strong safety, interpretability, and adaptability guarantees.

