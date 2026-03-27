# Contributing & Development Conventions

This document defines the coding standards, naming rules, modularity requirements, and development philosophy for the Autonomous Classification and Heatmap Generation Toolkit.

All contributors (human or AI) must follow these conventions.

---

## 1. Language and Typing

- **Python 3.11+** is required.
- **Type hints on all function signatures** — parameters and return types.
- **pydantic models** for all typed artifacts (pipeline stage outputs).
- **Protocols or ABCs** for all pluggable interfaces (model adapters, calibrators, explainability strategies, etc.).
- Use `typing` module constructs (`Optional`, `Literal`, `Union`, etc.) where appropriate.

---

## 2. Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Modules/files | `snake_case` | `schema_parser.py` |
| Functions/methods | `snake_case` | `detect_modality()` |
| Classes | `PascalCase` | `DatasetManifest` |
| Enums | `PascalCase` class, `UPPER_SNAKE` members | `ModalityType.TABULAR` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_SEED` |
| Config keys | `snake_case` | `compute_budget` |
| Artifact files | `<stage>_<type>.<ext>` | `audit_split_report.json` |

---

## 3. Modularity Rules

### One concern per module
Each major pipeline concern lives in its own package:
`intake/`, `audit/`, `profiling/`, `probes/`, `interventions/`, `models/`, `runtime/`, `postprocessing/`, `explainability/`, `reporting/`, `orchestration/`.

### No monolithic scripts
Do not put multiple pipeline stages into a single file. Do not write giant notebooks as implementations.

### No business logic in CLI
The CLI (`api/cli.py`) is a thin wrapper that parses arguments and calls the orchestrator. Decision logic, model selection, and policy evaluation belong in their respective modules.

### Inter-module communication through typed artifacts
Modules pass data to each other via serializable pydantic artifact objects (e.g., `DatasetManifest`, `SplitAuditReport`, `DataProfile`). Do not pass raw dicts or untyped structures between stages.

---

## 4. Config Philosophy

### Config-first
All thresholds and behaviors that affect ML decisions must be editable through YAML config files. This includes:
- Imbalance thresholds
- Allowed interventions
- Candidate model families
- Runtime decision thresholds (warm-up epochs, early stop patience)
- Calibration and ensemble settings
- Explainability method selection

### Fixed behaviors must be justified
If a behavior is intentionally not configurable, add a code comment explaining why.

### Hierarchical config loading
Config is loaded in layers, where later layers override earlier ones:
1. `configs/default.yaml` — global defaults
2. `configs/modes/<mode>.yaml` — operating mode overrides
3. Dataset/task-specific overrides (if provided)
4. CLI argument overrides

See `configs/README.md` for the full config philosophy.

---

## 5. Phase Completion Criteria

A phase is **not** complete just because code exists. A phase is complete when ALL of the following are true:

1. **Code exists** — all files listed in the phase spec are created or updated
2. **Tests exist** — unit tests covering the phase's acceptance criteria
3. **Config exists** — config sections are added or updated as needed
4. **Cross-check passes** — backward cross-check against all prior phases reports no gaps
5. **Acceptance checklist satisfied** — every item in the phase's checklist is explicitly met

---

## 6. No-Premature-Model Rule

Do not write real model training pipelines until the designated phase (Phase 9 in the playbook).

Before that phase:
- Define **interfaces** (protocols/ABCs) for model adapters
- Define **registries** for pluggable model families
- Define **scaffolding** (adapter shells with method signatures)
- Write **tests for structure** (interface conformance, registry registration)

This prevents architectural lock-in from premature training code.

---

## 7. Safety-First Stage Ordering

The pipeline enforces strict stage ordering. These invariants must never be violated:

1. **No balancing before split validation** — augmentation/resampling only happens after split audit passes
2. **No training before profiling** — the system must understand the data before trying models
3. **No final model claims without rationale** — rejected alternatives and selection reasons must be logged
4. **No threshold finalization without calibration assessment** — raw model scores are not final
5. **No ensemble by default** — ensemble is selected only when complementary error and gain threshold are met

---

## 8. Error Handling

- Use custom exception classes from `core/exceptions.py` (e.g., `SchemaValidationError`, `LeakageDetectedError`, `ResourceAbstentionError`).
- Resource exhaustion (OOM, GPU failure) must produce structured abstention events, not unhandled crashes.
- Invalid schema must fail early with explicit, helpful error messages.
- Unsupported modality or explainability method must fail gracefully with a warning, not crash the entire pipeline.

---

## 9. Testing Standards

- Use **pytest** for all tests.
- Tests live in `tests/unit/` and `tests/integration/`.
- Test fixtures live in `tests/fixtures/`.
- Every phase must include tests that cover its acceptance criteria.
- Integration tests must verify that pipeline stages respect safety gates (e.g., training executor refuses to run if split audit is blocking).

---

## 10. Execution Model

- **Serial execution in v1.** Do not introduce threading, multiprocessing, or async candidate training. Parallelism is a stretch goal for later.
- All execution must be traceable through structured logging with `run_id`, stage, component, event type, and timestamp.
