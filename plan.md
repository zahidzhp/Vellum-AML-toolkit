# Phase-by-Phase Execution Playbook
## Autonomous Classification and Heatmap Generation Toolkit

This document is designed to be handed directly to an AI coding assistant.

The intended usage pattern is simple:

> Read this file completely. Execute **Phase X** only. Do not skip steps. Do not start later phases. Do not implement models unless the phase explicitly requires model adapter scaffolding. At the end of the phase, run the required cross-checks against previous phases and report gaps before claiming completion.

This playbook is deliberately written to reduce implementation drift, hidden assumptions, and premature coding.

---

# 1. Master Operating Instruction for Any AI

Use this exact instruction when giving this playbook to an AI:

```text
Read the file `plan.md` completely before writing code.
Also read `system_design.md` for component-level architecture, data flow, and the state machine.

Then execute Phase X only.

Rules:
1. Do not implement future phases.
2. Do not invent extra model training logic unless Phase X explicitly asks for it.
3. Keep the system modular, typed, and config-driven.
4. All behavior must be controllable through editable config files unless there is a strong reason not to.
5. At the start of the phase, perform the required backward cross-check against all previously completed phases and list any missing pieces.
6. If previous phases are incomplete or inconsistent, fix only what is necessary for Phase X and clearly report those fixes.
7. At the end of the phase, provide:
   - changed file tree,
   - code,
   - tests,
   - config updates,
   - assumptions,
   - cross-check report,
   - acceptance checklist.
8. Do not collapse the architecture into monolithic files.
9. Do not silently hardcode policy decisions that should live in config.
10. Do not proceed to the next phase until asked.
```

---

# 2. Global Build Doctrine

These rules apply in every phase.

## 2.1 Modular-first
Every major concern must live in its own module:
- intake
- audit
- profiling
- probes
- interventions
- models
- runtime
- postprocessing
- explainability
- reporting
- orchestration

No giant scripts. No giant notebooks. No hidden business logic inside CLI wrappers.

## 2.2 Config-first
The system must be editable through user-facing config files.

That means:
- thresholds should be configurable,
- enabled interventions should be configurable,
- allowed candidate families should be configurable,
- operating mode should be configurable,
- runtime decision thresholds should be configurable,
- explainability and calibration settings should be configurable.

If the AI adds behavior that affects training or decision-making, it must either:
1. expose it in config, or
2. explain why it is intentionally fixed.

## 2.3 No premature model building
The system must **not** start coding actual model training logic too early.

Until the proper phase arrives:
- define interfaces,
- define adapters,
- define registries,
- define placeholders,
- define contracts,
- define tests for structure,
- but do not build real training pipelines yet.

This is critical. Premature model code usually causes architectural lock-in.

## 2.4 Open and extensible design
The toolkit should feel like a platform, not a one-off project.

Use:
- typed artifacts,
- protocols or abstract base classes,
- registries,
- isolated policy logic,
- versionable configs,
- serializable outputs.

Future adaptation should be easy for:
- new model families,
- new modalities,
- new explainability methods,
- new policies,
- new metrics,
- new deployment paths.

## 2.5 Casual reasoning requirement
For each phase, the AI should include a short section called **“Why this phase is shaped this way”**.

It should explain, in plain engineering language:
- what the phase is trying to protect,
- why the implementation order matters,
- what future problems this design avoids.

Not philosophical. Just practical reasoning.

## 2.6 Evidence before completion
A phase is not complete because code exists.
A phase is complete only when:
- code exists,
- tests exist,
- configs exist or were updated,
- cross-checks pass,
- acceptance criteria are explicitly satisfied.

---

# 3. Repository Shape That Must Be Preserved

The AI should preserve this structure unless there is a very strong reason to evolve it.

```text
autonomous_ml_toolkit/
  pyproject.toml
  README.md
  configs/
    default.yaml
    modes/
      conservative.yaml
      balanced.yaml
      aggressive.yaml
      interpretable.yaml
  src/
    aml_toolkit/
      __init__.py
      api/
        cli.py
        service.py
      core/
        config.py
        enums.py
        exceptions.py
        logging_utils.py
        paths.py
        seeds.py
      artifacts/
        dataset_manifest.py
        split_audit_report.py
        data_profile.py
        probe_result.py
        intervention_plan.py
        candidate_portfolio.py
        runtime_decision_log.py
        calibration_report.py
        ensemble_report.py
        explainability_report.py
        final_report.py
      interfaces/
        dataset_loader.py
        profiler.py
        probe_model.py
        intervention.py
        candidate_model.py
        calibrator.py
        ensemble_strategy.py
        explainability.py
        reporter.py
      intake/
        schema_parser.py
        modality_detector.py
        task_detector.py
        split_builder.py
        dataset_intake_manager.py
      audit/
        split_auditor.py
        leakage_checks.py
        augmentation_guard.py
      profiling/
        class_distribution.py
        duplicates.py
        label_conflicts.py
        missingness.py
        outliers.py
        drift_ood.py
        profiler_engine.py
      probes/
        baseline_models.py
        tabular_probes.py
        image_embedding_probes.py
        probe_engine.py
      interventions/
        weighting.py
        resampling.py
        augmentation.py
        thresholding.py
        intervention_planner.py
      models/
        registry.py
        tabular/
          logistic_adapter.py
          rf_adapter.py
          xgb_adapter.py
          mlp_adapter.py
        image/
          cnn_adapter.py
          vit_adapter.py
          embedding_head_adapter.py
      runtime/
        warmup_policy.py
        metric_tracker.py
        decision_engine.py
        training_executor.py
      postprocessing/
        calibration.py
        threshold_optimizer.py
        ensemble_builder.py
      explainability/
        tabular_explain.py
        image_heatmaps.py
        faithfulness.py
      reporting/
        audit_logger.py
        report_builder.py
        visualizations.py
      orchestration/
        policy_engine.py
        orchestrator.py
        state_machine.py
      utils/
        serialization.py
        io_utils.py
        metrics.py
        split_utils.py
        resource_guard.py
  tests/
    unit/
    integration/
    fixtures/
  examples/
    tabular_demo/
    image_demo/
```

Minor adjustments are acceptable. Architectural collapse is not.

---

# 4. Cross-Check Protocol Used In Every Phase

At the beginning of every phase after Phase 1, the AI must perform a **Backward Cross-Check**.

## 4.1 Backward Cross-Check Template
The AI must answer these before writing new code:

1. Which previous phases are assumed complete?
2. Which files from previous phases are required by the current phase?
3. Are those files present?
4. Do their interfaces still match what this phase needs?
5. Are any configs missing fields now required?
6. Are any tests from prior phases likely to break because of this phase?
7. Are any previous abstractions too weak and need minimal extension?
8. If a fix is needed, is it:
   - mandatory for current phase,
   - safe and minimal,
   - backward-compatible?

The AI must produce a short **Cross-Check Report** before implementation.

## 4.2 End-of-Phase Forward Guard
At the end of each phase, the AI must also state:
- what future phases now depend on,
- what contracts were established,
- what must not be changed casually later.

This prevents invisible breakage.

---

# 5. Phase 0 - Readiness and Guardrails

## Goal
Establish the working agreement, implementation constraints, and initial config philosophy before real coding begins.

## Why this phase is shaped this way
Without guardrails, many AIs jump straight into training code or bury behavior in scripts. This phase exists to force architectural discipline before any implementation momentum appears.

## Do
- create or refine README project purpose section,
- define the package scope in plain language,
- define coding conventions,
- define naming conventions,
- define config philosophy,
- define what counts as “phase complete”,
- define artifact directory conventions.

## Do not
- build training logic,
- build model code,
- build data pipelines.

## Required outputs
- README scaffold,
- contributor/developer notes if useful,
- config philosophy note,
- artifact output directory convention.

## Cross-check
None. This is the first phase.

## Acceptance checklist
- project purpose is clearly documented,
- config-first philosophy is documented,
- modularity and no-premature-model rules are documented,
- artifact directory standards are documented.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 0 only.
Do not start any implementation beyond repository-level readiness and guardrails.
Create the minimal repository documentation and conventions needed for the later phases.
Return changed file tree, files created, assumptions, and acceptance checklist.
```

---

# 6. Phase 1 - Core Skeleton, Typed Artifacts, Config, and Exceptions

## Goal
Create the base package structure, typed artifacts, config system, enums, exceptions, serialization helpers, and logging utilities.

## Why this phase is shaped this way
Everything later depends on stable contracts. If artifacts and config are weak, all later modules become tangled and brittle.

## Inputs assumed
- repository exists,
- Phase 0 documentation conventions exist.

## Backward Cross-Check
Check:
- README and conventions from Phase 0 exist,
- repository shape is compatible with modular package creation,
- config philosophy is documented.

## Implement
### Core
- `core/config.py`
- `core/enums.py`
- `core/exceptions.py`
- `core/logging_utils.py`
- `core/paths.py`
- `core/seeds.py`

### Artifacts
Implement typed, serializable artifacts for:
- `DatasetManifest`
- `SplitAuditReport`
- `DataProfile`
- `ProbeResultSet`
- `InterventionPlan`
- `CandidatePortfolio`
- `RuntimeDecisionLog`
- `CalibrationReport`
- `EnsembleReport`
- `ExplainabilityReport`
- `FinalReport`

### Utilities
- JSON/YAML serialization helpers
- artifact save/load helpers

### Configs
- `configs/default.yaml`
- mode override files

## Do not
- implement real dataset loading,
- implement profiling logic,
- implement probe logic,
- implement models.

## Config requirements
The config must include stubs or real sections for:
- dataset
- splitting
- profiling
- probes
- interventions
- candidates
- runtime_decision
- calibration
- ensemble
- explainability
- reporting
- compute

The user must be able to edit these files later.

## Required tests
- artifact serialization tests,
- config loading tests,
- enum and exception smoke tests,
- logging utility smoke test.

## End-of-phase forward guard
Future phases will depend on:
- stable artifact names,
- stable config access patterns,
- stable enum semantics,
- stable exception taxonomy.

Do not casually rename these later.

## Acceptance checklist
- package installs,
- config loads with mode overlays,
- artifacts serialize and deserialize,
- base logging works,
- tests pass.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 1 only.
Before coding, perform the backward cross-check against Phase 0.
Build the package skeleton, typed artifacts, config system, enums, exceptions, logging utilities, and serialization helpers.
Do not implement dataset logic or model logic.
All thresholds and behaviors that will matter later must have config placeholders.
Return:
- cross-check report,
- changed file tree,
- code,
- tests,
- config files,
- why this phase is shaped this way,
- acceptance checklist.
```

---

# 7. Phase 2 - Interfaces and Abstract Contracts

## Goal
Define the interfaces and contracts that later implementations must follow.

## Why this phase is shaped this way
This phase prevents future modules from inventing their own incompatible APIs. It keeps the system open for extension.

## Backward Cross-Check
Verify from Phase 1:
- artifacts exist and are importable,
- enums exist for modality/task/decision/risk,
- config sections exist for all future modules,
- exception taxonomy is sufficient.

## Implement
### Interfaces / Protocols / ABCs
- dataset loader
- profiler
- probe model
- intervention
- candidate model
- calibrator
- ensemble strategy
- explainability strategy
- reporter

Each should define minimal but useful contracts.

## Also implement
- model family tags or metadata contract,
- artifact-producing mixins if helpful,
- adapter registration hooks if needed.

## Do not
- implement actual dataset parsing,
- implement actual training,
- implement actual explanation methods.

## Required tests
- interface conformance tests using lightweight dummy implementations,
- import and typing smoke tests.

## End-of-phase forward guard
Future module implementations must conform to these interfaces instead of creating private local methods.

## Acceptance checklist
- interfaces are clear and typed,
- dummy implementations can satisfy them,
- no real model logic exists yet,
- later modules have a clear contract target.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 2 only.
Before coding, perform the backward cross-check against Phases 0 and 1.
Implement the shared interfaces and abstract contracts for all major module types.
Do not implement real loaders, models, or explainability logic yet.
Return the cross-check report, changed file tree, code, tests, reasoning, and acceptance checklist.
```

---

# 8. Phase 3 - Dataset Intake, Schema Parsing, Task Detection, and Split Building

## Goal
Enable safe dataset understanding and split creation.

## Why this phase is shaped this way
No ML should happen before the system understands what kind of problem it is solving and how the data is partitioned.

## Backward Cross-Check
Verify:
- artifact contracts for `DatasetManifest` and related config exist,
- dataset loader and related interfaces exist,
- exception types support schema and split errors,
- config sections for dataset and splitting are present.

## Implement
- schema parser
- modality detector
- task detector
- split builder
- dataset intake manager

## Supported v1 inputs
- tabular CSV
- image folder classification structure
- embedding matrix + label input

## Required behaviors
- detect binary/multiclass/multilabel,
- detect modality,
- validate required label information,
- support user-provided splits,
- create stratified splits where possible,
- support grouped split logic,
- support temporal split logic when configured,
- emit `DatasetManifest`.

## Do not
- run leakage checks yet (those belong to Phase 4); split formation may check for class absence only,
- perform profiling,
- perform balancing,
- implement model code.

## Required tests
- valid CSV intake,
- invalid schema handling,
- multiclass vs multilabel detection,
- grouped split builder test,
- temporal split builder test,
- image folder intake test,
- embedding input manifest test.

## End-of-phase forward guard
Later auditing and profiling will assume `DatasetManifest` is reliable and stable.

## Acceptance checklist
- dataset intake works for v1 supported formats,
- manifest is produced,
- split builder works with config,
- task detection is correct on test fixtures.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 3 only.
Before coding, perform the backward cross-check against Phases 0, 1, and 2.
Implement dataset intake, schema parsing, modality detection, task detection, split building, and the dataset intake manager.
Do not implement profiling, leakage auditing, interventions, or model logic.
Everything must remain config-driven.
Return the cross-check report, changed file tree, code, tests, reasoning, and acceptance checklist.
```

---

# 9. Phase 4 - Split Auditing, Leakage Checks, and Augmentation Guard

## Goal
Block unsafe data flows before any profiling or training behavior appears.

## Why this phase is shaped this way
Most downstream results become meaningless if split integrity is compromised. This phase is the safety gate.

## Backward Cross-Check
Verify:
- `DatasetManifest` produced by Phase 3 includes fields needed for audit,
- split config fields exist,
- exception types support split integrity failures,
- current tests from earlier phases still pass.

## Implement
- split auditor
- duplicate overlap checks
- grouped/entity leakage checks
- temporal leakage checks
- class absence checks
- augmentation guard

## Required behaviors
- output `SplitAuditReport`,
- block unsafe augmentation before split finalization,
- distinguish warnings vs blocking issues,
- support resource-light duplicate overlap check first,
- log leakage findings in structured form.

## Do not
- perform full profiling yet,
- apply any balancing,
- run any probes or candidate models.

## Required tests
- duplicate across train/test,
- grouped leakage,
- temporal leakage,
- class absent in validation split,
- augmentation requested before split finalization.

## End-of-phase forward guard
All later phases must check audit status before acting.
No intervention or training module should bypass this.

## Acceptance checklist
- audit report is generated consistently,
- blocking issues work,
- augmentation leakage prevention works,
- prior tests remain green.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 4 only.
Before coding, perform the backward cross-check against Phases 0 through 3.
Implement split auditing, leakage checks, class absence checks, and augmentation safety enforcement.
Do not implement profiling or model logic.
Ensure later phases will be forced to respect the audit result.
Return the cross-check report, changed file tree, code, tests, reasoning, and acceptance checklist.
```

---

# 10. Phase 5 - Profiling Engine and Data Risk Analysis

## Goal
Profile dataset health and generate actionable risk summaries.

## Why this phase is shaped this way
Before trying models, the system should understand what kind of mess or opportunity exists in the data.

## Backward Cross-Check
Verify:
- audit report exists and can be consumed,
- `DataProfile` artifact exists and has enough fields,
- config sections for profiling thresholds are present,
- earlier phases enforce that profiling only runs after intake/audit.

## Implement
- class distribution profiler
- missingness profiler
- duplicate summary integration
- label conflict detector
- outlier summary module
- basic train-vs-test shift/OOD summary
- profiler engine aggregator

## Required behaviors
- detect severe imbalance,
- detect identical or near-identical input with conflicting labels where feasible,
- summarize split-wise data health,
- emit risk flags,
- remain cheap enough for early pipeline execution.

## Do not
- implement probes yet,
- implement balancing logic,
- implement model training.

## Required tests
- severe imbalance fixture,
- conflicting-label fixture,
- missingness summary test,
- OOD-like shift smoke test,
- profile aggregation test.

## End-of-phase forward guard
Intervention planning later will assume risk flags and profile summaries are trustworthy.

## Acceptance checklist
- profile report generated,
- risk flags populated,
- label conflict detection works on fixtures,
- earlier tests remain green.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 5 only.
Before coding, perform the backward cross-check against Phases 0 through 4.
Implement the profiling engine and data risk analysis modules.
Do not implement probes, intervention execution, or model training yet.
Keep profiling outputs structured, typed, and config-driven.
Return the cross-check report, changed file tree, code, tests, reasoning, and acceptance checklist.
```

---

# 11. Phase 6 - Probe Engine Scaffolding and Baseline Diagnostics

## Goal
Introduce low-cost diagnostic experimentation without turning this into the full training system.

## Why this phase is shaped this way
The toolkit needs evidence before intervention planning. But at this stage, probes are diagnostic tools, not the final optimization engine.

## Backward Cross-Check
Verify:
- audit gating exists,
- profiling outputs are accessible,
- probe artifact types exist,
- interface contracts for probe models are sufficient,
- config sections for probes exist.

## Implement
- majority baseline
- stratified baseline
- tabular shallow probe wrappers
- image embedding probe scaffolding
- probe engine orchestrator
- probe result logging

## Important guardrail
Only implement **probe-level models** here.
Do not build the full candidate model training system.
If a simple logistic regression or random forest is used as a probe, keep it clearly inside the probe module and do not let it become the full candidate orchestration layer yet.

## Required behaviors
- run baseline diagnostics,
- run a small configurable set of shallow probes,
- run lightweight intervention variants directly inside the probe module (e.g. pass `class_weight='balanced'` to a logistic probe, apply SMOTE before a shallow fit); the full Intervention Planner in Phase 7 will consume these results, not re-implement them,
- save `ProbeResultSet`.

## Do not
- build real candidate training loops,
- build runtime decision engine,
- build ensemble logic.

## Required tests
- baseline probe ranking test,
- probe engine output format test,
- config-driven probe selection test,
- image embedding probe smoke test using stub or fixture.

## End-of-phase forward guard
Probe outputs will later inform intervention planning. Do not overload them with final-model assumptions.

## Acceptance checklist
- probe engine runs on fixtures,
- probe outputs serialize,
- no premature full-model training logic has appeared,
- earlier tests remain green.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 6 only.
Before coding, perform the backward cross-check against Phases 0 through 5.
Implement the diagnostic probe engine and baseline probes.
Do not build the full candidate model training system yet.
Keep probe logic separate from later model orchestration.
Return the cross-check report, changed file tree, code, tests, reasoning, and acceptance checklist.
```

---

# 12. Phase 7 - Intervention Planning Rules and Policy Engine Foundations

## Goal
Convert audit and profiling evidence plus probe outcomes into a structured intervention plan.

## Why this phase is shaped this way
This is where the toolkit starts making decisions, but still at the planning level rather than heavy execution.

## Backward Cross-Check
Verify:
- profile and probe artifacts expose needed metrics,
- audit results are enforceable,
- config has thresholds for imbalance, noise, oversampling permissions, and related policy items,
- probe outputs remain clearly diagnostic.

## Implement
- intervention planner
- rule-based policy engine foundation
- weighting rule evaluation
- oversampling/undersampling eligibility logic
- augmentation eligibility logic
- calibration requirement flagging
- abstention suggestion logic

## Required behaviors
- intervention selection must use config-driven thresholds,
- rejected interventions must be logged with reasons,
- oversampling must be blocked when noise or leakage concerns demand it,
- planner must not execute interventions yet unless very lightweight simulation helpers are needed.

## Do not
- build real candidate training loops,
- build runtime decision engine,
- build ensemble logic.

## Required tests
- prefer weighting over oversampling case,
- reject oversampling due to label conflict/noise case,
- require calibration case,
- abstention recommendation case,
- intervention plan serialization test.

## End-of-phase forward guard
Candidate orchestration later must consume the plan rather than bypass it.

## Acceptance checklist
- intervention plan is structured and serializable,
- selected and rejected actions are explicit,
- config controls the decision rules,
- earlier tests remain green.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 7 only.
Before coding, perform the backward cross-check against Phases 0 through 6.
Implement the rule-based intervention planner and policy engine foundation.
Do not build candidate training loops or runtime execution yet.
Keep all major decision thresholds editable in config.
Return the cross-check report, changed file tree, code, tests, reasoning, and acceptance checklist.
```

---

# 13. Phase 8 - Candidate Registry and Model Adapter Scaffolding

## Goal
Define how real candidate models will plug into the system, without yet writing the full model execution stack.

## Why this phase is shaped this way
The system now knows what it wants to try, but it still needs a clean adapter surface before training code shows up.

## Backward Cross-Check
Verify:
- candidate portfolio artifact exists,
- model interfaces from Phase 2 are still suitable,
- intervention planning outputs can inform candidate selection,
- config sections for candidate families and budgets exist.

## Implement
- model registry
- candidate family metadata
- adapter scaffolding for tabular/image/embedding candidates
- candidate selection logic
- candidate portfolio builder

## Required behaviors
- candidate families selectable by config,
- registry pluggable,
- adapters may still be placeholder or partial at this phase,
- candidate portfolio records warm-up rule references and budget allocations.

## Do not
- write full training loops yet,
- write deep model optimization logic yet,
- wire runtime decision behavior yet.

## Required tests
- registry registration test,
- candidate selection test,
- config whitelist/blacklist test,
- portfolio serialization test.

## End-of-phase forward guard
Future training execution must use the registry and adapters rather than bypassing them.

## Acceptance checklist
- registry works,
- adapters are scaffolded cleanly,
- candidate portfolio is produced,
- still no premature training pipeline exists.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 8 only.
Before coding, perform the backward cross-check against Phases 0 through 7.
Implement the model registry, adapter scaffolding, candidate selection logic, and candidate portfolio builder.
Do not build the training executor yet.
Keep the system open and pluggable.
Return the cross-check report, changed file tree, code, tests, reasoning, and acceptance checklist.
```

---

# 14. Phase 9 - Training Executor and Resource Guard

## Goal
Introduce controlled candidate execution with safe resource handling.

## Why this phase is shaped this way
Only now is it appropriate to let the system start real candidate training behavior, because the contracts, safety checks, and planning layers already exist.

## Backward Cross-Check
Verify:
- audit gating is enforced,
- intervention plan can be consumed,
- candidate registry and adapters exist,
- resource-related exceptions are available,
- config sections for training and compute budgets exist.

## Implement
- training executor
- resource guard
- training lifecycle hooks
- candidate execution tracing
- structured handling for OOM/resource abstention

## Required behaviors
- executor must refuse to run if split audit is blocking,
- executor must use candidate registry/adapters,
- OOM or resource failures must convert into structured abstention events,
- execution trace must be logged.

## Important guardrail
Do not build autonomous runtime decision logic yet beyond basic execution safety. This phase is about execution plumbing.

Execution is serial in v1. Do not introduce threading, multiprocessing, or async candidate training here. Parallelism is a stretch goal; adding it prematurely will break the logging and state tracking layers.

## Required tests
- executor runs a minimal candidate,
- audit-blocked execution test,
- resource abstention test,
- execution trace generation test.

## End-of-phase forward guard
Runtime decision engine in the next phase will depend on stable execution traces and metric reporting.

## Acceptance checklist
- controlled training execution exists,
- resource failures do not crash uncontrolled,
- executor respects audit gate,
- earlier tests remain green.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 9 only.
Before coding, perform the backward cross-check against Phases 0 through 8.
Implement the training executor and resource guard.
Do not build the runtime decision engine yet beyond what is needed for basic execution safety.
Use config-driven training and compute settings.
Return the cross-check report, changed file tree, code, tests, reasoning, and acceptance checklist.
```

---

# 15. Phase 10 - Runtime Decision Engine and Warm-Up Policies

## Goal
Make training adaptive without being reckless.

## Why this phase is shaped this way
This is the first time the system is allowed to act on early learning signals. It must do so conservatively and transparently.

## Backward Cross-Check
Verify:
- executor provides metric traces,
- candidate portfolio contains warm-up references,
- config has runtime thresholds,
- decision enums and runtime log artifacts are sufficient,
- resource abstention flows are stable.

## Implement
- metric tracker
- warm-up policy manager
- runtime decision engine
- runtime decision logging

## Required behaviors
- no candidate terminated before architecture-specific minimum threshold except hard failure,
- decisions limited to CONTINUE, STOP, EXPAND, ABSTAIN,
- reasons and triggering metrics must be logged,
- behavior must be rule-based and config-driven.

## Do not
- introduce meta-learning or black-box policy selection,
- silently compare different families with naive absolute metrics only,
- apply gradient stability signals to non-neural candidates (XGBoost, RF, logistic); for those, use metric trend and generalization gap only.

## Required tests
- slow warm-up candidate not stopped too early,
- clear underperformer stopped after threshold,
- unstable candidate abstention/stop case,
- runtime decision log content test.

## End-of-phase forward guard
Postprocessing later assumes the runtime-selected candidates are traceable and justified.

## Acceptance checklist
- warm-up policy enforced,
- runtime decisions logged,
- config controls decision thresholds,
- earlier tests remain green.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 10 only.
Before coding, perform the backward cross-check against Phases 0 through 9.
Implement the runtime decision engine, metric tracker, and warm-up policies.
Keep the logic rule-based, transparent, and config-driven.
Return the cross-check report, changed file tree, code, tests, reasoning, and acceptance checklist.
```

---

# 16. Phase 11 - Calibration and Threshold Optimization

## Goal
Make final candidate outputs usable for decision-making, not just classification labels.

## Why this phase is shaped this way
Raw model scores are often misleading. This phase turns predictions into better-behaved decision signals.

## Backward Cross-Check
Verify:
- candidate outputs include probabilities where expected,
- config includes calibration and objective settings,
- final candidate selection traces are available,
- relevant exceptions and artifacts exist.

## Implement
- calibration module
- threshold optimizer
- calibration report builder

## Required behaviors
- support at least temperature scaling and isotonic regression where applicable,
- support ECE and Brier Score reporting,
- allow primary objective configuration for probability-sensitive use cases,
- fail clearly or warn appropriately for non-probabilistic paths.

## Do not
- build ensemble logic yet,
- mix calibration logic into training executor.

## Required tests
- calibration report generation,
- threshold optimization test,
- ECE/Brier primary objective config test,
- non-probabilistic candidate handling test.

## End-of-phase forward guard
Ensembling later should consume calibrated outputs where policy requires.

## Acceptance checklist
- calibration works on supported candidates,
- thresholding is logged,
- config controls objective selection,
- earlier tests remain green.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 11 only.
Before coding, perform the backward cross-check against Phases 0 through 10.
Implement calibration and threshold optimization.
Keep calibration separate from training execution.
Make probability-quality objectives configurable.
Return the cross-check report, changed file tree, code, tests, reasoning, and acceptance checklist.
```

---

# 17. Phase 12 - Ensemble Builder

## Goal
Allow selective model combination when there is real evidence it helps.

## Why this phase is shaped this way
Ensembling is easy to overuse. This phase must keep it disciplined and deployment-aware.

## Backward Cross-Check
Verify:
- candidate evaluation outputs are stable,
- calibrated outputs are available where needed,
- config includes ensemble thresholds and allowed strategies,
- ensemble interfaces exist.

## Implement
- soft voting
- weighted averaging
- ensemble evaluation logic
- ensemble report builder

## Required behaviors
- ensemble only when complementary error or measurable gain threshold is met,
- record rejection reason when ensemble is not chosen,
- keep strategy pluggable.

## Do not
- implement complex stacking unless explicitly planned,
- hard-enable ensemble by default.

## Required tests
- ensemble accepted on real gain case,
- ensemble rejected on marginal gain case,
- ensemble report serialization test.

## End-of-phase forward guard
Explainability and reporting later must be able to represent both single-model and ensemble outcomes.

## Acceptance checklist
- ensemble logic is selective,
- rejections are logged,
- config controls strategy and thresholds,
- earlier tests remain green.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 12 only.
Before coding, perform the backward cross-check against Phases 0 through 11.
Implement selective ensemble building and reporting.
Do not default to always using an ensemble.
Return the cross-check report, changed file tree, code, tests, reasoning, and acceptance checklist.
```

---

# 18. Phase 13 - Explainability, Heatmaps, and Faithfulness Checks

## Goal
Produce interpretable outputs without pretending they are automatically trustworthy.

## Why this phase is shaped this way
Explainability visuals are useful, but they can also mislead. This phase adds the outputs and the honesty layer.

## Backward Cross-Check
Verify:
- final candidate and/or ensemble outputs are available,
- explainability interfaces exist,
- config includes enabled explainability methods and warning policies,
- reporting artifacts can store explainability output references.

## Implement
- confusion heatmaps,
- tabular explainability outputs,
- image heatmap generation for supported backbones,
- faithfulness metric helper,
- graceful fallback behavior.

## Required behaviors
- unsupported explainability path must warn, not crash the full system,
- explainability methods must be labeled in outputs,
- faithfulness-oriented metric should be included where supported,
- report must contain caveats.

## Do not
- bury explainability inside model adapters,
- make unsupported methods silently disappear.

## Required tests
- confusion heatmap artifact test,
- unsupported explainability fallback test,
- faithfulness helper smoke test,
- explainability report serialization test.

## End-of-phase forward guard
Final reporting later depends on explainability outputs being optional but structured.

## Acceptance checklist
- explainability outputs exist,
- failures degrade gracefully,
- faithfulness check path exists where supported,
- earlier tests remain green.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 13 only.
Before coding, perform the backward cross-check against Phases 0 through 12.
Implement explainability outputs, heatmaps, faithfulness checks, and graceful fallback behavior.
Keep the module separate and pluggable.
Return the cross-check report, changed file tree, code, tests, reasoning, and acceptance checklist.
```

---

# 19. Phase 14 - Reporting, Audit Logging, CLI, and Orchestration Wiring

## Goal
Make the system usable end-to-end while preserving separation of concerns.

## Why this phase is shaped this way
The core engine is now present. This phase exposes it cleanly instead of leaving it as a pile of internal modules.

## Backward Cross-Check
Verify:
- artifacts across all previous phases serialize consistently,
- config loading works for all sections,
- orchestrator can wire stages in the proper order,
- reporting structures can summarize every stage.

## Implement
- orchestrator wiring
- state machine
- report builder
- audit logger
- CLI entrypoint
- optional service stub

## Required behaviors
- end-to-end flow must obey stage order,
- audit gate must prevent illegal downstream actions,
- all outputs saved to structured artifact directory,
- user can point to a config file and override defaults,
- CLI must not bury logic that belongs in orchestration.

## Do not
- move business logic into CLI,
- bypass modular components.

## Required tests
- orchestrator stage-order test,
- CLI config override test,
- final report generation test,
- end-to-end happy path smoke test.

## End-of-phase forward guard
Sanity and hardening in the next phase assume a complete but modular system.

## Acceptance checklist
- CLI works,
- orchestrator is stage-correct,
- reports are generated,
- user-editable config is active end-to-end,
- earlier tests remain green.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 14 only.
Before coding, perform the backward cross-check against Phases 0 through 13.
Implement orchestration wiring, reporting, audit logging, and a CLI entrypoint.
Keep the system modular and config-driven.
Return the cross-check report, changed file tree, code, tests, reasoning, and acceptance checklist.
```

---

# 20. Phase 15 - Hardening, Regression Tests, Fixtures, and End-to-End Sanity Layer

## Goal
Turn the assembled system into something trustworthy.

## Why this phase is shaped this way
A system this modular can still fail through integration gaps. This phase exists to catch exactly that.

## Backward Cross-Check
Verify every phase contract:
- configs still load,
- artifacts still serialize,
- stage ordering remains correct,
- no later phase bypassed earlier safety gates,
- no module collapsed into a hidden monolith.

## Implement
- regression tests,
- integration tests,
- fixtures,
- end-to-end sanity suite,
- documentation updates,
- gap-fix patches if needed.

## Mandatory edge-case tests
1. severe imbalance
2. duplicate leakage
3. grouped leakage
4. temporal leakage
5. class absence in split
6. conflicting labels
7. OOM/resource abstention
8. slow warm-up candidate
9. oversampling rejected due to noise risk
10. non-probabilistic calibration request
11. unsupported explainability route
12. ensemble rejected for tiny gain
13. abstention when no robust model found
14. OOD-like shift flag
15. augmentation leakage prevention

## Required behaviors
- all major failure types must produce structured outputs,
- regression suite must verify previous phases remain intact,
- missing config or broken serialization must be caught.

## Acceptance checklist
- end-to-end tests pass,
- regression coverage exists across key contracts,
- sanity suite passes,
- docs updated.

## Execution prompt for an AI
```text
Read the playbook completely.
Execute Phase 15 only.
Before coding, perform the backward cross-check against all earlier phases.
Implement hardening, regression coverage, fixtures, integration tests, and the end-to-end sanity layer.
If you discover missing pieces from earlier phases, patch them minimally and document the reason.
Return the cross-check report, changed file tree, code, tests, reasoning, acceptance checklist, and final sanity summary.
```

---

# 21. Final Sanity Check - Must Be Run After All Phases

After Phase 15, the AI must run a whole-system sanity review.

## Final sanity objectives
Confirm that:
1. the system is still modular,
2. the system is still config-driven,
3. early safety gates cannot be bypassed casually,
4. no training or decision logic is hardcoded in the CLI,
5. no late-phase module broke typed artifact contracts,
6. no phase silently introduced hidden defaults that should be user-editable,
7. candidate training did not appear before the designated phase,
8. runtime decisions are transparent and logged,
9. calibration and explainability failures degrade gracefully,
10. ensemble logic is selective rather than always-on,
11. final reports reflect abstention states correctly,
12. future extension points remain open.

## Final sanity report structure
The AI must provide:
- **Architecture sanity**
- **Config sanity**
- **Safety gate sanity**
- **Interface contract sanity**
- **Reporting sanity**
- **Extensibility sanity**
- **Known limitations**
- **Recommended next improvements**

---

# 22. Best Possible Hand-Off Prompt

Use this exact prompt when you want another AI to execute a phase.

```text
Read the file `plan.md` completely.
Also read `system_design.md` for component-level architecture, data flow, and the state machine.

Then execute Phase X only.

You must follow the playbook exactly.

Before coding:
- perform the backward cross-check for this phase,
- report missing or weak parts from previous phases,
- patch only what is minimally necessary.

During coding:
- keep everything modular,
- keep everything config-driven,
- do not implement future phases,
- do not prematurely build model training logic unless this phase explicitly allows it,
- do not hardcode policy decisions that belong in config,
- include a short “Why this phase is shaped this way” section.

At the end return:
1. cross-check report,
2. changed file tree,
3. code,
4. tests,
5. config changes,
6. assumptions,
7. acceptance checklist,
8. forward guard notes.
```

---

# 23. If You Want the Prompt Rewritten for Even Better AI Behavior

Use this stricter version:

```text
You are implementing one controlled phase of a modular Python toolkit.
Your job is not to “finish the whole system.”
Your job is to execute exactly one phase from the playbook.

Read `plan.md` and `system_design.md` fully.
Execute Phase X only.

Non-negotiable rules:
- Do not skip the backward cross-check.
- Do not implement future-phase behavior.
- Do not bury logic inside the CLI.
- Do not create monolithic files.
- Do not hardcode values that belong in config.
- Do not silently alter earlier contracts without documenting it.
- Do not claim completion without tests and an acceptance checklist.

You must preserve:
- modularity,
- typed artifacts,
- interface-driven design,
- config-based control,
- safety-first stage ordering,
- extensibility for future adaptation.

Output format:
A. Cross-check report
B. Why this phase is shaped this way
C. Changed file tree
D. Code
E. Tests
F. Config changes
G. Assumptions
H. Acceptance checklist
I. Forward guard notes
```

---

# 24. Final Note

This playbook is intentionally strict.

That is not bureaucracy. It is protection against the most common AI implementation failure modes:
- skipping architecture,
- starting models too early,
- hardcoding policies,
- burying decisions in scripts,
- forgetting regression checks,
- making the code impossible to extend later.

If an AI follows this file phase by phase, the result should stay:
- modular,
- open,
- user-configurable,
- safer to evolve,
- and much easier to audit.

