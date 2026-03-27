# Artifact Directory Conventions

This document defines the output structure, naming, and versioning standards for artifacts produced by the toolkit.

---

## 1. Output Directory Structure

Every run writes to a unique directory under `outputs/`:

```
outputs/
  <run_id>/
    run_manifest.json
    intake/
      dataset_manifest.json
    audit/
      split_audit_report.json
    profiling/
      data_profile.json
    probes/
      probe_result_set.json
    interventions/
      intervention_plan.json
    candidates/
      candidate_portfolio.json
    runtime/
      runtime_decision_log.json
    calibration/
      calibration_report.json
    ensemble/
      ensemble_report.json
    explainability/
      explainability_report.json
      heatmaps/           (image heatmap files, if generated)
    reporting/
      final_report.json
      final_report.md
    logs/
      run.log             (structured JSON log)
```

---

## 2. Run ID Format

Each run gets a unique `run_id` composed of:

```
<YYYYMMDD>_<HHMMSS>_<short_hash>
```

Example: `20260327_143052_a1b2c3`

The `short_hash` is derived from the config + dataset path to help distinguish concurrent or repeat runs.

---

## 3. Artifact Naming

Artifacts follow this pattern:

```
<artifact_type>.<ext>
```

| Artifact | Filename | Format |
|----------|----------|--------|
| Dataset Manifest | `dataset_manifest.json` | JSON |
| Split Audit Report | `split_audit_report.json` | JSON |
| Data Profile | `data_profile.json` | JSON |
| Probe Result Set | `probe_result_set.json` | JSON |
| Intervention Plan | `intervention_plan.json` | JSON |
| Candidate Portfolio | `candidate_portfolio.json` | JSON |
| Runtime Decision Log | `runtime_decision_log.json` | JSON |
| Calibration Report | `calibration_report.json` | JSON |
| Ensemble Report | `ensemble_report.json` | JSON |
| Explainability Report | `explainability_report.json` | JSON |
| Final Report | `final_report.json` + `final_report.md` | JSON + Markdown |
| Run Manifest | `run_manifest.json` | JSON |
| Run Log | `run.log` | JSON lines |

---

## 4. Artifact Format Standards

### JSON artifacts
- All typed pydantic artifacts serialize to JSON via their `.model_dump_json()` method.
- JSON must be pretty-printed (indented) for human readability.
- Datetime values stored as ISO 8601 strings.
- Enum values stored as their string name.

### Markdown reports
- The final report is also rendered as Markdown for quick human review.
- Markdown includes summaries of each stage, not raw data dumps.

### Image artifacts
- Heatmaps saved as PNG files in the `explainability/heatmaps/` subdirectory.
- Named as `<sample_id>_<method>.png` (e.g., `img_042_gradcam.png`).

---

## 5. Run Manifest

Every run produces a `run_manifest.json` at the root of its output directory. It contains:

- `run_id`
- `timestamp` (ISO 8601)
- `config_snapshot` (full resolved config used for this run)
- `dataset_path`
- `modality`
- `task_type`
- `operating_mode`
- `stages_completed` (list of stage names that finished)
- `final_status` (`COMPLETED` or `ABSTAINED` with reason)
- `artifact_paths` (map of artifact name to relative file path)

---

## 6. Artifact Immutability

Once a stage writes its artifact, that artifact is not modified by later stages. Later stages read prior artifacts as inputs but produce their own outputs. This makes every stage's output independently auditable.

---

## 7. Output Directory Configuration

The base output directory defaults to `./outputs/` but is configurable via:
- Config key: `reporting.output_dir`
- CLI flag: `--output-dir`
