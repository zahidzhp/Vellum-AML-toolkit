"""Report builder: assembles FinalReport from stage artifacts and writes JSON/Markdown."""

import json
import logging
from pathlib import Path
from typing import Any

from aml_toolkit.artifacts import FinalReport
from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import PipelineStage
from aml_toolkit.interfaces.reporter import Reporter

logger = logging.getLogger("aml_toolkit")


class JsonReporter(Reporter):
    """Write the final report as a JSON file."""

    def generate(
        self,
        artifacts: dict[str, Any],
        output_dir: Path,
        config: ToolkitConfig,
    ) -> FinalReport:
        report = _build_final_report(artifacts, config)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "final_report.json"
        with open(path, "w") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
        logger.info(f"JSON report written to {path}")
        return report

    def format_name(self) -> str:
        return "json"


class MarkdownReporter(Reporter):
    """Write the final report as a Markdown summary."""

    def generate(
        self,
        artifacts: dict[str, Any],
        output_dir: Path,
        config: ToolkitConfig,
    ) -> FinalReport:
        report = _build_final_report(artifacts, config)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "final_report.md"

        lines = [
            f"# Pipeline Report: {report.run_id}",
            "",
            f"**Status:** {report.final_status.value}",
            f"**Timestamp:** {report.timestamp}",
            "",
        ]

        if report.abstention_reason:
            lines.append(f"**Abstention Reason:** {report.abstention_reason.value}")
            lines.append("")

        lines.append("## Stages Completed")
        for stage in report.stages_completed:
            lines.append(f"- {stage.value}")
        lines.append("")

        if report.final_recommendation:
            lines.append("## Recommendation")
            lines.append(report.final_recommendation)
            lines.append("")

        sections = [
            ("Dataset", report.dataset_summary),
            ("Split Audit", report.split_audit_summary),
            ("Profile", report.profile_summary),
            ("Probes", report.probe_summary),
            ("Interventions", report.intervention_summary),
            ("Candidates", report.candidate_summary),
            ("Runtime Decisions", report.runtime_decision_summary),
            ("Calibration", report.calibration_summary),
            ("Ensemble", report.ensemble_summary),
            ("Explainability", report.explainability_summary),
        ]

        for title, summary in sections:
            if summary:
                lines.append(f"## {title}")
                for k, v in summary.items():
                    lines.append(f"- **{k}:** {v}")
                lines.append("")

        if report.warnings:
            lines.append("## Warnings")
            for w in report.warnings:
                lines.append(f"- {w}")
            lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown report written to {path}")
        return report

    def format_name(self) -> str:
        return "markdown"


def build_report(
    artifacts: dict[str, Any],
    output_dir: Path,
    config: ToolkitConfig,
) -> FinalReport:
    """Build and write reports in all configured formats.

    Args:
        artifacts: Dict of stage artifacts.
        output_dir: Output directory.
        config: Toolkit config.

    Returns:
        The FinalReport.
    """
    reporters: dict[str, Reporter] = {
        "json": JsonReporter(),
        "markdown": MarkdownReporter(),
    }

    report = None
    for fmt in config.reporting.formats:
        if fmt in reporters:
            report = reporters[fmt].generate(artifacts, output_dir, config)
        else:
            logger.warning(f"Unknown report format: {fmt}")

    if report is None:
        report = _build_final_report(artifacts, config)

    return report


def _build_final_report(artifacts: dict[str, Any], config: ToolkitConfig) -> FinalReport:
    """Assemble a FinalReport from collected artifacts."""
    report = FinalReport(
        run_id=artifacts.get("run_id", ""),
        final_status=artifacts.get("final_status", PipelineStage.COMPLETED),
        abstention_reason=artifacts.get("abstention_reason"),
        stages_completed=artifacts.get("stages_completed", []),
        warnings=artifacts.get("warnings", []),
    )

    # Summaries from each stage artifact
    if "dataset_manifest" in artifacts:
        m = artifacts["dataset_manifest"]
        report.dataset_summary = _safe_dump(m)

    if "split_audit_report" in artifacts:
        a = artifacts["split_audit_report"]
        report.split_audit_summary = _safe_dump(a)

    if "data_profile" in artifacts:
        report.profile_summary = _safe_dump(artifacts["data_profile"])

    if "probe_results" in artifacts:
        report.probe_summary = _safe_dump(artifacts["probe_results"])

    if "intervention_plan" in artifacts:
        report.intervention_summary = _safe_dump(artifacts["intervention_plan"])

    if "candidate_portfolio" in artifacts:
        report.candidate_summary = _safe_dump(artifacts["candidate_portfolio"])

    if "runtime_decision_log" in artifacts:
        report.runtime_decision_summary = _safe_dump(artifacts["runtime_decision_log"])

    if "calibration_report" in artifacts:
        report.calibration_summary = _safe_dump(artifacts["calibration_report"])

    if "ensemble_report" in artifacts:
        report.ensemble_summary = _safe_dump(artifacts["ensemble_report"])

    if "explainability_report" in artifacts:
        report.explainability_summary = _safe_dump(artifacts["explainability_report"])

    # Final recommendation
    if report.final_status == PipelineStage.COMPLETED:
        best = artifacts.get("best_candidate_id", "unknown")
        report.final_recommendation = f"Recommended model: {best}"
    elif report.final_status == PipelineStage.ABSTAINED:
        report.final_recommendation = (
            f"Pipeline abstained. Reason: {report.abstention_reason.value if report.abstention_reason else 'unknown'}."
        )

    return report


def _safe_dump(obj: Any) -> dict[str, Any]:
    """Safely convert a Pydantic model or dict to a summary dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}
