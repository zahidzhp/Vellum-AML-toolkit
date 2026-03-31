"""CLI entrypoint for the Autonomous ML Toolkit.

Thin CLI layer — all business logic lives in the orchestrator.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

import logging

from aml_toolkit.core.config import load_config
from aml_toolkit.core.enums import OperatingMode
from aml_toolkit.orchestration.orchestrator import PipelineOrchestrator

app = typer.Typer(
    name="aml-toolkit",
    help="Autonomous Classification and Heatmap Generation Toolkit",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    dataset: Path = typer.Argument(..., help="Path to the input dataset (CSV or directory)."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to a YAML config file."),
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Operating mode: conservative, balanced, aggressive, interpretable."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Override output directory."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed override."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Run the full pipeline on a dataset."""
    # Setup basic console logging before orchestrator creates a run
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("aml_toolkit").setLevel(log_level)

    # Build config with overrides
    overrides = {}
    if output_dir:
        overrides["reporting"] = {"output_dir": output_dir}
    if seed is not None:
        overrides["seed"] = seed

    mode_enum = None
    if mode:
        mode_enum = mode.upper()

    toolkit_config = load_config(
        config_path=config,
        mode=mode_enum,
        overrides=overrides if overrides else None,
    )

    console.print(f"[bold]AML Toolkit[/bold] — mode: {toolkit_config.mode.value}, seed: {toolkit_config.seed}")
    console.print(f"Dataset: {dataset}")

    # Run pipeline
    orchestrator = PipelineOrchestrator(toolkit_config)
    report = orchestrator.run(dataset)

    # Summary
    console.print()
    console.print(f"[bold]Status:[/bold] {report.final_status.value}")
    if report.abstention_reason:
        console.print(f"[bold red]Abstention:[/bold red] {report.abstention_reason.value}")
    if report.final_recommendation:
        console.print(f"[bold green]Recommendation:[/bold green] {report.final_recommendation}")

    run_dir = orchestrator.run_dir
    if run_dir:
        console.print(f"[bold]Outputs:[/bold] {run_dir}")


@app.command()
def validate(
    dataset: Path = typer.Argument(..., help="Path to the input dataset."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to a YAML config file."),
) -> None:
    """Validate a dataset without running the full pipeline."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("aml_toolkit").setLevel(logging.INFO)

    toolkit_config = load_config(config_path=config)

    from aml_toolkit.intake.dataset_intake_manager import DatasetIntakeManager
    manager = DatasetIntakeManager(toolkit_config)

    try:
        result = manager.ingest(dataset)
        console.print(f"[green]Valid[/green]: modality={result.modality.value}, task={result.task_type.value}")
        console.print(f"  Samples: train={len(result.y_train)}, val={len(result.y_val)}, test={len(result.y_test)}")
    except Exception as e:
        console.print(f"[red]Validation failed:[/red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
