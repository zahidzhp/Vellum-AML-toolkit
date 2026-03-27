"""Path management for output directories and artifact storage."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("outputs")


def generate_run_id(config_repr: str = "", dataset_path: str = "") -> str:
    """Generate a unique run ID: YYYYMMDD_HHMMSS_<short_hash>.

    Args:
        config_repr: String representation of the config for hashing.
        dataset_path: Dataset path for hashing.

    Returns:
        A unique run ID string.
    """
    now = datetime.now(tz=timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    hash_input = f"{config_repr}:{dataset_path}:{now.isoformat()}"
    short_hash = sha256(hash_input.encode()).hexdigest()[:6]
    return f"{timestamp}_{short_hash}"


def create_run_directory(
    run_id: str,
    base_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Create the output directory structure for a run.

    Args:
        run_id: The unique run identifier.
        base_dir: Base output directory. Defaults to ./outputs/.

    Returns:
        Path to the run directory.
    """
    run_dir = base_dir / run_id

    subdirs = [
        "intake",
        "audit",
        "profiling",
        "probes",
        "interventions",
        "candidates",
        "runtime",
        "calibration",
        "ensemble",
        "explainability",
        "explainability/heatmaps",
        "reporting",
        "logs",
    ]

    for subdir in subdirs:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    return run_dir
