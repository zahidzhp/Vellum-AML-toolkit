"""Interface for report generation."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from aml_toolkit.artifacts import FinalReport
from aml_toolkit.core.config import ToolkitConfig


class Reporter(ABC):
    """Abstract contract for generating pipeline reports.

    Reporters take the collected artifacts from all stages and produce
    the final output (JSON report, Markdown summary, etc.).
    """

    @abstractmethod
    def generate(
        self,
        artifacts: dict[str, Any],
        output_dir: Path,
        config: ToolkitConfig,
    ) -> FinalReport:
        """Generate the final report from collected stage artifacts.

        Args:
            artifacts: Dict mapping stage names to their artifact objects.
            output_dir: Directory to write report files.
            config: Toolkit configuration.

        Returns:
            FinalReport artifact.
        """

    @abstractmethod
    def format_name(self) -> str:
        """Return the reporter's output format identifier (e.g., 'json', 'markdown')."""
