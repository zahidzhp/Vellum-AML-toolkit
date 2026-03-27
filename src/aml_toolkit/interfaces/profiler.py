"""Interface for data profiling modules."""

from abc import ABC, abstractmethod
from typing import Any

from aml_toolkit.artifacts import DataProfile, DatasetManifest
from aml_toolkit.core.config import ToolkitConfig


class Profiler(ABC):
    """Abstract contract for profiling a dataset and producing risk summaries.

    Each profiler implementation handles one aspect of data health
    (class distribution, duplicates, label conflicts, etc.).
    """

    @abstractmethod
    def profile(
        self,
        data: Any,
        manifest: DatasetManifest,
        config: ToolkitConfig,
    ) -> DataProfile:
        """Profile the dataset and return a DataProfile.

        Args:
            data: The raw dataset (format depends on modality).
            manifest: The dataset manifest from intake.
            config: Toolkit configuration.

        Returns:
            A DataProfile with statistics and risk flags.
        """

    @abstractmethod
    def name(self) -> str:
        """Return the profiler's identifier (e.g., 'class_distribution')."""
