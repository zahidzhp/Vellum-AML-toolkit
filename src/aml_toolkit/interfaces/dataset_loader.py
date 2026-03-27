"""Interface for dataset loading and intake."""

from abc import ABC, abstractmethod
from typing import Any

from aml_toolkit.artifacts import DatasetManifest
from aml_toolkit.core.config import ToolkitConfig


class DatasetLoader(ABC):
    """Abstract contract for loading datasets and producing a DatasetManifest.

    Implementations handle a specific input format (CSV, image folder, embeddings)
    and must detect modality, task type, and split information.
    """

    @abstractmethod
    def load(self, config: ToolkitConfig) -> tuple[DatasetManifest, Any]:
        """Load the dataset and return a manifest plus the raw data.

        Args:
            config: Toolkit configuration (dataset path, target column, etc.).

        Returns:
            A tuple of (DatasetManifest, raw data object). The raw data format
            is loader-specific (DataFrame, dict of arrays, etc.).
        """

    @abstractmethod
    def supports_modality(self, modality: str) -> bool:
        """Check whether this loader handles the given modality.

        Args:
            modality: A ModalityType value string.

        Returns:
            True if this loader can handle the modality.
        """
