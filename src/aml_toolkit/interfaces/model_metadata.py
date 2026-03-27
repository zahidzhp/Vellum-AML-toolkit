"""Model family metadata contract and adapter registry."""

from dataclasses import dataclass, field
from typing import Any

from aml_toolkit.core.enums import ModalityType
from aml_toolkit.interfaces.candidate_model import CandidateModel


@dataclass
class ModelFamilyMetadata:
    """Metadata describing a model family's properties and constraints."""

    family_name: str
    display_name: str
    supported_modalities: list[ModalityType]
    is_neural: bool = False
    default_warmup_epochs: int = 5
    is_probabilistic: bool = True
    supports_gradcam: bool = False
    tags: list[str] = field(default_factory=list)


class ModelRegistry:
    """Registry for pluggable model adapters.

    Adapters are registered by family name and looked up at runtime
    when building the candidate portfolio.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, type[CandidateModel]] = {}
        self._metadata: dict[str, ModelFamilyMetadata] = {}

    def register(
        self,
        family_name: str,
        adapter_class: type[CandidateModel],
        metadata: ModelFamilyMetadata,
    ) -> None:
        """Register a model adapter with its metadata.

        Args:
            family_name: Unique identifier for this model family.
            adapter_class: The CandidateModel subclass.
            metadata: Descriptive metadata for this family.
        """
        self._adapters[family_name] = adapter_class
        self._metadata[family_name] = metadata

    def get_adapter(self, family_name: str) -> type[CandidateModel]:
        """Look up a registered adapter class by family name.

        Args:
            family_name: The model family identifier.

        Returns:
            The registered CandidateModel subclass.

        Raises:
            KeyError: If the family name is not registered.
        """
        if family_name not in self._adapters:
            raise KeyError(
                f"Model family '{family_name}' is not registered. "
                f"Available: {list(self._adapters.keys())}"
            )
        return self._adapters[family_name]

    def get_metadata(self, family_name: str) -> ModelFamilyMetadata:
        """Look up metadata for a registered model family.

        Args:
            family_name: The model family identifier.

        Returns:
            The ModelFamilyMetadata for this family.

        Raises:
            KeyError: If the family name is not registered.
        """
        if family_name not in self._metadata:
            raise KeyError(f"Model family '{family_name}' is not registered.")
        return self._metadata[family_name]

    def list_families(self) -> list[str]:
        """Return all registered family names."""
        return list(self._adapters.keys())

    def list_families_for_modality(self, modality: ModalityType) -> list[str]:
        """Return family names that support the given modality.

        Args:
            modality: The modality to filter by.

        Returns:
            List of family names supporting this modality.
        """
        return [
            name
            for name, meta in self._metadata.items()
            if modality in meta.supported_modalities
        ]
