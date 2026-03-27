"""Automatic modality detection based on input data or config."""

from pathlib import Path

from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import ModalityType
from aml_toolkit.core.exceptions import UnsupportedModalityError


def detect_modality(config: ToolkitConfig) -> ModalityType:
    """Detect the dataset modality from config or path inspection.

    Detection logic:
    1. If config.dataset.modality_override is set, use it directly.
    2. If the dataset path points to a directory, assume IMAGE.
    3. If the dataset path ends in .csv, assume TABULAR.
    4. If the dataset path ends in .npy or .npz, assume EMBEDDING.
    5. Otherwise, raise UnsupportedModalityError.

    Args:
        config: Toolkit configuration.

    Returns:
        Detected ModalityType.

    Raises:
        UnsupportedModalityError: If modality cannot be determined.
    """
    # Explicit override
    if config.dataset.modality_override:
        override = config.dataset.modality_override.upper()
        try:
            return ModalityType(override)
        except ValueError:
            raise UnsupportedModalityError(
                f"Unsupported modality override: '{config.dataset.modality_override}'. "
                f"Supported: {[m.value for m in ModalityType]}"
            )

    # Auto-detect from path
    dataset_path = Path(config.dataset.path)

    if dataset_path.is_dir():
        return ModalityType.IMAGE

    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return ModalityType.TABULAR
    elif suffix in (".npy", ".npz"):
        return ModalityType.EMBEDDING

    raise UnsupportedModalityError(
        f"Cannot detect modality from path '{config.dataset.path}'. "
        f"Provide a .csv file (tabular), a directory (image), "
        f"a .npy/.npz file (embedding), or set modality_override in config."
    )
