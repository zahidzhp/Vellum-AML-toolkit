"""Typed artifact for dataset intake output."""

from pydantic import BaseModel, Field

from aml_toolkit.core.enums import ModalityType, SplitStrategy, TaskType


class DatasetManifest(BaseModel):
    """Complete description of the ingested dataset and its splits."""

    dataset_id: str
    modality: ModalityType
    task_type: TaskType
    target_column: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    split_strategy: SplitStrategy
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0
    class_labels: list[str] = Field(default_factory=list)
    group_column: str | None = None
    time_column: str | None = None
    metadata_columns: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
