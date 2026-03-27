"""JSON/YAML serialization helpers for artifact save/load."""

import json
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def save_artifact_json(artifact: BaseModel, path: Path) -> None:
    """Save a pydantic artifact to a pretty-printed JSON file.

    Args:
        artifact: The pydantic model instance to serialize.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    json_str = artifact.model_dump_json(indent=2)
    path.write_text(json_str)


def load_artifact_json(artifact_class: type[T], path: Path) -> T:
    """Load a pydantic artifact from a JSON file.

    Args:
        artifact_class: The pydantic model class to deserialize into.
        path: Source file path.

    Returns:
        An instance of artifact_class populated from the JSON file.
    """
    data = json.loads(path.read_text())
    return artifact_class.model_validate(data)


def save_artifact_yaml(artifact: BaseModel, path: Path) -> None:
    """Save a pydantic artifact to a YAML file.

    Args:
        artifact: The pydantic model instance to serialize.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = artifact.model_dump(mode="json")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_artifact_yaml(artifact_class: type[T], path: Path) -> T:
    """Load a pydantic artifact from a YAML file.

    Args:
        artifact_class: The pydantic model class to deserialize into.
        path: Source file path.

    Returns:
        An instance of artifact_class populated from the YAML file.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    return artifact_class.model_validate(data if data is not None else {})
