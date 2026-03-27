"""Audit logger: structured event log for pipeline execution."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("aml_toolkit")


class AuditEntry:
    """A single audit log entry."""

    def __init__(self, stage: str, event: str, detail: dict[str, Any] | None = None) -> None:
        self.timestamp = datetime.now(tz=timezone.utc).isoformat()
        self.stage = stage
        self.event = event
        self.detail = detail or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "stage": self.stage,
            "event": self.event,
            "detail": self.detail,
        }


class AuditLogger:
    """Collects and persists audit events for a pipeline run."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def log(self, stage: str, event: str, detail: dict[str, Any] | None = None) -> None:
        entry = AuditEntry(stage, event, detail)
        self._entries.append(entry)
        logger.info(f"[AUDIT] {stage}: {event}")

    def save(self, path: Path) -> None:
        """Save audit log to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self._entries]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries]
