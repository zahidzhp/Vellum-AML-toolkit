"""Structured logging utilities for pipeline execution tracing."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredFormatter(logging.Formatter):
    """Formats log records as JSON lines for machine-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "stage": getattr(record, "stage", None),
            "component": getattr(record, "component", None),
            "event_type": getattr(record, "event_type", None),
            "run_id": getattr(record, "run_id", None),
            "message": record.getMessage(),
        }
        payload = getattr(record, "payload", None)
        if payload is not None:
            log_entry["payload"] = payload

        return json.dumps(log_entry, default=str)


def setup_logging(
    run_id: str,
    log_dir: Path | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Configure structured logging with console and optional file output.

    Args:
        run_id: The current run ID, attached to all log records.
        log_dir: Directory for the JSON log file. If None, file logging is skipped.
        console_level: Logging level for console output.
        file_level: Logging level for file output.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("aml_toolkit")
    logger.setLevel(min(console_level, file_level))
    logger.handlers.clear()

    # Console handler — human-readable
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler — structured JSON lines
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "run.log")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    return logger


def log_event(
    logger: logging.Logger,
    message: str,
    *,
    stage: str | None = None,
    component: str | None = None,
    event_type: str | None = None,
    run_id: str | None = None,
    payload: dict[str, Any] | None = None,
    level: int = logging.INFO,
) -> None:
    """Emit a structured log event with pipeline-specific context.

    Args:
        logger: The logger instance.
        message: Human-readable log message.
        stage: Pipeline stage name.
        component: Component or module name.
        event_type: Event category (e.g., "start", "complete", "warning", "error").
        run_id: The current run ID.
        payload: Additional structured data to attach to the log record.
        level: Logging level.
    """
    extra = {
        "stage": stage,
        "component": component,
        "event_type": event_type,
        "run_id": run_id,
        "payload": payload,
    }
    logger.log(level, message, extra=extra)
