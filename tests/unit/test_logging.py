"""Smoke tests for structured logging utilities."""

import json
import logging
from pathlib import Path

from aml_toolkit.core.logging_utils import StructuredFormatter, log_event, setup_logging


def test_setup_logging_returns_logger():
    logger = setup_logging(run_id="test_run")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "aml_toolkit"


def test_setup_logging_with_file(tmp_path: Path):
    log_dir = tmp_path / "logs"
    logger = setup_logging(run_id="test_run", log_dir=log_dir)
    logger.info("test message")

    log_file = log_dir / "run.log"
    assert log_file.exists()
    content = log_file.read_text().strip()
    record = json.loads(content)
    assert record["message"] == "test message"
    assert record["level"] == "INFO"


def test_log_event_with_context(tmp_path: Path):
    log_dir = tmp_path / "logs"
    logger = setup_logging(run_id="test_run", log_dir=log_dir)

    log_event(
        logger,
        "Split audit completed",
        stage="audit",
        component="split_auditor",
        event_type="complete",
        run_id="test_run",
        payload={"passed": True, "warnings": 2},
    )

    log_file = log_dir / "run.log"
    content = log_file.read_text().strip()
    record = json.loads(content)
    assert record["stage"] == "audit"
    assert record["component"] == "split_auditor"
    assert record["event_type"] == "complete"
    assert record["run_id"] == "test_run"
    assert record["payload"]["passed"] is True


def test_structured_formatter_produces_valid_json():
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    data = json.loads(output)
    assert data["message"] == "test message"
    assert "timestamp" in data
