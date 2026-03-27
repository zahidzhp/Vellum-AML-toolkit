"""Tests for the CLI entrypoint (Phase 14 gap fill).

Tests:
1. CLI imports and constructs without error.
2. Config loading from CLI arguments.
3. Validate subcommand.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from aml_toolkit.api.cli import app
from aml_toolkit.core.config import ToolkitConfig

runner = CliRunner()


class TestCLIRun:

    def test_cli_help_exits_zero(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "aml-toolkit" in result.output.lower() or "autonomous" in result.output.lower()

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--mode" in result.output

    def test_validate_help(self):
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    @patch("aml_toolkit.api.cli.setup_logging")
    @patch("aml_toolkit.api.cli.PipelineOrchestrator")
    def test_run_invokes_orchestrator(self, mock_orch_cls, mock_setup_log, tmp_path):
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({"f1": [1, 2, 3], "label": [0, 1, 0]})
        df.to_csv(csv_path, index=False)

        mock_report = MagicMock()
        mock_report.final_status.value = "COMPLETED"
        mock_report.abstention_reason = None
        mock_report.final_recommendation = "logistic_001"

        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_report
        mock_instance.run_dir = tmp_path
        mock_orch_cls.return_value = mock_instance

        result = runner.invoke(app, ["run", str(csv_path)])
        assert result.exit_code == 0
        mock_instance.run.assert_called_once()

    @patch("aml_toolkit.api.cli.setup_logging")
    @patch("aml_toolkit.api.cli.PipelineOrchestrator")
    def test_run_with_mode_override(self, mock_orch_cls, mock_setup_log, tmp_path):
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({"f1": [1, 2, 3], "label": [0, 1, 0]})
        df.to_csv(csv_path, index=False)

        mock_report = MagicMock()
        mock_report.final_status.value = "COMPLETED"
        mock_report.abstention_reason = None
        mock_report.final_recommendation = ""

        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_report
        mock_instance.run_dir = tmp_path
        mock_orch_cls.return_value = mock_instance

        result = runner.invoke(app, ["run", str(csv_path), "--mode", "conservative"])
        assert result.exit_code == 0

    @patch("aml_toolkit.api.cli.setup_logging")
    @patch("aml_toolkit.api.cli.PipelineOrchestrator")
    def test_run_with_seed_override(self, mock_orch_cls, mock_setup_log, tmp_path):
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({"f1": [1, 2, 3], "label": [0, 1, 0]})
        df.to_csv(csv_path, index=False)

        mock_report = MagicMock()
        mock_report.final_status.value = "COMPLETED"
        mock_report.abstention_reason = None
        mock_report.final_recommendation = ""

        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_report
        mock_instance.run_dir = tmp_path
        mock_orch_cls.return_value = mock_instance

        result = runner.invoke(app, ["run", str(csv_path), "--seed", "99"])
        assert result.exit_code == 0
