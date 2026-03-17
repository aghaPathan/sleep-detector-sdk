"""Tests for sleep_detector_sdk.cli — CLI entry point."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from sleep_detector_sdk.cli import main


class TestCLI:
    def test_download_model_command(self):
        with patch("sleep_detector_sdk.cli.ModelManager") as MockMM:
            mock_mm = MockMM.return_value
            mock_mm.is_cached = False
            mock_mm.download.return_value = "/fake/path/model.dat"

            exit_code = main(["download-model"])
            assert exit_code == 0
            mock_mm.download.assert_called_once()

    def test_download_model_with_custom_path(self):
        with patch("sleep_detector_sdk.cli.ModelManager") as MockMM:
            mock_mm = MockMM.return_value
            mock_mm.is_cached = False
            mock_mm.download.return_value = "/custom/model.dat"

            exit_code = main(["download-model", "--path", "/custom/dir"])
            MockMM.assert_called_with(cache_dir="/custom/dir")

    def test_download_model_skips_if_cached(self, capsys):
        with patch("sleep_detector_sdk.cli.ModelManager") as MockMM:
            mock_mm = MockMM.return_value
            mock_mm.is_cached = True
            mock_mm.model_path = "/cached/model.dat"

            exit_code = main(["download-model"])
            assert exit_code == 0
            mock_mm.download.assert_not_called()
            output = capsys.readouterr().out
            assert "already" in output.lower()

    def test_no_args_shows_help(self, capsys):
        exit_code = main([])
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "sleep-detector-sdk" in output.lower() or "usage" in output.lower()

    def test_unknown_command_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["unknown-cmd"])
        assert exc_info.value.code == 2
