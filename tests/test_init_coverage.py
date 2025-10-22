"""Test coverage for the main entry point and initialization functions."""

import runpy
import subprocess
import sys
from unittest.mock import patch

import coverage

from pflow import get_project_info
from pflow import main


class TestInitCoverage:
    """Test coverage for __init__.py functionality."""

    def test_get_project_info_success(self):
        """Test successful project info retrieval."""
        description, version = get_project_info()
        assert isinstance(description, str)
        assert isinstance(version, str)
        assert len(description) > 0
        assert len(version) > 0

    def test_main_function_execution(self):
        """Test the main() function by running the module as a script."""
        # Test running the module as __main__
        result = subprocess.run(
            [sys.executable, "-m", "pflow"],
            capture_output=True,
            text=True,
            check=False,  # Don't raise on non-zero exit
        )

        assert result.returncode == 0
        assert "pflow v" in result.stdout
        assert "Description:" in result.stdout

    @patch("pflow.get_project_info")
    def test_main_function_with_mock(self, mock_get_info):
        """Test main function with mocked get_project_info."""
        mock_get_info.return_value = ("Test Description", "1.0.0")

        with patch("builtins.print") as mock_print:
            main()

        # Verify print was called with expected content
        mock_print.assert_any_call("pflow v1.0.0")
        mock_print.assert_any_call("Description: Test Description")

    def test_main_module_execution(self):
        """Test running __main__.py directly."""
        # Test the __main__.py file directly
        with patch("pflow.main") as mock_main:
            runpy.run_module("pflow.__main__", run_name="__main__")
            mock_main.assert_called_once()

    def test_main_module_actual_execution(self):
        """Test running __main__.py with actual execution to cover exit path."""
        # Run the actual __main__.py to cover the completion/exit path
        result = subprocess.run(
            [sys.executable, "-m", "pflow.__main__"],
            capture_output=True,
            text=True,
            check=False,
        )

        # This should succeed and cover the exit path
        assert result.returncode == 0
        assert "pflow v" in result.stdout
        assert "Description:" in result.stdout

    def test_main_module_with_coverage(self):
        """Test __main__ execution with coverage capture."""
        # Create a coverage instance
        cov = coverage.Coverage()
        cov.start()

        try:
            # Run the __main__ module directly
            runpy.run_module("pflow.__main__", run_name="__main__")
        finally:
            cov.stop()
            cov.save()

        # This should have captured the exit path in __main__.py
