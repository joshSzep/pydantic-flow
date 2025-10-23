"""Test coverage for the main entry point and initialization functions."""

import subprocess
import sys
from unittest.mock import patch

import coverage

from pydantic_flow import ProjectInfo
from pydantic_flow import get_project_info
from pydantic_flow.__main__ import main


class TestInitCoverage:
    """Test coverage for __init__.py functionality."""

    def test_get_project_info_success(self):
        """Test successful project info retrieval."""
        info = get_project_info()
        assert isinstance(info, ProjectInfo)
        assert isinstance(info.description, str)
        assert isinstance(info.version, str)
        assert len(info.description) > 0
        assert len(info.version) > 0

    def test_main_function_execution(self):
        """Test the main() function by running the module as a script."""
        # Test running the module as __main__
        result = subprocess.run(
            [sys.executable, "-m", "pydantic_flow"],
            capture_output=True,
            text=True,
            check=False,  # Don't raise on non-zero exit
        )

        assert result.returncode == 0
        assert "pydantic-flow v" in result.stdout
        assert "pydantic-ai based framework" in result.stdout

    @patch("pydantic_flow.__main__.get_project_info")
    def test_main_function_with_mock(self, mock_get_info):
        """Test main function with mocked get_project_info."""
        mock_get_info.return_value = ProjectInfo(
            description="Test Description", version="1.0.0"
        )

        with patch("builtins.print") as mock_print:
            main()

        # Verify print was called with expected content
        mock_print.assert_any_call("pydantic-flow v1.0.0: Test Description")

    def test_main_module_execution(self):
        """Test running __main__.py directly."""
        # Test the __main__.py main function
        with patch("pydantic_flow.__main__.get_project_info") as mock_get_info:
            mock_get_info.return_value = ProjectInfo(
                description="Test", version="1.0.0"
            )
            with patch("builtins.print"):
                main()
            mock_get_info.assert_called_once()

    def test_main_module_actual_execution(self):
        """Test running __main__.py with actual execution to cover exit path."""
        # Run the actual __main__.py to cover the completion/exit path
        result = subprocess.run(
            [sys.executable, "-m", "pydantic_flow.__main__"],
            capture_output=True,
            text=True,
            check=False,
        )

        # This should succeed and cover the exit path
        assert result.returncode == 0
        assert "pydantic-flow v" in result.stdout
        assert "pydantic-ai based framework" in result.stdout

    def test_main_module_with_coverage(self):
        """Test __main__ execution with coverage capture."""
        # Create a coverage instance
        cov = coverage.Coverage()
        cov.start()

        try:
            # Call main directly to ensure coverage captures it
            with patch("builtins.print"):
                main()
        finally:
            cov.stop()
            cov.save()

        # This should have captured the main function execution
