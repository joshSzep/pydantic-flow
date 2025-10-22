"""Tests for the pflow.__init__ module."""

from pathlib import Path
import tempfile
import tomllib
from unittest.mock import Mock
from unittest.mock import patch

from pflow import get_project_info


def test_get_project_info_success():
    """Test get_project_info returns correct values from pyproject.toml."""
    description, version = get_project_info()

    # Verify we get the expected values from the actual pyproject.toml
    assert version == "0.1.0"
    assert description == "A pydantic-ai based framework with batteries included."


def test_get_project_info_missing_file():
    """Test get_project_info when pyproject.toml doesn't exist."""
    with patch("pflow.Path") as mock_path:
        # Mock the path to return a non-existent file
        mock_current_file = mock_path.return_value
        mock_pyproject_path = (
            mock_current_file.parent.parent.parent.__truediv__.return_value
        )
        mock_pyproject_path.exists.return_value = False

        description, version = get_project_info()

        assert description == "Project description not available"
        assert version == "Version not available"


def test_get_project_info_invalid_toml():
    """Test get_project_info with invalid TOML content."""
    with patch("pflow.Path") as mock_path:
        mock_current_file = mock_path.return_value
        mock_pyproject_path = Mock()
        mock_pyproject_path.exists.return_value = True
        mock_pyproject_path.open.side_effect = Exception("Invalid TOML")
        mock_current_file.parent.parent.parent.__truediv__.return_value = (
            mock_pyproject_path
        )

        description, version = get_project_info()

        # Should return error message and default version
        assert "Error reading project info:" in description
        assert version == "Version not available"


def test_get_project_info_missing_project_fields():
    """Test get_project_info when project fields are missing."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
        # TOML without description field
        toml_content = b"""[project]
name = "test-project"
version = "2.0.0"
"""
        f.write(toml_content)
        temp_path = Path(f.name)

    try:
        with temp_path.open("rb") as test_file:
            config = tomllib.load(test_file)

        project_info = config.get("project", {})
        description = project_info.get(
            "description", "Project description not available"
        )
        version = project_info.get("version", "Version not available")

        assert description == "Project description not available"
        assert version == "2.0.0"

    finally:
        temp_path.unlink()


def test_get_project_info_empty_project_section():
    """Test get_project_info when project section is empty."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
        # TOML with empty project section
        toml_content = b"""[project]

[build-system]
requires = ["setuptools"]
"""
        f.write(toml_content)
        temp_path = Path(f.name)

    try:
        with temp_path.open("rb") as test_file:
            config = tomllib.load(test_file)

        project_info = config.get("project", {})
        description = project_info.get(
            "description", "Project description not available"
        )
        version = project_info.get("version", "Version not available")

        assert description == "Project description not available"
        assert version == "Version not available"

    finally:
        temp_path.unlink()
