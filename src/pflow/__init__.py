"""PFlow - A pydantic-ai based framework with batteries included.

This package provides a comprehensive framework built on top of pydantic-ai,
offering a batteries-included approach for building AI-powered applications.
"""

from pathlib import Path
import tomllib


def get_project_info():
    """Get project information from pyproject.toml file.

    Returns:
        tuple: A tuple containing (description, version) from the project config.

    """
    # Find the pyproject.toml file - go up from current file to project root
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        return "Project description not available", "Version not available"

    try:
        with pyproject_path.open("rb") as f:
            config = tomllib.load(f)

        project_info = config.get("project", {})
        description = project_info.get(
            "description", "Project description not available"
        )
        version = project_info.get("version", "Version not available")

        return description, version
    except Exception as e:
        return f"Error reading project info: {e}", "Version not available"


def main():
    """Print project description and version."""
    description, version = get_project_info()
    print(f"PFlow v{version}")
    print(f"Description: {description}")


if __name__ == "__main__":
    main()
