"""Project information utilities."""

from pathlib import Path
import tomllib

from pydantic import BaseModel


class ProjectInfo(BaseModel):
    """Project information from pyproject.toml."""

    description: str
    version: str


def get_project_info() -> ProjectInfo:
    """Get project information from pyproject.toml file.

    Returns:
        ProjectInfo: A Pydantic model containing description and version.

    """
    # Find the pyproject.toml file - go up from current file to project root
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        return ProjectInfo(
            description="Project description not available",
            version="Version not available",
        )

    try:
        with pyproject_path.open("rb") as f:
            config = tomllib.load(f)

        project_info = config.get("project", {})
        description = project_info.get(
            "description", "Project description not available"
        )
        version = project_info.get("version", "Version not available")

        return ProjectInfo(description=description, version=version)
    except Exception as e:
        return ProjectInfo(
            description=f"Error reading project info: {e}",
            version="Version not available",
        )
