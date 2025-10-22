"""PFlow - A pydantic-ai based framework with batteries included.

This package provides a comprehensive framework built on top of pydantic-ai,
offering a batteries-included approach for building AI-powered applications.

The framework enables type-safe, composable AI workflows using Pydantic models
as inputs and outputs for each processing node.
"""

from pathlib import Path
import tomllib

from pflow.flow import CyclicDependencyError
from pflow.flow import Flow
from pflow.flow import FlowError
from pflow.nodes import BaseNode
from pflow.nodes import IfNode
from pflow.nodes import NodeOutput
from pflow.nodes import NodeWithInput
from pflow.nodes import ParserNode
from pflow.nodes import PromptConfig
from pflow.nodes import PromptNode
from pflow.nodes import RetryNode
from pflow.nodes import ToolNode

# Public API - supports both direct and module imports
__all__ = [
    "BaseNode",
    "CyclicDependencyError",
    "Flow",
    "FlowError",
    "IfNode",
    "NodeOutput",
    "NodeWithInput",
    "ParserNode",
    "PromptConfig",
    "PromptNode",
    "RetryNode",
    "ToolNode",
]


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
