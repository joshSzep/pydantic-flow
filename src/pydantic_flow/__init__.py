"""pydantic-flow - A pydantic-ai based framework with batteries included.

This package provides a comprehensive framework built on top of pydantic-ai,
offering a batteries-included approach for building AI-powered applications.

The framework enables type-safe, composable AI workflows using Pydantic models
as inputs and outputs for each processing node.
"""

from pydantic_flow.flow import CyclicDependencyError
from pydantic_flow.flow import Flow
from pydantic_flow.flow import FlowError
from pydantic_flow.nodes import BaseNode
from pydantic_flow.nodes import FlowNode
from pydantic_flow.nodes import IfNode
from pydantic_flow.nodes import NodeOutput
from pydantic_flow.nodes import NodeWithInput
from pydantic_flow.nodes import ParserNode
from pydantic_flow.nodes import PromptConfig
from pydantic_flow.nodes import PromptNode
from pydantic_flow.nodes import RetryNode
from pydantic_flow.nodes import ToolNode
from pydantic_flow.project_info import ProjectInfo
from pydantic_flow.project_info import get_project_info

# Public API - supports both direct and module imports
__all__ = [
    "BaseNode",
    "CyclicDependencyError",
    "Flow",
    "FlowError",
    "FlowNode",
    "IfNode",
    "NodeOutput",
    "NodeWithInput",
    "ParserNode",
    "ProjectInfo",
    "PromptConfig",
    "PromptNode",
    "RetryNode",
    "ToolNode",
    "get_project_info",
]
__version__ = get_project_info().version
