"""pydantic-flow - A pydantic-ai based framework with batteries included.

This package provides a comprehensive framework built on top of pydantic-ai,
offering a batteries-included approach for building AI-powered applications.

The framework enables type-safe, composable AI workflows using Pydantic models
as inputs and outputs for each processing node.
"""

from pydantic_flow.core import FlowTimeoutError
from pydantic_flow.core import RecursionLimitError
from pydantic_flow.core import Route
from pydantic_flow.core import RouterFunction
from pydantic_flow.core import RoutingError
from pydantic_flow.core import RunConfig
from pydantic_flow.flow import CompiledFlow
from pydantic_flow.flow import CyclicDependencyError
from pydantic_flow.flow import ExecutionMode
from pydantic_flow.flow import Flow
from pydantic_flow.flow import FlowError
from pydantic_flow.nodes import BaseNode
from pydantic_flow.nodes import FlowNode
from pydantic_flow.nodes import IfNode
from pydantic_flow.nodes import MergeNode
from pydantic_flow.nodes import MergeParserNode
from pydantic_flow.nodes import MergePromptNode
from pydantic_flow.nodes import MergeToolNode
from pydantic_flow.nodes import NodeOutput
from pydantic_flow.nodes import NodeWithInput
from pydantic_flow.nodes import ParserNode
from pydantic_flow.nodes import PromptConfig
from pydantic_flow.nodes import PromptNode
from pydantic_flow.nodes import RetryNode
from pydantic_flow.nodes import ToolNode
from pydantic_flow.project_info import ProjectInfo
from pydantic_flow.project_info import get_project_info
from pydantic_flow.prompt import ChatMessage
from pydantic_flow.prompt import ChatPromptTemplate
from pydantic_flow.prompt import ChatRole
from pydantic_flow.prompt import JoinStrategy
from pydantic_flow.prompt import PromptTemplate
from pydantic_flow.prompt import TemplateFormat
from pydantic_flow.prompt import from_template

# Public API - supports both direct and module imports
__all__ = [
    "BaseNode",
    "ChatMessage",
    "ChatPromptTemplate",
    "ChatRole",
    "CompiledFlow",
    "CyclicDependencyError",
    "ExecutionMode",
    "Flow",
    "FlowError",
    "FlowNode",
    "FlowTimeoutError",
    "IfNode",
    "JoinStrategy",
    "MergeNode",
    "MergeParserNode",
    "MergePromptNode",
    "MergeToolNode",
    "NodeOutput",
    "NodeWithInput",
    "ParserNode",
    "ProjectInfo",
    "PromptConfig",
    "PromptNode",
    "PromptTemplate",
    "RecursionLimitError",
    "RetryNode",
    "Route",
    "RouterFunction",
    "RoutingError",
    "RunConfig",
    "TemplateFormat",
    "ToolNode",
    "from_template",
    "get_project_info",
]
__version__ = get_project_info().version
