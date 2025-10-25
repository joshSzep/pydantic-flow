"""pydantic-flow - A pydantic-ai based framework with batteries included.

This package provides a comprehensive framework built on top of pydantic-ai,
offering a batteries-included approach for building AI-powered applications.

The framework is streaming-native: every node exposes an async stream of
progress as its primary interface, with non-streaming results produced by
consuming the stream internally.
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
from pydantic_flow.nodes.agent import AgentNode
from pydantic_flow.nodes.agent import LLMNode
from pydantic_flow.nodes.retriever import RetrieverNode
from pydantic_flow.project_info import ProjectInfo
from pydantic_flow.project_info import get_project_info
from pydantic_flow.prompt import ChatMessage
from pydantic_flow.prompt import ChatPromptTemplate
from pydantic_flow.prompt import ChatRole
from pydantic_flow.prompt import JoinStrategy
from pydantic_flow.prompt import PromptTemplate
from pydantic_flow.prompt import TemplateFormat
from pydantic_flow.prompt import from_template
from pydantic_flow.streaming import Heartbeat
from pydantic_flow.streaming import NonFatalError
from pydantic_flow.streaming import PartialFields
from pydantic_flow.streaming import ProgressItem
from pydantic_flow.streaming import ProgressType
from pydantic_flow.streaming import RetrievalItem
from pydantic_flow.streaming import StreamEnd
from pydantic_flow.streaming import StreamStart
from pydantic_flow.streaming import TokenChunk
from pydantic_flow.streaming import ToolArgProgress
from pydantic_flow.streaming import ToolCall
from pydantic_flow.streaming import ToolResult
from pydantic_flow.streaming.helpers import collect_all_tokens
from pydantic_flow.streaming.helpers import collect_final_result
from pydantic_flow.streaming.helpers import iter_fields
from pydantic_flow.streaming.helpers import iter_tokens
from pydantic_flow.streaming.parser import StreamingParser
from pydantic_flow.streaming.parser import parse_json_stream

# Public API - supports both direct and module imports
__all__ = [
    "AgentNode",
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
    "Heartbeat",
    "IfNode",
    "JoinStrategy",
    "LLMNode",
    "MergeNode",
    "MergeParserNode",
    "MergePromptNode",
    "MergeToolNode",
    "NodeOutput",
    "NodeWithInput",
    "NonFatalError",
    "ParserNode",
    "PartialFields",
    "ProgressItem",
    "ProgressType",
    "ProjectInfo",
    "PromptConfig",
    "PromptNode",
    "PromptTemplate",
    "RecursionLimitError",
    "RetrievalItem",
    "RetrieverNode",
    "RetryNode",
    "Route",
    "RouterFunction",
    "RoutingError",
    "RunConfig",
    "StreamEnd",
    "StreamStart",
    "StreamingParser",
    "TemplateFormat",
    "TokenChunk",
    "ToolArgProgress",
    "ToolCall",
    "ToolNode",
    "ToolResult",
    "collect_all_tokens",
    "collect_final_result",
    "from_template",
    "get_project_info",
    "iter_fields",
    "iter_tokens",
    "parse_json_stream",
]
__version__ = get_project_info().version
