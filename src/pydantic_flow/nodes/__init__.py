"""Node types for building workflows.

This module provides various node types for constructing type-safe workflows.

Core Architecture:
- BaseNode: Abstract base class with flexible input handling
- Specialized nodes: AgentNode, ToolNode, ParserNode, FlowNode, etc.

All nodes support N inputs natively via the inputs parameter.
"""

from pydantic_flow.nodes.agent import AgentNode as AgentNode
from pydantic_flow.nodes.base import BaseNode as BaseNode
from pydantic_flow.nodes.base import NodeOutput as NodeOutput
from pydantic_flow.nodes.conditional import IfNode as IfNode
from pydantic_flow.nodes.flow import FlowNode as FlowNode
from pydantic_flow.nodes.parser import ParserNode as ParserNode
from pydantic_flow.nodes.retry import RetryNode as RetryNode
from pydantic_flow.nodes.tool import ToolNode as ToolNode

__all__ = [
    "AgentNode",
    "BaseNode",
    "FlowNode",
    "IfNode",
    "NodeOutput",
    "ParserNode",
    "RetryNode",
    "ToolNode",
]
