"""Tool-related streaming events.

This module defines events related to tool invocation and execution,
including tool calls, argument progress, and results.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.base import ProgressType


class ToolCall(ProgressItem):
    """Tool invocation intent declared by the agent.

    Attributes:
        tool_name: Name of the tool being invoked.
        call_id: Unique identifier for this specific call.

    """

    type: ProgressType = ProgressType.TOOL_CALL
    tool_name: str = ""
    call_id: str = ""


class ToolArgProgress(ProgressItem):
    """Tool argument formation in progress.

    Attributes:
        tool_name: Name of the tool.
        call_id: Unique identifier for this specific call.
        partial_args: Partially formed argument dict.

    """

    type: ProgressType = ProgressType.TOOL_ARG_PROGRESS
    tool_name: str = ""
    call_id: str = ""
    partial_args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(ProgressItem):
    """Tool execution result.

    Attributes:
        tool_name: Name of the tool.
        call_id: Unique identifier for this specific call.
        result: The tool's return value.
        error: Error message if the tool failed.

    """

    type: ProgressType = ProgressType.TOOL_RESULT
    tool_name: str = ""
    call_id: str = ""
    result: Any = None
    error: str | None = None
