"""Streaming event types for progress reporting.

This module defines a small, focused vocabulary of progress items
that nodes emit during streaming execution.
"""

from datetime import UTC
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class ProgressType(StrEnum):
    """Type discriminator for progress items.

    Attributes:
        START: Stream begins.
        TOKEN: Text token from LLM.
        PARTIAL_FIELDS: Incremental structured field updates.
        TOOL_CALL: Tool invocation intent declared.
        TOOL_ARG_PROGRESS: Tool argument formation in progress.
        TOOL_RESULT: Tool execution result.
        RETRIEVAL: Retrieved item from search/db.
        METRIC: Performance or quality metric.
        ERROR: Non-fatal error or warning.
        END: Stream completes successfully.
        HEARTBEAT: Liveness signal during long operation.

    """

    START = "start"
    TOKEN = "token"
    PARTIAL_FIELDS = "partial_fields"
    TOOL_CALL = "tool_call"
    TOOL_ARG_PROGRESS = "tool_arg_progress"
    TOOL_RESULT = "tool_result"
    RETRIEVAL = "retrieval"
    METRIC = "metric"
    ERROR = "error"
    END = "end"
    HEARTBEAT = "heartbeat"


class ProgressItem(BaseModel):
    """Base class for all streaming progress events.

    Attributes:
        type: Discriminator for the progress item type.
        timestamp: When the event occurred.
        run_id: Unique identifier for this execution run.
        node_id: Identifier of the node emitting this event.

    """

    model_config = {"frozen": True}

    type: ProgressType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str = ""
    node_id: str = ""


class StreamStart(ProgressItem):
    """Signals the start of a node's execution stream.

    Attributes:
        input_preview: Optional preview of input data.

    """

    type: ProgressType = ProgressType.START
    input_preview: dict[str, Any] | None = None


class TokenChunk(ProgressItem):
    """A single token or text fragment from an LLM.

    Attributes:
        text: The token text.
        token_index: Optional position in the full sequence.

    """

    type: ProgressType = ProgressType.TOKEN
    text: str = ""
    token_index: int | None = None


class PartialFields(ProgressItem):
    """Incremental structured field updates.

    As structured output forms, this carries partial field values
    that can be used to show progress before final validation.

    Attributes:
        fields: Dict of field names to partially extracted values.

    """

    type: ProgressType = ProgressType.PARTIAL_FIELDS
    fields: dict[str, Any] = Field(default_factory=dict)


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


class RetrievalItem(ProgressItem):
    """A single retrieved item from search or database.

    Attributes:
        item_id: Identifier for the retrieved item.
        content: The retrieved content.
        score: Optional relevance score.
        metadata: Optional additional metadata.

    """

    type: ProgressType = ProgressType.RETRIEVAL
    item_id: str = ""
    content: Any = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NonFatalError(ProgressItem):
    """A non-fatal error or warning during execution.

    Attributes:
        message: Error description.
        recoverable: Whether execution can continue.

    """

    type: ProgressType = ProgressType.ERROR
    message: str = ""
    recoverable: bool = True


class StreamEnd(ProgressItem):
    """Signals successful completion of a node's execution stream.

    Attributes:
        result_preview: Optional preview of final result.

    """

    type: ProgressType = ProgressType.END
    result_preview: dict[str, Any] | None = None


class Heartbeat(ProgressItem):
    """Liveness signal during long-running operations.

    Attributes:
        message: Optional status message.

    """

    type: ProgressType = ProgressType.HEARTBEAT
    message: str = ""
