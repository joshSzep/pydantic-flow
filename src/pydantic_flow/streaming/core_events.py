"""Core streaming events for basic execution flow.

This module defines the fundamental streaming events that represent
the basic execution lifecycle: start, token emission, partial field updates,
and completion.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.base import ProgressType


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


class StreamEnd(ProgressItem):
    """Signals successful completion of a node's execution stream.

    Attributes:
        result_preview: Optional preview of final result.

    """

    type: ProgressType = ProgressType.END
    result_preview: dict[str, Any] | None = None


class FlowResult(ProgressItem):
    """Final result of flow execution.

    Attributes:
        result: The final validated output data from the flow.

    """

    type: ProgressType = ProgressType.FLOW_RESULT
    result: Any = None
