"""Core streaming events for basic execution flow.

This module defines the fundamental streaming events that represent
the basic execution lifecycle: start, token emission, partial field updates,
and completion.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

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


class GenericResult(BaseModel):
    """Wrapper for non-BaseModel results.

    Attributes:
        value: The wrapped result value.

    """

    value: Any


class StreamEnd(ProgressItem):
    """Signals successful completion of a node's execution stream.

    Attributes:
        result: Optional final result as a BaseModel.

    """

    type: ProgressType = ProgressType.END
    result: BaseModel | None = None

    @field_validator("result", mode="before")
    @classmethod
    def wrap_non_basemodel(cls, v: Any) -> BaseModel | None:
        """Wrap non-BaseModel values in GenericResult."""
        if v is None:
            return None
        if isinstance(v, BaseModel):
            return v
        # Wrap other values (dict, primitives, etc.) in GenericResult
        return GenericResult(value=v)


class FlowResult(ProgressItem):
    """Final result of flow execution.

    Attributes:
        result: The final validated output data from the flow.

    """

    type: ProgressType = ProgressType.FLOW_RESULT
    result: Any = None
