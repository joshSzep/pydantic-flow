"""Ergonomic streaming helpers for common use cases."""

from collections.abc import AsyncIterator
from typing import Any

from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.base import ProgressType
from pydantic_flow.streaming.core_events import GenericResult
from pydantic_flow.streaming.core_events import PartialFields
from pydantic_flow.streaming.core_events import TokenChunk


async def collect_result[T](stream: AsyncIterator[ProgressItem]) -> T:
    """Collect the final result from an astream() call.

    Preferred over the deprecated run() method. Extracts the result value
    from the stream's final progress item.

    For flows, this will wait for FlowResult (the aggregated flow output).
    For individual nodes, this will use StreamEnd (the node output).

    Args:
        stream: Progress item stream from node.astream() or flow.astream().

    Returns:
        The final result value, with GenericResult automatically unwrapped.

    Raises:
        ValueError: If stream completes without a result.

    """
    result = None
    flow_result = None

    async for item in stream:
        # Check if this is a FlowResult (takes priority)
        if item.type == ProgressType.FLOW_RESULT:
            flow_result = getattr(item, "result", None)
            # For flows, FlowResult is the final answer - break immediately
            if flow_result is not None:
                result = flow_result
                break
        # Otherwise, capture any result-bearing item (StreamEnd, etc)
        elif hasattr(item, "result"):
            candidate = getattr(item, "result", None)
            if candidate is not None:
                result = candidate

    if result is None:
        msg = "Stream completed without producing a result"
        raise ValueError(msg)

    # Unwrap GenericResult for non-BaseModel values
    if isinstance(result, GenericResult):
        return result.value  # type: ignore

    return result  # type: ignore


async def collect_final_result[T](stream: AsyncIterator[ProgressItem]) -> T:
    """Alias for collect_result() - maintained for backward compatibility."""
    return await collect_result(stream)


async def iter_tokens(stream: AsyncIterator[ProgressItem]) -> AsyncIterator[str]:
    """Extract only text tokens from a progress stream.

    Useful for CLIs and demos that just want to display text.

    Args:
        stream: Progress item stream.

    Yields:
        Text content from TokenChunk items.

    """
    async for item in stream:
        if isinstance(item, TokenChunk):
            yield item.text


async def iter_fields(
    stream: AsyncIterator[ProgressItem],
) -> AsyncIterator[dict[str, Any]]:
    """Extract only partial field updates from a progress stream.

    Useful for observing structured field formation without handling
    all progress types.

    Args:
        stream: Progress item stream.

    Yields:
        Field update dictionaries from PartialFields items.

    """
    async for item in stream:
        if isinstance(item, PartialFields):
            yield item.fields


async def collect_all_tokens(stream: AsyncIterator[ProgressItem]) -> str:
    """Consume a stream and concatenate all tokens into a single string.

    Args:
        stream: Progress item stream.

    Returns:
        Concatenated text from all TokenChunk items.

    """
    tokens = []
    async for item in stream:
        if isinstance(item, TokenChunk):
            tokens.append(item.text)
    return "".join(tokens)
