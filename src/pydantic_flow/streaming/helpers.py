"""Ergonomic streaming helpers for common use cases."""

from collections.abc import AsyncIterator
from typing import Any

from pydantic_flow.streaming.events import PartialFields
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import ProgressType
from pydantic_flow.streaming.events import TokenChunk


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


async def collect_final_result(stream: AsyncIterator[ProgressItem]) -> Any:
    """Consume a stream and return only the final result.

    Args:
        stream: Progress item stream.

    Returns:
        The result_preview from the StreamEnd item.

    """
    final_result = None
    async for item in stream:
        if item.type == ProgressType.END:
            final_result = item.result_preview  # type: ignore
    return final_result


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
