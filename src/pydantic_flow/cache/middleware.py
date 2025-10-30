"""Cache middleware for node execution.

This module provides the central caching logic that wraps node execution,
checking cache before execution and writing results after.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
import sys
import time
from typing import Any
from typing import TypeVar

from pydantic_flow.cache.base import CacheBackend
from pydantic_flow.cache.base import CacheContentType
from pydantic_flow.cache.base import CacheEntry
from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.cache.events import CacheError
from pydantic_flow.cache.events import CacheHit
from pydantic_flow.cache.events import CacheMiss
from pydantic_flow.cache.events import CacheWrite
from pydantic_flow.cache.key import build_cache_key
from pydantic_flow.cache.stampede import Singleflight
from pydantic_flow.streaming.base import ProgressItem

T = TypeVar("T")

_singleflight = Singleflight()


async def maybe_cached_execute[T](  # noqa: PLR0913
    node_name: str,
    inputs: dict[str, Any],
    exec_fn: Callable[[], Awaitable[T]],
    backend: CacheBackend | None,
    policy: CachePolicy | None,
    context: dict[str, Any] | None = None,
) -> tuple[T, list[ProgressItem]]:
    """Execute a node with optional caching.

    This function wraps node execution with cache lookup and write logic.
    If caching is disabled or policy is missing, it calls exec_fn directly.

    Args:
        node_name: Name of the node being executed.
        inputs: Input data for the node.
        exec_fn: Async function that performs the actual computation.
        backend: Cache backend to use, or None to disable caching.
        policy: Cache policy, or None to disable caching.
        context: Optional execution context for cache key.

    Returns:
        Tuple of (result, cache_events).

    """
    cache_events: list[ProgressItem] = []

    if backend is None or policy is None or not policy.enabled or policy.bypass:
        result = await exec_fn()
        return result, cache_events

    try:
        key = build_cache_key(node_name, inputs, policy, context)
        current_time = time.time()

        from pydantic_flow.telemetry.helpers import record_counter
        from pydantic_flow.telemetry.helpers import traced_cache_lookup
        from pydantic_flow.telemetry.helpers import traced_cache_write

        # Cache lookup with telemetry
        async with traced_cache_lookup(node_name, backend.__class__.__name__, key[:16]):
            entry = await backend.get(key)

        if entry is not None:
            record_counter(
                "pflow.cache.hits",
                attributes={
                    "pflow.node.id": node_name,
                    "pflow.cache.backend": backend.__class__.__name__,
                    "pflow.cache.key_hash": key[:16],
                },
            )
            ttl_remaining = entry.ttl_remaining(current_time)
            cache_events.append(
                CacheHit(
                    node_id=node_name,
                    key=key,
                    backend=backend.__class__.__name__,
                    ttl_remaining=ttl_remaining,
                )
            )
            return entry.value, cache_events

        record_counter(
            "pflow.cache.misses",
            attributes={
                "pflow.node.id": node_name,
                "pflow.cache.backend": backend.__class__.__name__,
                "pflow.cache.key_hash": key[:16],
            },
        )
        cache_events.append(
            CacheMiss(
                node_id=node_name,
                key=key,
                backend=backend.__class__.__name__,
            )
        )

        async def compute_and_cache() -> T:
            """Compute result and write to cache."""
            result = await exec_fn()

            entry = CacheEntry(
                value=result,
                content_type=CacheContentType.LLM_COMPLETION,
                created_at=time.time(),
                ttl_seconds=policy.ttl_seconds(),
            )

            try:
                # Cache write with telemetry
                async with traced_cache_write(
                    node_name, backend.__class__.__name__, key[:16]
                ):
                    await backend.set(key, entry)

                value_size = sys.getsizeof(result)
                cache_events.append(
                    CacheWrite(
                        node_id=node_name,
                        key=key,
                        backend=backend.__class__.__name__,
                        value_size_bytes=value_size,
                    )
                )
            except Exception as write_error:
                cache_events.append(
                    CacheError(
                        node_id=node_name,
                        error=str(write_error),
                        operation="set",
                        key=key,
                    )
                )

            return result

        result = await _singleflight.do(key, compute_and_cache)
        return result, cache_events

    except Exception as error:
        cache_events.append(
            CacheError(
                node_id=node_name,
                error=str(error),
                operation="get",
                key=None,
            )
        )
        result = await exec_fn()
        return result, cache_events


async def maybe_cached_stream(  # noqa: PLR0913
    node_name: str,
    inputs: dict[str, Any],
    stream_fn: Callable[[], AsyncIterator[ProgressItem]],
    backend: CacheBackend | None,
    policy: CachePolicy | None,
    context: dict[str, Any] | None = None,
) -> AsyncIterator[ProgressItem]:
    """Execute a streaming node with optional caching.

    This function handles streaming execution with two modes:
    1. store_streams=False (default): Only cache final result
    2. store_streams=True: Capture and replay entire event stream

    Args:
        node_name: Name of the node being executed.
        inputs: Input data for the node.
        stream_fn: Async generator that yields progress items.
        backend: Cache backend to use, or None to disable caching.
        policy: Cache policy, or None to disable caching.
        context: Optional execution context for cache key.

    Yields:
        Progress items from execution or cache replay.

    """
    if backend is None or policy is None or not policy.enabled or policy.bypass:
        async for item in stream_fn():
            yield item
        return

    try:
        key = build_cache_key(node_name, inputs, policy, context)
        current_time = time.time()

        entry = await backend.get(key)
        if entry is not None:
            ttl_remaining = entry.ttl_remaining(current_time)
            yield CacheHit(
                node_id=node_name,
                key=key,
                backend=backend.__class__.__name__,
                ttl_remaining=ttl_remaining,
            )

            if (
                policy.store_streams
                and entry.content_type == CacheContentType.STREAM_EVENTS
            ):
                for event_dict in entry.value:
                    yield _deserialize_progress_item(event_dict)
            else:
                yield entry.value

            return

        yield CacheMiss(
            node_id=node_name,
            key=key,
            backend=backend.__class__.__name__,
        )

        if policy.store_streams:
            events: list[dict[str, Any]] = []
            async for item in stream_fn():
                events.append(_serialize_progress_item(item))
                yield item

            entry = CacheEntry(
                value=events,
                content_type=CacheContentType.STREAM_EVENTS,
                created_at=time.time(),
                ttl_seconds=policy.ttl_seconds(),
            )

            try:
                await backend.set(key, entry)
                value_size = sys.getsizeof(events)
                yield CacheWrite(
                    node_id=node_name,
                    key=key,
                    backend=backend.__class__.__name__,
                    value_size_bytes=value_size,
                )
            except Exception as write_error:
                yield CacheError(
                    node_id=node_name,
                    error=str(write_error),
                    operation="set",
                    key=key,
                )
        else:
            final_result = None
            async for item in stream_fn():
                yield item
                final_result = item

            if final_result is not None:
                entry = CacheEntry(
                    value=final_result,
                    content_type=CacheContentType.LLM_COMPLETION,
                    created_at=time.time(),
                    ttl_seconds=policy.ttl_seconds(),
                )

                try:
                    await backend.set(key, entry)
                    value_size = sys.getsizeof(final_result)
                    yield CacheWrite(
                        node_id=node_name,
                        key=key,
                        backend=backend.__class__.__name__,
                        value_size_bytes=value_size,
                    )
                except Exception as write_error:
                    yield CacheError(
                        node_id=node_name,
                        error=str(write_error),
                        operation="set",
                        key=key,
                    )

    except Exception as error:
        yield CacheError(
            node_id=node_name,
            error=str(error),
            operation="get",
            key=None,
        )
        async for item in stream_fn():
            yield item


def _serialize_progress_item(item: ProgressItem) -> dict[str, Any]:
    """Serialize a progress item for storage.

    Args:
        item: Progress item to serialize.

    Returns:
        Dictionary representation.

    """
    return {
        "type": item.type,
        "data": item.model_dump(),
    }


def _deserialize_progress_item(data: dict[str, Any]) -> ProgressItem:
    """Deserialize a progress item from storage.

    Args:
        data: Dictionary representation.

    Returns:
        Reconstructed progress item.

    """
    return ProgressItem(**data["data"])
