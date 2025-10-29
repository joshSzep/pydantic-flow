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
from pydantic_flow.cache.key import build_cache_key
from pydantic_flow.cache.stampede import Singleflight
from pydantic_flow.streaming.events import CacheError
from pydantic_flow.streaming.events import CacheHit
from pydantic_flow.streaming.events import CacheMiss
from pydantic_flow.streaming.events import CacheWrite
from pydantic_flow.streaming.events import ProgressItem

T = TypeVar("T")

_singleflight = Singleflight()


async def maybe_cached_execute[T](  # noqa: PLR0913, PLR0915
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

        # Telemetry: check if enabled before importing/instrumenting
        from contextlib import nullcontext

        from pydantic_flow.telemetry.setup import is_enabled

        telemetry_enabled = is_enabled()
        cache_attrs: dict[str, Any] = {}

        if telemetry_enabled:
            from pydantic_flow.telemetry.attributes import AttributeKey
            from pydantic_flow.telemetry.attributes import MetricName
            from pydantic_flow.telemetry.attributes import SpanKind
            from pydantic_flow.telemetry.helpers import create_span_async
            from pydantic_flow.telemetry.helpers import measure_duration_async
            from pydantic_flow.telemetry.helpers import record_counter

            cache_attrs = {
                str(AttributeKey.NODE_ID): node_name,
                str(AttributeKey.CACHE_BACKEND): backend.__class__.__name__,
                str(AttributeKey.CACHE_KEY_HASH): key[:16],
            }

            record_counter(MetricName.CACHE_LOOKUPS, attributes=cache_attrs)

            lookup_span_ctx = create_span_async(
                SpanKind.CACHE_LOOKUP, attributes=cache_attrs
            )
            lookup_duration_ctx = measure_duration_async(
                MetricName.CACHE_LOOKUP_DURATION, attributes=cache_attrs
            )
        else:
            lookup_span_ctx = nullcontext()
            lookup_duration_ctx = nullcontext()

        async with lookup_span_ctx, lookup_duration_ctx:
            entry = await backend.get(key)

        if entry is not None:
            if telemetry_enabled:
                record_counter(MetricName.CACHE_HITS, attributes=cache_attrs)  # type: ignore[possibly-undefined]
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

        if telemetry_enabled:
            record_counter(MetricName.CACHE_MISSES, attributes=cache_attrs)  # type: ignore[possibly-undefined]
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
                # Telemetry for cache write
                if telemetry_enabled:
                    from pydantic_flow.telemetry.attributes import (
                        MetricName as _MetricName,
                    )
                    from pydantic_flow.telemetry.attributes import SpanKind as _SpanKind
                    from pydantic_flow.telemetry.helpers import (
                        create_span_async as _create_span,
                    )
                    from pydantic_flow.telemetry.helpers import (
                        measure_duration_async as _measure,
                    )
                    from pydantic_flow.telemetry.helpers import (
                        record_counter as _counter,
                    )

                    write_attrs: dict[str, Any] = {**cache_attrs}
                    _counter(_MetricName.CACHE_WRITES, attributes=write_attrs)

                    write_span_ctx = _create_span(
                        _SpanKind.CACHE_WRITE, attributes=write_attrs
                    )
                    write_duration_ctx = _measure(
                        _MetricName.CACHE_WRITE_DURATION, attributes=write_attrs
                    )
                else:
                    write_span_ctx = nullcontext()
                    write_duration_ctx = nullcontext()

                async with write_span_ctx, write_duration_ctx:
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
