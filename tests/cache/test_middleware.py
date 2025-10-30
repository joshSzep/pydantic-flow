"""Direct tests for cache middleware functions.

These tests directly exercise maybe_cached_execute and maybe_cached_stream
to achieve comprehensive coverage.
"""

from datetime import timedelta
from unittest.mock import AsyncMock

from pydantic import BaseModel
import pytest

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.cache.events import CacheError
from pydantic_flow.cache.events import CacheHit
from pydantic_flow.cache.events import CacheMiss
from pydantic_flow.cache.events import CacheWrite
from pydantic_flow.cache.memory import InMemoryCache
from pydantic_flow.cache.middleware import maybe_cached_execute
from pydantic_flow.cache.middleware import maybe_cached_stream
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.core_events import TokenChunk


class ResultModel(BaseModel):
    """Test result model."""

    value: int


@pytest.mark.asyncio
async def test_maybe_cached_execute_no_backend():
    """Test execute with no cache backend."""
    call_count = 0

    async def exec_fn():
        nonlocal call_count
        call_count += 1
        return ResultModel(value=42)

    policy = CachePolicy(enabled=True, ttl=timedelta(seconds=60))
    result, events = await maybe_cached_execute(
        node_name="test",
        inputs={"x": 1},
        exec_fn=exec_fn,
        backend=None,  # No backend
        policy=policy,
    )

    assert result.value == 42
    assert call_count == 1
    assert len(events) == 0  # No cache events


@pytest.mark.asyncio
async def test_maybe_cached_execute_no_policy():
    """Test execute with no cache policy."""
    cache = InMemoryCache()
    call_count = 0

    async def exec_fn():
        nonlocal call_count
        call_count += 1
        return ResultModel(value=42)

    result, events = await maybe_cached_execute(
        node_name="test",
        inputs={"x": 1},
        exec_fn=exec_fn,
        backend=cache,
        policy=None,  # No policy
    )

    assert result.value == 42
    assert call_count == 1
    assert len(events) == 0


@pytest.mark.asyncio
async def test_maybe_cached_execute_disabled():
    """Test execute with disabled cache."""
    cache = InMemoryCache()
    call_count = 0

    async def exec_fn():
        nonlocal call_count
        call_count += 1
        return ResultModel(value=42)

    policy = CachePolicy(enabled=False, ttl=timedelta(seconds=60))
    result, events = await maybe_cached_execute(
        node_name="test",
        inputs={"x": 1},
        exec_fn=exec_fn,
        backend=cache,
        policy=policy,
    )

    assert result.value == 42
    assert call_count == 1
    assert len(events) == 0


@pytest.mark.asyncio
async def test_maybe_cached_execute_bypass():
    """Test execute with bypass enabled."""
    cache = InMemoryCache()
    call_count = 0

    async def exec_fn():
        nonlocal call_count
        call_count += 1
        return ResultModel(value=42)

    policy = CachePolicy(enabled=True, bypass=True, ttl=timedelta(seconds=60))
    result, events = await maybe_cached_execute(
        node_name="test",
        inputs={"x": 1},
        exec_fn=exec_fn,
        backend=cache,
        policy=policy,
    )

    assert result.value == 42
    assert call_count == 1
    assert len(events) == 0  # Bypass means no cache events


@pytest.mark.asyncio
async def test_maybe_cached_execute_cache_miss_and_hit():
    """Test execute with cache miss then hit."""
    cache = InMemoryCache()
    call_count = 0

    async def exec_fn():
        nonlocal call_count
        call_count += 1
        return ResultModel(value=42)

    policy = CachePolicy(enabled=True, ttl=timedelta(seconds=60))

    # First call - cache miss
    result1, events1 = await maybe_cached_execute(
        node_name="test",
        inputs={"x": 1},
        exec_fn=exec_fn,
        backend=cache,
        policy=policy,
    )

    assert result1.value == 42
    assert call_count == 1
    assert len(events1) == 2
    assert isinstance(events1[0], CacheMiss)
    assert isinstance(events1[1], CacheWrite)

    # Second call - cache hit
    result2, events2 = await maybe_cached_execute(
        node_name="test",
        inputs={"x": 1},
        exec_fn=exec_fn,
        backend=cache,
        policy=policy,
    )

    assert result2.value == 42
    assert call_count == 1  # Not called again
    assert len(events2) == 1
    assert isinstance(events2[0], CacheHit)
    assert events2[0].ttl_remaining is not None


@pytest.mark.asyncio
async def test_maybe_cached_execute_write_error():
    """Test execute when cache write fails."""
    # Create mock backend that fails on set
    cache = AsyncMock()
    cache.get.return_value = None
    cache.set.side_effect = Exception("Write failed")
    cache.__class__.__name__ = "MockCache"

    call_count = 0

    async def exec_fn():
        nonlocal call_count
        call_count += 1
        return ResultModel(value=42)

    policy = CachePolicy(enabled=True, ttl=timedelta(seconds=60))

    result, events = await maybe_cached_execute(
        node_name="test",
        inputs={"x": 1},
        exec_fn=exec_fn,
        backend=cache,
        policy=policy,
    )

    assert result.value == 42
    assert call_count == 1
    assert len(events) == 2
    assert isinstance(events[0], CacheMiss)
    assert isinstance(events[1], CacheError)
    assert events[1].operation == "set"
    assert "Write failed" in events[1].error


@pytest.mark.asyncio
async def test_maybe_cached_execute_get_error():
    """Test execute when cache get fails."""
    # Create mock backend that fails on get
    cache = AsyncMock()
    cache.get.side_effect = Exception("Get failed")
    cache.__class__.__name__ = "MockCache"

    call_count = 0

    async def exec_fn():
        nonlocal call_count
        call_count += 1
        return ResultModel(value=42)

    policy = CachePolicy(enabled=True, ttl=timedelta(seconds=60))

    result, events = await maybe_cached_execute(
        node_name="test",
        inputs={"x": 1},
        exec_fn=exec_fn,
        backend=cache,
        policy=policy,
    )

    assert result.value == 42
    assert call_count == 1
    assert len(events) == 1
    assert isinstance(events[0], CacheError)
    assert events[0].operation == "get"
    assert "Get failed" in events[0].error


@pytest.mark.asyncio
async def test_maybe_cached_stream_no_backend():
    """Test stream with no cache backend."""

    async def stream_fn():
        yield StreamStart(node_id="test")
        yield TokenChunk(node_id="test", text="hello")
        yield StreamEnd(node_id="test")

    policy = CachePolicy(enabled=True, ttl=timedelta(seconds=60))

    events = []
    async for item in maybe_cached_stream(
        node_name="test",
        inputs={"x": 1},
        stream_fn=stream_fn,
        backend=None,
        policy=policy,
    ):
        events.append(item)

    assert len(events) == 3
    assert isinstance(events[0], StreamStart)
    assert isinstance(events[1], TokenChunk)
    assert isinstance(events[2], StreamEnd)


@pytest.mark.asyncio
async def test_maybe_cached_stream_disabled():
    """Test stream with disabled cache."""
    cache = InMemoryCache()

    async def stream_fn():
        yield StreamStart(node_id="test")
        yield TokenChunk(node_id="test", text="hello")
        yield StreamEnd(node_id="test")

    policy = CachePolicy(enabled=False, ttl=timedelta(seconds=60))

    events = []
    async for item in maybe_cached_stream(
        node_name="test",
        inputs={"x": 1},
        stream_fn=stream_fn,
        backend=cache,
        policy=policy,
    ):
        events.append(item)

    assert len(events) == 3


@pytest.mark.asyncio
async def test_maybe_cached_stream_miss_and_write():
    """Test stream with cache miss and write (final result mode)."""
    cache = InMemoryCache()

    async def stream_fn():
        yield StreamStart(node_id="test")
        yield TokenChunk(node_id="test", text="hello")
        yield StreamEnd(node_id="test")

    policy = CachePolicy(enabled=True, ttl=timedelta(seconds=60))

    events = []
    async for item in maybe_cached_stream(
        node_name="test",
        inputs={"x": 1},
        stream_fn=stream_fn,
        backend=cache,
        policy=policy,
    ):
        events.append(item)

    # Should have: CacheMiss, StreamStart, TokenChunk, StreamEnd, CacheWrite
    assert len(events) == 5
    assert isinstance(events[0], CacheMiss)
    assert isinstance(events[1], StreamStart)
    assert isinstance(events[2], TokenChunk)
    assert isinstance(events[3], StreamEnd)
    assert isinstance(events[4], CacheWrite)


@pytest.mark.asyncio
async def test_maybe_cached_stream_hit():
    """Test stream with cache hit (final result mode)."""
    cache = InMemoryCache()

    async def stream_fn():
        yield StreamStart(node_id="test")
        yield StreamEnd(node_id="test")

    policy = CachePolicy(enabled=True, ttl=timedelta(seconds=60))

    # First run to populate cache
    events1 = []
    async for item in maybe_cached_stream(
        node_name="test",
        inputs={"x": 1},
        stream_fn=stream_fn,
        backend=cache,
        policy=policy,
    ):
        events1.append(item)

    # Second run - should hit cache
    events2 = []
    async for item in maybe_cached_stream(
        node_name="test",
        inputs={"x": 1},
        stream_fn=stream_fn,
        backend=cache,
        policy=policy,
    ):
        events2.append(item)

    # Should have: CacheHit, StreamEnd (cached result)
    assert len(events2) == 2
    assert isinstance(events2[0], CacheHit)
    assert isinstance(events2[1], StreamEnd)


@pytest.mark.asyncio
async def test_maybe_cached_stream_store_streams():
    """Test stream with store_streams=True."""
    cache = InMemoryCache()

    async def stream_fn():
        yield StreamStart(node_id="test")
        yield TokenChunk(node_id="test", text="hello")
        yield StreamEnd(node_id="test")

    policy = CachePolicy(
        enabled=True,
        ttl=timedelta(seconds=60),
        store_streams=True,
    )

    # First run - capture events
    events1 = []
    async for item in maybe_cached_stream(
        node_name="test",
        inputs={"x": 1},
        stream_fn=stream_fn,
        backend=cache,
        policy=policy,
    ):
        events1.append(item)

    # Should have: CacheMiss, then all stream events, then CacheWrite
    assert len(events1) == 5
    assert isinstance(events1[0], CacheMiss)
    assert isinstance(events1[-1], CacheWrite)

    # Second run - replay events from cache
    events2 = []
    async for item in maybe_cached_stream(
        node_name="test",
        inputs={"x": 1},
        stream_fn=stream_fn,
        backend=cache,
        policy=policy,
    ):
        events2.append(item)

    # Should have: CacheHit, then all replayed stream events
    assert len(events2) == 4
    assert isinstance(events2[0], CacheHit)
    # Replayed events are base ProgressItem, check type field
    assert events2[1].type.value == "start"
    assert events2[2].type.value == "token"
    assert events2[3].type.value == "end"


@pytest.mark.asyncio
async def test_maybe_cached_stream_write_error():
    """Test stream when cache write fails."""
    cache = AsyncMock()
    cache.get.return_value = None
    cache.set.side_effect = Exception("Write failed")
    cache.__class__.__name__ = "MockCache"

    async def stream_fn():
        yield StreamStart(node_id="test")
        yield StreamEnd(node_id="test")

    policy = CachePolicy(enabled=True, ttl=timedelta(seconds=60))

    events = []
    async for item in maybe_cached_stream(
        node_name="test",
        inputs={"x": 1},
        stream_fn=stream_fn,
        backend=cache,
        policy=policy,
    ):
        events.append(item)

    # Should have: CacheMiss, StreamStart, StreamEnd, CacheError
    assert len(events) == 4
    assert isinstance(events[0], CacheMiss)
    assert isinstance(events[-1], CacheError)
    assert events[-1].operation == "set"


@pytest.mark.asyncio
async def test_maybe_cached_stream_get_error():
    """Test stream when cache get fails."""
    cache = AsyncMock()
    cache.get.side_effect = Exception("Get failed")
    cache.__class__.__name__ = "MockCache"

    async def stream_fn():
        yield StreamStart(node_id="test")
        yield StreamEnd(node_id="test")

    policy = CachePolicy(enabled=True, ttl=timedelta(seconds=60))

    events = []
    async for item in maybe_cached_stream(
        node_name="test",
        inputs={"x": 1},
        stream_fn=stream_fn,
        backend=cache,
        policy=policy,
    ):
        events.append(item)

    # Should have: CacheError, then normal stream
    assert len(events) == 3
    assert isinstance(events[0], CacheError)
    assert events[0].operation == "get"
    assert isinstance(events[1], StreamStart)
    assert isinstance(events[2], StreamEnd)


@pytest.mark.asyncio
async def test_maybe_cached_stream_store_streams_write_error():
    """Test stream with store_streams=True when write fails."""
    cache = AsyncMock()
    cache.get.return_value = None
    cache.set.side_effect = Exception("Write failed")
    cache.__class__.__name__ = "MockCache"

    async def stream_fn():
        yield StreamStart(node_id="test")
        yield TokenChunk(node_id="test", text="hello")
        yield StreamEnd(node_id="test")

    policy = CachePolicy(
        enabled=True,
        ttl=timedelta(seconds=60),
        store_streams=True,
    )

    events = []
    async for item in maybe_cached_stream(
        node_name="test",
        inputs={"x": 1},
        stream_fn=stream_fn,
        backend=cache,
        policy=policy,
    ):
        events.append(item)

    # Should have: CacheMiss, stream events, CacheError
    assert len(events) == 5
    assert isinstance(events[0], CacheMiss)
    assert isinstance(events[-1], CacheError)
    assert events[-1].operation == "set"
