"""Tests for InMemoryCache backend."""

import asyncio
from collections.abc import AsyncGenerator
import time

import pytest

from pydantic_flow.cache.base import CacheContentType
from pydantic_flow.cache.base import CacheEntry
from pydantic_flow.cache.memory import InMemoryCache


@pytest.fixture
async def cache() -> AsyncGenerator[InMemoryCache]:
    """Create a memory cache instance."""
    cache = InMemoryCache(max_entries=100, cleanup_interval=1.0)
    await cache.start()
    yield cache
    await cache.stop()


@pytest.mark.asyncio
async def test_set_and_get(cache: InMemoryCache) -> None:
    """Should store and retrieve entries."""
    entry = CacheEntry(
        value="test_value",
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=60,
    )

    await cache.set("key1", entry)
    retrieved = await cache.get("key1")

    assert retrieved is not None
    assert retrieved.value == "test_value"


@pytest.mark.asyncio
async def test_get_nonexistent(cache: InMemoryCache) -> None:
    """Should return None for nonexistent keys."""
    result = await cache.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_ttl_expiry(cache: InMemoryCache) -> None:
    """Should expire entries after TTL."""
    entry = CacheEntry(
        value="test_value",
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=1,
    )

    await cache.set("key1", entry)

    await asyncio.sleep(1.5)

    result = await cache.get("key1")
    assert result is None


@pytest.mark.asyncio
async def test_delete(cache: InMemoryCache) -> None:
    """Should delete entries."""
    entry = CacheEntry(
        value="test_value",
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
    )

    await cache.set("key1", entry)
    await cache.delete("key1")

    result = await cache.get("key1")
    assert result is None


@pytest.mark.asyncio
async def test_exists(cache: InMemoryCache) -> None:
    """Should check key existence."""
    entry = CacheEntry(
        value="test_value",
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
    )

    assert not await cache.exists("key1")

    await cache.set("key1", entry)
    assert await cache.exists("key1")


@pytest.mark.asyncio
async def test_lru_eviction() -> None:
    """Should evict least recently used entries when full."""
    cache = InMemoryCache(max_entries=3, cleanup_interval=10.0)
    await cache.start()

    try:
        for i in range(5):
            entry = CacheEntry(
                value=f"value_{i}",
                content_type=CacheContentType.LLM_COMPLETION,
                created_at=time.time(),
            )
            await cache.set(f"key_{i}", entry)

        size = await cache.size()
        assert size == 3

        result = await cache.get("key_0")
        assert result is None

        result = await cache.get("key_4")
        assert result is not None
    finally:
        await cache.stop()


@pytest.mark.asyncio
async def test_invalidate_namespace(cache: InMemoryCache) -> None:
    """Should invalidate all keys in a namespace."""
    for i in range(5):
        entry = CacheEntry(
            value=f"value_{i}",
            content_type=CacheContentType.LLM_COMPLETION,
            created_at=time.time(),
        )
        await cache.set(f"pf:ns:test:{i}", entry)

    entry = CacheEntry(
        value="other",
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
    )
    await cache.set("pf:ns:other:1", entry)

    deleted = await cache.invalidate_namespace("test")
    assert deleted == 5

    assert await cache.exists("pf:ns:other:1")


@pytest.mark.asyncio
async def test_clear(cache: InMemoryCache) -> None:
    """Should clear all entries."""
    for i in range(3):
        entry = CacheEntry(
            value=f"value_{i}",
            content_type=CacheContentType.LLM_COMPLETION,
            created_at=time.time(),
        )
        await cache.set(f"key_{i}", entry)

    await cache.clear()

    size = await cache.size()
    assert size == 0


@pytest.mark.asyncio
async def test_no_ttl(cache: InMemoryCache) -> None:
    """Should support entries without TTL."""
    entry = CacheEntry(
        value="permanent",
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=None,
    )

    await cache.set("key1", entry)
    await asyncio.sleep(0.5)

    result = await cache.get("key1")
    assert result is not None
    assert result.value == "permanent"
