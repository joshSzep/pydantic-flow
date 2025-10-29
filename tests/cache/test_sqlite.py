"""Tests for SQLite cache backend."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
import tempfile
import time

import pytest

from pydantic_flow.cache.base import CacheContentType
from pydantic_flow.cache.base import CacheEntry
from pydantic_flow.cache.sqlite import SQLiteCache


@pytest.fixture
async def temp_db_path() -> AsyncGenerator[Path]:
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    yield path
    # Cleanup
    if path.exists():
        path.unlink()
    wal_path = path.with_suffix(".db-wal")
    if wal_path.exists():
        wal_path.unlink()
    shm_path = path.with_suffix(".db-shm")
    if shm_path.exists():
        shm_path.unlink()


@pytest.fixture
async def cache(temp_db_path: Path) -> AsyncGenerator[SQLiteCache]:
    """Create a SQLite cache instance."""
    cache = SQLiteCache(db_path=temp_db_path, cleanup_interval=0.5)
    await cache.start()
    yield cache
    await cache.stop()


@pytest.mark.asyncio
async def test_set_and_get(cache: SQLiteCache) -> None:
    """Test basic get and set operations."""
    entry = CacheEntry(
        value={"result": "test_value"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=3600,
    )

    await cache.set("test_key", entry)
    retrieved = await cache.get("test_key")

    assert retrieved is not None
    assert retrieved.value == {"result": "test_value"}
    assert retrieved.content_type == CacheContentType.LLM_COMPLETION


@pytest.mark.asyncio
async def test_get_nonexistent(cache: SQLiteCache) -> None:
    """Test getting a key that doesn't exist."""
    result = await cache.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_ttl_expiry(cache: SQLiteCache) -> None:
    """Test that entries expire after their TTL."""
    entry = CacheEntry(
        value={"result": "expires_soon"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=1,
    )

    await cache.set("expires", entry)

    # Should exist immediately
    result = await cache.get("expires")
    assert result is not None

    # Wait for expiry
    await asyncio.sleep(1.5)

    # Should be expired now
    result = await cache.get("expires")
    assert result is None


@pytest.mark.asyncio
async def test_delete(cache: SQLiteCache) -> None:
    """Test deleting entries."""
    entry = CacheEntry(
        value={"result": "to_delete"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=3600,
    )

    await cache.set("delete_me", entry)
    assert await cache.exists("delete_me")

    await cache.delete("delete_me")
    assert not await cache.exists("delete_me")


@pytest.mark.asyncio
async def test_exists(cache: SQLiteCache) -> None:
    """Test checking key existence."""
    assert not await cache.exists("new_key")

    entry = CacheEntry(
        value={"result": "exists"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=3600,
    )
    await cache.set("new_key", entry)

    assert await cache.exists("new_key")


@pytest.mark.asyncio
async def test_invalidate_namespace(cache: SQLiteCache) -> None:
    """Test namespace invalidation."""
    entry = CacheEntry(
        value={"result": "value"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=3600,
    )

    await cache.set("ns1_key1", entry, namespace="namespace1")
    await cache.set("ns1_key2", entry, namespace="namespace1")
    await cache.set("ns2_key1", entry, namespace="namespace2")

    # Invalidate namespace1
    deleted = await cache.invalidate_namespace("namespace1")
    assert deleted == 2

    # namespace1 keys should be gone
    assert not await cache.exists("ns1_key1")
    assert not await cache.exists("ns1_key2")

    # namespace2 key should still exist
    assert await cache.exists("ns2_key1")


@pytest.mark.asyncio
async def test_clear(cache: SQLiteCache) -> None:
    """Test clearing all entries."""
    entry = CacheEntry(
        value={"result": "value"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=3600,
    )

    await cache.set("key1", entry)
    await cache.set("key2", entry)
    await cache.set("key3", entry)

    await cache.clear()

    assert not await cache.exists("key1")
    assert not await cache.exists("key2")
    assert not await cache.exists("key3")


@pytest.mark.asyncio
async def test_no_ttl(cache: SQLiteCache) -> None:
    """Test entries with no TTL never expire."""
    entry = CacheEntry(
        value={"result": "permanent"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=None,
    )

    await cache.set("permanent", entry)

    # Wait a bit
    await asyncio.sleep(0.5)

    # Should still exist
    result = await cache.get("permanent")
    assert result is not None
    assert result.value == {"result": "permanent"}


@pytest.mark.asyncio
async def test_cleanup_task(cache: SQLiteCache) -> None:
    """Test that background cleanup removes expired entries."""
    entry = CacheEntry(
        value={"result": "expires_soon"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=1,
    )

    await cache.set("auto_cleanup", entry)

    # Wait for expiry and cleanup (cleanup_interval=0.5)
    await asyncio.sleep(2.0)

    # Entry should have been cleaned up
    result = await cache.get("auto_cleanup")
    assert result is None


@pytest.mark.asyncio
async def test_persistence_across_reopens(temp_db_path: Path) -> None:
    """Test that data persists when cache is reopened."""
    # Create cache and store value
    cache1 = SQLiteCache(db_path=temp_db_path)
    await cache1.start()

    entry = CacheEntry(
        value={"result": "persistent"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=3600,
    )
    await cache1.set("persistent_key", entry)
    await cache1.stop()

    # Reopen cache
    cache2 = SQLiteCache(db_path=temp_db_path)
    await cache2.start()

    # Data should still be there
    result = await cache2.get("persistent_key")
    assert result is not None
    assert result.value == {"result": "persistent"}

    await cache2.stop()


@pytest.mark.asyncio
async def test_context_manager(temp_db_path: Path) -> None:
    """Test using cache as async context manager."""
    entry = CacheEntry(
        value={"result": "context"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=3600,
    )

    async with SQLiteCache(db_path=temp_db_path) as cache:
        await cache.set("ctx_key", entry)
        result = await cache.get("ctx_key")
        assert result is not None
        assert result.value == {"result": "context"}


@pytest.mark.asyncio
async def test_concurrent_writes(cache: SQLiteCache) -> None:
    """Test concurrent writes to the cache."""

    async def write_entry(key: str, value: str) -> None:
        entry = CacheEntry(
            value={"result": value},
            content_type=CacheContentType.LLM_COMPLETION,
            created_at=time.time(),
            ttl_seconds=3600,
        )
        await cache.set(key, entry)

    # Write 10 entries concurrently
    await asyncio.gather(*[write_entry(f"key_{i}", f"value_{i}") for i in range(10)])

    # All entries should be present
    for i in range(10):
        result = await cache.get(f"key_{i}")
        assert result is not None
        assert result.value == {"result": f"value_{i}"}


@pytest.mark.asyncio
async def test_update_existing_entry(cache: SQLiteCache) -> None:
    """Test updating an existing entry."""
    entry1 = CacheEntry(
        value={"result": "original"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=3600,
    )
    await cache.set("update_key", entry1)

    entry2 = CacheEntry(
        value={"result": "updated"},
        content_type=CacheContentType.EMBEDDING_VECTOR,
        created_at=time.time(),
        ttl_seconds=7200,
    )
    await cache.set("update_key", entry2)

    result = await cache.get("update_key")
    assert result is not None
    assert result.value == {"result": "updated"}
    assert result.content_type == CacheContentType.EMBEDDING_VECTOR
    assert result.ttl_seconds == 7200


@pytest.mark.asyncio
async def test_different_content_types(cache: SQLiteCache) -> None:
    """Test storing different content types."""
    llm_entry = CacheEntry(
        value={"response": "llm"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=3600,
    )
    await cache.set("llm", llm_entry)

    embedding_entry = CacheEntry(
        value={"vector": [0.1, 0.2, 0.3]},
        content_type=CacheContentType.EMBEDDING_VECTOR,
        created_at=time.time(),
        ttl_seconds=3600,
    )
    await cache.set("embedding", embedding_entry)

    node_entry = CacheEntry(
        value={"output": "node_result"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=3600,
    )
    await cache.set("node", node_entry)

    # Verify all types
    llm = await cache.get("llm")
    assert llm is not None
    assert llm.content_type == CacheContentType.LLM_COMPLETION

    emb = await cache.get("embedding")
    assert emb is not None
    assert emb.content_type == CacheContentType.EMBEDDING_VECTOR

    node = await cache.get("node")
    assert node is not None
    assert node.content_type == CacheContentType.LLM_COMPLETION
