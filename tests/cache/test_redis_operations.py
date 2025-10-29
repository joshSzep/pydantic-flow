"""Comprehensive tests for RedisCache operations with mocked Redis.

These tests achieve high coverage of Redis cache functionality without
requiring a real Redis server.
"""

import json
import time
from unittest.mock import AsyncMock
from unittest.mock import patch
import zlib

import pytest

from pydantic_flow.cache.base import CacheContentType
from pydantic_flow.cache.base import CacheEntry
from pydantic_flow.cache.redis import RedisCache


@pytest.mark.asyncio
async def test_get_existing_entry():
    """Test retrieving an existing non-expired entry."""
    mock_client = AsyncMock()

    # Create a valid entry
    entry = CacheEntry(
        value={"result": "test"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=3600,
    )

    # Serialize it like Redis would store it
    entry_dict = entry.model_dump()
    json_bytes = json.dumps(entry_dict).encode("utf-8")
    mock_client.get.return_value = b"J" + json_bytes

    cache = RedisCache(redis=mock_client, key_prefix="test")
    result = await cache.get("test_key")

    assert result is not None
    assert result.value == {"result": "test"}
    mock_client.get.assert_called_once_with("test:test_key")


@pytest.mark.asyncio
async def test_get_nonexistent_entry():
    """Test retrieving a non-existent entry."""
    mock_client = AsyncMock()
    mock_client.get.return_value = None

    cache = RedisCache(redis=mock_client)
    result = await cache.get("missing_key")

    assert result is None


@pytest.mark.asyncio
async def test_get_expired_entry():
    """Test that expired entries are deleted and return None."""
    mock_client = AsyncMock()

    # Create an expired entry
    entry = CacheEntry(
        value={"result": "test"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time() - 7200,  # 2 hours ago
        ttl_seconds=3600,  # 1 hour TTL
    )

    entry_dict = entry.model_dump()
    json_bytes = json.dumps(entry_dict).encode("utf-8")
    mock_client.get.return_value = b"J" + json_bytes
    mock_client.delete = AsyncMock()

    cache = RedisCache(redis=mock_client, key_prefix="test")
    result = await cache.get("expired_key")

    assert result is None
    # Should delete the expired entry
    mock_client.delete.assert_called_once_with("test:expired_key")


@pytest.mark.asyncio
async def test_set_with_ttl():
    """Test storing entry with TTL."""
    mock_client = AsyncMock()
    mock_client.setex = AsyncMock()

    entry = CacheEntry(
        value={"data": "test"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=300,
    )

    cache = RedisCache(redis=mock_client, key_prefix="test")
    await cache.set("test_key", entry)

    mock_client.setex.assert_called_once()
    call_args = mock_client.setex.call_args
    assert call_args[0][0] == "test:test_key"
    assert call_args[0][1] == 300  # TTL seconds
    assert call_args[0][2].startswith(b"J")  # JSON marker


@pytest.mark.asyncio
async def test_set_without_ttl():
    """Test storing entry without TTL (permanent)."""
    mock_client = AsyncMock()
    mock_client.set = AsyncMock()

    entry = CacheEntry(
        value={"data": "test"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=None,
    )

    cache = RedisCache(redis=mock_client, key_prefix="test")
    await cache.set("test_key", entry)

    mock_client.set.assert_called_once()
    call_args = mock_client.set.call_args
    assert call_args[0][0] == "test:test_key"
    assert call_args[0][1].startswith(b"J")


@pytest.mark.asyncio
async def test_delete_key():
    """Test deleting a key."""
    mock_client = AsyncMock()
    mock_client.delete = AsyncMock()

    cache = RedisCache(redis=mock_client, key_prefix="test")
    await cache.delete("test_key")

    mock_client.delete.assert_called_once_with("test:test_key")


@pytest.mark.asyncio
async def test_exists_true():
    """Test exists returns True for existing entry."""
    mock_client = AsyncMock()

    entry = CacheEntry(
        value={"result": "test"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=3600,
    )

    entry_dict = entry.model_dump()
    json_bytes = json.dumps(entry_dict).encode("utf-8")
    mock_client.get.return_value = b"J" + json_bytes

    cache = RedisCache(redis=mock_client)
    result = await cache.exists("test_key")

    assert result is True


@pytest.mark.asyncio
async def test_exists_false():
    """Test exists returns False for non-existent entry."""
    mock_client = AsyncMock()
    mock_client.get.return_value = None

    cache = RedisCache(redis=mock_client)
    result = await cache.exists("missing_key")

    assert result is False


@pytest.mark.asyncio
async def test_compression_large_value():
    """Test that large values are compressed."""
    mock_client = AsyncMock()
    mock_client.setex = AsyncMock()

    # Create large value that exceeds compression threshold (1024 bytes)
    large_value = {"data": "x" * 2000}
    entry = CacheEntry(
        value=large_value,
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=300,
    )

    cache = RedisCache(
        redis=mock_client,
        compression_threshold=1024,
    )
    await cache.set("large_key", entry)

    # Check that compressed data was stored
    call_args = mock_client.setex.call_args
    stored_data = call_args[0][2]
    assert stored_data.startswith(b"Z")  # Compression marker

    # Verify we can deserialize it
    deserialized = cache._deserialize(stored_data)
    assert deserialized["value"]["data"] == "x" * 2000


@pytest.mark.asyncio
async def test_no_compression_small_value():
    """Test that small values are not compressed."""
    mock_client = AsyncMock()
    mock_client.setex = AsyncMock()

    small_value = {"data": "small"}
    entry = CacheEntry(
        value=small_value,
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=300,
    )

    cache = RedisCache(
        redis=mock_client,
        compression_threshold=1024,
    )
    await cache.set("small_key", entry)

    # Check that uncompressed JSON was stored
    call_args = mock_client.setex.call_args
    stored_data = call_args[0][2]
    assert stored_data.startswith(b"J")  # JSON marker, not compressed


@pytest.mark.asyncio
async def test_deserialize_compressed():
    """Test deserializing compressed data."""
    cache = RedisCache(redis=AsyncMock())

    original_data = {"key": "value" * 100}
    json_bytes = json.dumps(original_data).encode("utf-8")
    compressed = zlib.compress(json_bytes, level=6)
    serialized = b"Z" + compressed

    result = cache._deserialize(serialized)
    assert result == original_data


@pytest.mark.asyncio
async def test_deserialize_uncompressed():
    """Test deserializing uncompressed data."""
    cache = RedisCache(redis=AsyncMock())

    original_data = {"key": "value"}
    json_bytes = json.dumps(original_data).encode("utf-8")
    serialized = b"J" + json_bytes

    result = cache._deserialize(serialized)
    assert result == original_data


@pytest.mark.asyncio
async def test_acquire_lock_success():
    """Test successfully acquiring a lock."""
    mock_client = AsyncMock()
    mock_client.set.return_value = True  # Lock acquired

    cache = RedisCache(redis=mock_client, key_prefix="test", lock_ttl_ms=5000)
    result = await cache.acquire_lock("test_key")

    assert result is True
    mock_client.set.assert_called_once()
    call_args = mock_client.set.call_args
    assert call_args[0][0] == "test:lock:test_key"
    assert call_args[1]["px"] == 5000  # TTL in milliseconds
    assert call_args[1]["nx"] is True  # SET NX (only if not exists)


@pytest.mark.asyncio
async def test_acquire_lock_failure():
    """Test failing to acquire a lock (already held)."""
    mock_client = AsyncMock()
    mock_client.set.return_value = None  # Lock already held

    cache = RedisCache(redis=mock_client)
    result = await cache.acquire_lock("test_key")

    assert result is False


@pytest.mark.asyncio
async def test_release_lock():
    """Test releasing a lock."""
    mock_client = AsyncMock()
    mock_client.delete = AsyncMock()

    cache = RedisCache(redis=mock_client, key_prefix="test")
    await cache.release_lock("test_key")

    mock_client.delete.assert_called_once_with("test:lock:test_key")


@pytest.mark.asyncio
async def test_wait_for_inflight_success():
    """Test waiting for in-flight computation that completes."""
    mock_client = AsyncMock()

    # First call returns None, second call returns entry
    entry = CacheEntry(
        value={"result": "computed"},
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=time.time(),
        ttl_seconds=300,
    )

    entry_dict = entry.model_dump()
    json_bytes = json.dumps(entry_dict).encode("utf-8")

    mock_client.get.side_effect = [
        None,  # First check: not ready
        b"J" + json_bytes,  # Second check: ready
    ]

    cache = RedisCache(redis=mock_client, max_lock_wait=1.0)
    result = await cache.wait_for_inflight("test_key")

    assert result is not None
    assert result.value == {"result": "computed"}


@pytest.mark.asyncio
async def test_wait_for_inflight_timeout():
    """Test waiting for in-flight computation that times out."""
    mock_client = AsyncMock()
    mock_client.get.return_value = None  # Never completes

    cache = RedisCache(redis=mock_client, max_lock_wait=0.2)
    result = await cache.wait_for_inflight("test_key")

    assert result is None


@pytest.mark.asyncio
async def test_invalidate_namespace():
    """Test invalidating all keys in a namespace."""
    mock_client = AsyncMock()

    # Mock scan returning keys in batches
    mock_client.scan.side_effect = [
        (10, [b"pf:ns:test:key1", b"pf:ns:test:key2"]),  # First batch
        (0, [b"pf:ns:test:key3"]),  # Last batch (cursor=0)
    ]
    mock_client.delete.side_effect = [2, 1]  # Deleted counts

    cache = RedisCache(redis=mock_client, key_prefix="pf")
    deleted = await cache.invalidate_namespace("test")

    assert deleted == 3
    assert mock_client.scan.call_count == 2
    assert mock_client.delete.call_count == 2


@pytest.mark.asyncio
async def test_invalidate_namespace_no_keys():
    """Test invalidating namespace with no matching keys."""
    mock_client = AsyncMock()
    mock_client.scan.return_value = (0, [])  # No keys found

    cache = RedisCache(redis=mock_client)
    deleted = await cache.invalidate_namespace("empty")

    assert deleted == 0
    mock_client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_stop_with_flush():
    """Test stopping cache with flush=True."""
    with patch("redis.asyncio.Redis") as mock_redis_class:
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.scan.side_effect = [
            (5, [b"test:key1", b"test:key2"]),
            (0, []),  # End of scan
        ]
        mock_client.delete = AsyncMock()
        mock_client.aclose = AsyncMock()
        mock_redis_class.return_value = mock_client

        cache = RedisCache(key_prefix="test")
        await cache.start()

        # Stop with flush
        await cache.stop(flush=True)

        # Should scan and delete keys
        assert mock_client.scan.call_count == 2
        mock_client.delete.assert_called_once()
        mock_client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_stop_without_flush():
    """Test stopping cache without flush."""
    with patch("redis.asyncio.Redis") as mock_redis_class:
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.aclose = AsyncMock()
        mock_redis_class.return_value = mock_client

        cache = RedisCache()
        await cache.start()

        # Stop without flush
        await cache.stop(flush=False)

        # Should NOT scan/delete keys
        mock_client.scan.assert_not_called()
        mock_client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_start_connection_failure():
    """Test start() failing to connect to Redis."""
    with patch("redis.asyncio.Redis") as mock_redis_class:
        mock_client = AsyncMock()
        mock_client.ping.side_effect = Exception("Connection refused")
        mock_client.aclose = AsyncMock()
        mock_redis_class.return_value = mock_client

        cache = RedisCache()

        with pytest.raises(ConnectionError, match="Failed to connect to Redis"):
            await cache.start(host="badhost", port=9999)

        # Should have attempted to close the client
        mock_client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_start_idempotent():
    """Test that start() is idempotent (can call multiple times)."""
    with patch("redis.asyncio.Redis") as mock_redis_class:
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_redis_class.return_value = mock_client

        cache = RedisCache()

        # Call start twice
        await cache.start()
        await cache.start()

        # Should only create client once
        assert mock_redis_class.call_count == 1


@pytest.mark.asyncio
async def test_stop_idempotent():
    """Test that stop() is idempotent."""
    with patch("redis.asyncio.Redis") as mock_redis_class:
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.aclose = AsyncMock()
        mock_redis_class.return_value = mock_client

        cache = RedisCache()
        await cache.start()

        # Call stop twice
        await cache.stop()
        await cache.stop()

        # Should only close once
        assert mock_client.aclose.call_count == 1


@pytest.mark.asyncio
async def test_ensure_connected_raises():
    """Test that operations fail if not connected."""
    cache = RedisCache()

    with pytest.raises(RuntimeError, match="Redis client not initialized"):
        await cache.get("key")

    with pytest.raises(RuntimeError, match="Redis client not initialized"):
        await cache.set(
            "key",
            CacheEntry(
                value={},
                content_type=CacheContentType.LLM_COMPLETION,
                created_at=time.time(),
                ttl_seconds=60,
            ),
        )

    with pytest.raises(RuntimeError, match="Redis client not initialized"):
        await cache.delete("key")

    with pytest.raises(RuntimeError, match="Redis client not initialized"):
        await cache.invalidate_namespace("ns")

    with pytest.raises(RuntimeError, match="Redis client not initialized"):
        await cache.acquire_lock("key")

    with pytest.raises(RuntimeError, match="Redis client not initialized"):
        await cache.release_lock("key")


@pytest.mark.asyncio
async def test_custom_key_prefix():
    """Test using custom key prefix."""
    mock_client = AsyncMock()
    mock_client.get.return_value = None

    cache = RedisCache(redis=mock_client, key_prefix="myapp")
    await cache.get("test_key")

    mock_client.get.assert_called_once_with("myapp:test_key")


@pytest.mark.asyncio
async def test_start_with_password():
    """Test starting cache with Redis password."""
    with patch("redis.asyncio.Redis") as mock_redis_class:
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_redis_class.return_value = mock_client

        cache = RedisCache()
        await cache.start(host="localhost", password="secret123")

        # Verify password was passed
        call_kwargs = mock_redis_class.call_args[1]
        assert call_kwargs["password"] == "secret123"


@pytest.mark.asyncio
async def test_start_with_custom_db():
    """Test starting cache with custom database number."""
    with patch("redis.asyncio.Redis") as mock_redis_class:
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_redis_class.return_value = mock_client

        cache = RedisCache()
        await cache.start(db=5)

        # Verify db number was passed
        call_kwargs = mock_redis_class.call_args[1]
        assert call_kwargs["db"] == 5


@pytest.mark.asyncio
async def test_start_with_custom_timeouts():
    """Test starting cache with custom timeout values."""
    with patch("redis.asyncio.Redis") as mock_redis_class:
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_redis_class.return_value = mock_client

        cache = RedisCache()
        await cache.start(
            socket_timeout=10.0,
            socket_connect_timeout=15.0,
        )

        # Verify timeouts were passed
        call_kwargs = mock_redis_class.call_args[1]
        assert call_kwargs["socket_timeout"] == 10.0
        assert call_kwargs["socket_connect_timeout"] == 15.0


@pytest.mark.asyncio
async def test_start_with_extra_kwargs():
    """Test starting cache with additional Redis kwargs."""
    with patch("redis.asyncio.Redis") as mock_redis_class:
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_redis_class.return_value = mock_client

        cache = RedisCache()
        await cache.start(
            max_connections=50,
            decode_responses=False,
        )

        # Verify extra kwargs were passed
        call_kwargs = mock_redis_class.call_args[1]
        assert call_kwargs["max_connections"] == 50
        assert call_kwargs["decode_responses"] is False
