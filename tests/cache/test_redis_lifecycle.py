"""Tests for RedisCache backend with dual-mode lifecycle support.

These tests verify both self-managed and cache-managed connection patterns
using mocked Redis clients to avoid requiring a real Redis server.
"""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from pydantic_flow.cache.base import CacheContentType
from pydantic_flow.cache.base import CacheEntry
from pydantic_flow.cache.redis import RedisCache


@pytest.mark.asyncio
async def test_cache_managed_lifecycle():
    """Test cache-managed connection pattern."""
    with patch("redis.asyncio.Redis") as mock_redis_class:
        # Mock Redis client instance
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.aclose = AsyncMock()
        mock_redis_class.return_value = mock_client

        cache = RedisCache(key_prefix="test_managed")

        # Should be able to start
        await cache.start(host="localhost", port=6379, db=15)

        # Verify Redis client was created with correct params
        mock_redis_class.assert_called_once_with(
            host="localhost",
            port=6379,
            db=15,
            password=None,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
        mock_client.ping.assert_called_once()

        # Mock get/set operations
        mock_client.get.return_value = None
        mock_client.setex = AsyncMock()

        # Should be able to use cache
        entry = CacheEntry(
            value="test_value",
            content_type=CacheContentType.LLM_COMPLETION,
            created_at=0.0,
            ttl_seconds=60,
        )
        await cache.set("test_key", entry)

        mock_client.setex.assert_called_once()

        # Should be able to stop
        await cache.stop()
        mock_client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_cache_managed_requires_start():
    """Test that operations fail if start() not called."""
    cache = RedisCache()

    # Should raise RuntimeError if not started
    with pytest.raises(RuntimeError, match="Redis client not initialized"):
        await cache.get("test_key")


@pytest.mark.asyncio
async def test_self_managed_lifecycle():
    """Test self-managed connection pattern."""
    # Create mock Redis client
    mock_client = AsyncMock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = None
    mock_client.setex = AsyncMock()
    mock_client.aclose = AsyncMock()

    # Pass client to cache
    cache = RedisCache(redis=mock_client, key_prefix="test_self")

    # start() should be a no-op for self-managed
    await cache.start()
    mock_client.ping.assert_not_called()  # start() is no-op

    # Should be able to use cache immediately
    entry = CacheEntry(
        value="test_value",
        content_type=CacheContentType.LLM_COMPLETION,
        created_at=0.0,
        ttl_seconds=60,
    )
    await cache.set("test_key", entry)
    mock_client.setex.assert_called_once()

    # stop() should be a no-op for self-managed
    await cache.stop()
    mock_client.aclose.assert_not_called()  # stop() is no-op, user owns connection
