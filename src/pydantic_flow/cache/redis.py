"""Redis cache backend with singleflight locks and compression.

This module provides a distributed cache backend using Redis, suitable
for multi-process and multi-server deployments.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING
from typing import Any
import zlib

from pydantic_flow.cache.base import CacheBackend
from pydantic_flow.cache.base import CacheEntry

if TYPE_CHECKING:
    from redis.asyncio import Redis


class RedisCache(CacheBackend):
    """Redis-based cache with singleflight and optional compression.

    Supports two usage modes:
    1. Self-managed: Pass existing Redis client, manage lifecycle yourself
    2. Cache-managed: Cache creates and manages Redis client automatically

    Attributes:
        redis: Redis client instance.
        key_prefix: Prefix for all cache keys.
        lock_ttl_ms: TTL for singleflight locks in milliseconds.
        compression_threshold: Compress values larger than this size.
        max_lock_wait: Maximum seconds to wait for a singleflight lock.

    Example:
        Self-managed connection::

            redis_client = Redis(host="localhost", port=6379)
            cache = RedisCache(redis=redis_client)
            result = await cache.get("key")
            await redis_client.aclose()

        Cache-managed connection::

            cache = RedisCache()
            await cache.start(host="localhost", port=6379)
            result = await cache.get("key")
            await cache.stop()

    """

    def __init__(
        self,
        redis: Redis[bytes] | None = None,  # type: ignore[type-arg]
        key_prefix: str = "pf",
        lock_ttl_ms: int = 5000,
        compression_threshold: int = 1024,
        max_lock_wait: float = 30.0,
    ) -> None:
        """Initialize Redis cache.

        Args:
            redis: Optional Redis async client. If None, client will be created
                in start() using connection parameters.
            key_prefix: Prefix for cache keys.
            lock_ttl_ms: Lock TTL in milliseconds.
            compression_threshold: Compress values larger than this.
            max_lock_wait: Max seconds to wait for lock.

        """
        self._redis = redis
        self._owns_connection = redis is None
        self._key_prefix = key_prefix
        self._lock_ttl_ms = lock_ttl_ms
        self._compression_threshold = compression_threshold
        self._max_lock_wait = max_lock_wait
        self._inflight: dict[str, asyncio.Future[CacheEntry | None]] = {}
        self._lock = asyncio.Lock()

    async def start(  # noqa: PLR0913
        self,
        *,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        **redis_kwargs: Any,
    ) -> None:
        """Start the Redis cache and create client if needed.

        Only creates a Redis client if one wasn't provided in __init__.
        If a client was provided, this is a no-op (client is assumed ready).

        Args:
            host: Redis server host (only used if creating connection).
            port: Redis server port (only used if creating connection).
            db: Redis database number (only used if creating connection).
            password: Redis password (only used if creating connection).
            socket_timeout: Socket timeout in seconds.
            socket_connect_timeout: Connection timeout in seconds.
            **redis_kwargs: Additional arguments passed to Redis client.

        Raises:
            ConnectionError: If unable to connect to Redis server.

        """
        if not self._owns_connection:
            # Client provided by user, assume it's ready
            return

        if self._redis is not None:
            # Already started
            return

        from redis.asyncio import Redis  # noqa: PLC0415

        self._redis = Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            **redis_kwargs,
        )

        # Verify connection
        try:
            await self._redis.ping()  # type: ignore[misc]
        except Exception as e:
            await self._redis.aclose()
            self._redis = None
            msg = f"Failed to connect to Redis at {host}:{port}: {e}"
            raise ConnectionError(msg) from e

    async def stop(self, *, flush: bool = False) -> None:
        """Stop the Redis cache and cleanup resources.

        Only closes the Redis client if it was created by the cache.
        If a client was provided by the user, this is a no-op.

        Args:
            flush: Whether to flush all keys with the cache prefix before
                closing. Only applies if cache owns the connection.

        """
        if not self._owns_connection:
            # User owns the connection, don't close it
            return

        if self._redis is None:
            # Already stopped
            return

        if flush:
            # Delete all keys with our prefix
            pattern = f"{self._key_prefix}:*"
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._redis.delete(*keys)
                if cursor == 0:
                    break

        await self._redis.aclose()
        self._redis = None

    def _ensure_connected(self) -> None:
        """Verify Redis client is available.

        Raises:
            RuntimeError: If Redis client is not initialized.

        """
        if self._redis is None:
            msg = (
                "Redis client not initialized. Call start() or provide "
                "a redis client in __init__."
            )
            raise RuntimeError(msg)

    def _make_key(self, key: str) -> str:
        """Construct full Redis key.

        Args:
            key: Cache key.

        Returns:
            Full Redis key with prefix.

        """
        return f"{self._key_prefix}:{key}"

    def _make_lock_key(self, key: str) -> str:
        """Construct lock key.

        Args:
            key: Cache key.

        Returns:
            Lock key for singleflight.

        """
        return f"{self._key_prefix}:lock:{key}"

    async def get(self, key: str) -> CacheEntry | None:
        """Retrieve entry by key.

        Args:
            key: Cache key.

        Returns:
            CacheEntry if found and not expired, None otherwise.

        Raises:
            RuntimeError: If Redis client not initialized.

        """
        self._ensure_connected()
        redis_key = self._make_key(key)
        data = await self._redis.get(redis_key)  # type: ignore[union-attr]

        if data is None:
            return None

        entry_dict = self._deserialize(data)
        entry = CacheEntry(**entry_dict)

        current_time = time.time()
        if entry.is_expired(current_time):
            await self._redis.delete(redis_key)  # type: ignore[union-attr]
            return None

        return entry

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Store entry.

        Args:
            key: Cache key.
            entry: Entry to store.

        Raises:
            RuntimeError: If Redis client not initialized.

        """
        self._ensure_connected()
        redis_key = self._make_key(key)
        data = self._serialize(entry.model_dump())

        if entry.ttl_seconds:
            await self._redis.setex(redis_key, entry.ttl_seconds, data)  # type: ignore[union-attr]
        else:
            await self._redis.set(redis_key, data)  # type: ignore[union-attr]

    async def delete(self, key: str) -> None:
        """Delete entry by key.

        Args:
            key: Cache key.

        Raises:
            RuntimeError: If Redis client not initialized.

        """
        self._ensure_connected()
        redis_key = self._make_key(key)
        await self._redis.delete(redis_key)  # type: ignore[union-attr]

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Cache key.

        Returns:
            True if exists and not expired.

        """
        entry = await self.get(key)
        return entry is not None

    async def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all keys in a namespace.

        Args:
            namespace: Namespace to invalidate.

        Returns:
            Number of keys invalidated.

        Raises:
            RuntimeError: If Redis client not initialized.

        """
        self._ensure_connected()
        pattern = f"{self._key_prefix}:ns:{namespace}:*"
        cursor = 0
        deleted = 0

        while True:
            cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)  # type: ignore[union-attr]
            if keys:
                deleted += await self._redis.delete(*keys)  # type: ignore[union-attr]
            if cursor == 0:
                break

        return deleted

    async def acquire_lock(self, key: str) -> bool:
        """Acquire a singleflight lock.

        Args:
            key: Cache key to lock.

        Returns:
            True if lock acquired, False if already held.

        Raises:
            RuntimeError: If Redis client not initialized.

        """
        self._ensure_connected()
        lock_key = self._make_lock_key(key)
        result = await self._redis.set(  # type: ignore[union-attr]
            lock_key,
            b"1",
            px=self._lock_ttl_ms,
            nx=True,
        )
        return bool(result)

    async def release_lock(self, key: str) -> None:
        """Release a singleflight lock.

        Args:
            key: Cache key to unlock.

        Raises:
            RuntimeError: If Redis client not initialized.

        """
        self._ensure_connected()
        lock_key = self._make_lock_key(key)
        await self._redis.delete(lock_key)  # type: ignore[union-attr]

    async def wait_for_inflight(self, key: str) -> CacheEntry | None:
        """Wait for an in-flight computation to complete.

        Args:
            key: Cache key.

        Returns:
            Result from in-flight computation, or None if timeout.

        """
        start_time = time.time()
        while (time.time() - start_time) < self._max_lock_wait:
            entry = await self.get(key)
            if entry is not None:
                return entry

            await asyncio.sleep(0.1)

        return None

    def _serialize(self, data: dict[str, Any]) -> bytes:
        """Serialize data to bytes with optional compression.

        Args:
            data: Dictionary to serialize.

        Returns:
            Serialized (possibly compressed) bytes.

        """
        json_bytes = json.dumps(data).encode("utf-8")

        if len(json_bytes) > self._compression_threshold:
            compressed = zlib.compress(json_bytes, level=6)
            return b"Z" + compressed

        return b"J" + json_bytes

    def _deserialize(self, data: bytes) -> dict[str, Any]:
        """Deserialize bytes to dictionary.

        Args:
            data: Serialized bytes.

        Returns:
            Deserialized dictionary.

        """
        marker = data[0:1]
        payload = data[1:]

        json_bytes = zlib.decompress(payload) if marker == b"Z" else payload
        return json.loads(json_bytes)
