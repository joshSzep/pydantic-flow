"""In-memory cache backend with TTL and LRU eviction.

This module provides a simple but efficient in-memory cache suitable
for single-process applications.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import contextlib
import time

from pydantic_flow.cache.base import CacheBackend
from pydantic_flow.cache.base import CacheEntry


class InMemoryCache(CacheBackend):
    """In-memory cache with TTL and LRU eviction.

    This backend maintains an LRU-ordered dictionary with automatic
    cleanup of expired entries. Thread-safe for async usage.

    Attributes:
        max_entries: Maximum number of entries before LRU eviction.
        cleanup_interval: Seconds between background TTL cleanup passes.

    """

    def __init__(
        self,
        max_entries: int = 10000,
        cleanup_interval: float = 60.0,
    ) -> None:
        """Initialize the in-memory cache.

        Args:
            max_entries: Maximum cache size before LRU eviction.
            cleanup_interval: Seconds between cleanup passes.

        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._cleanup_interval = cleanup_interval
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start background cleanup task."""
        if self._running:
            return
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop background cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

    async def get(self, key: str) -> CacheEntry | None:
        """Retrieve entry by key.

        Args:
            key: Cache key.

        Returns:
            CacheEntry if found and not expired, None otherwise.

        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            current_time = time.time()
            if entry.is_expired(current_time):
                del self._cache[key]
                return None

            self._cache.move_to_end(key)
            return entry

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Store entry.

        Args:
            key: Cache key.
            entry: Entry to store.

        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]

            self._cache[key] = entry

            if len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)

    async def delete(self, key: str) -> None:
        """Delete entry by key.

        Args:
            key: Cache key.

        """
        async with self._lock:
            self._cache.pop(key, None)

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

        """
        prefix = f"pf:ns:{namespace}:"
        async with self._lock:
            keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()

    async def size(self) -> int:
        """Get current cache size.

        Returns:
            Number of entries in cache.

        """
        async with self._lock:
            return len(self._cache)

    async def _cleanup_loop(self) -> None:
        """Background task to remove expired entries."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def _cleanup_expired(self) -> None:
        """Remove all expired entries."""
        current_time = time.time()
        async with self._lock:
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if entry.is_expired(current_time)
            ]
            for key in expired_keys:
                del self._cache[key]
