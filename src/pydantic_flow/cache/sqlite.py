"""SQLite cache backend with persistence and async operations.

This module provides a persistent local cache using SQLite, suitable
for single-server applications that need durability without external
dependencies.

Note: Requires aiosqlite package to be installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
import time
from typing import Any

import aiosqlite

from pydantic_flow.cache.base import CacheBackend
from pydantic_flow.cache.base import CacheContentType
from pydantic_flow.cache.base import CacheEntry


class SQLiteCache(CacheBackend):
    """SQLite-backed persistent cache with async operations.

    This backend provides local persistence using SQLite with WAL mode
    for better concurrency. Includes background cleanup of expired entries.

    Attributes:
        db_path: Path to SQLite database file.
        cleanup_interval: Seconds between background TTL cleanup passes.

    """

    def __init__(
        self,
        db_path: str | Path = ".pydantic-flow-cache.db",
        cleanup_interval: float = 300.0,
    ) -> None:
        """Initialize SQLite cache.

        Args:
            db_path: Path to SQLite database file.
            cleanup_interval: Seconds between cleanup passes.

        """
        self._db_path = Path(db_path)
        self._cleanup_interval = cleanup_interval
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start cache and initialize database."""
        if self._running:
            return

        self._running = True
        self._connection = await aiosqlite.connect(str(self._db_path))

        # Enable WAL mode for better concurrency
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA synchronous=NORMAL")

        # Create schema
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                content_type TEXT NOT NULL,
                created_at REAL NOT NULL,
                ttl_seconds REAL,
                namespace TEXT NOT NULL
            )
            """
        )

        # Create indexes for efficient queries
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_namespace ON cache_entries(namespace)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)"
        )

        await self._connection.commit()

        # Start background cleanup
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop cache and close database connection."""
        if not self._running:
            return

        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

        if self._connection:
            await self._connection.close()
            self._connection = None

    async def get(self, key: str) -> CacheEntry | None:
        """Retrieve entry from cache.

        Args:
            key: Cache key.

        Returns:
            Cache entry if found and not expired, None otherwise.

        """
        if not self._connection:
            return None

        async with self._lock:
            cursor = await self._connection.execute(
                """
                SELECT value, content_type, created_at, ttl_seconds
                FROM cache_entries
                WHERE key = ?
                """,
                (key,),
            )
            row = await cursor.fetchone()
            await cursor.close()

        if not row:
            return None

        value_json, content_type_str, created_at, ttl_seconds = row
        entry = CacheEntry(
            value=json.loads(value_json),
            content_type=CacheContentType(content_type_str),
            created_at=created_at,
            ttl_seconds=ttl_seconds,
        )

        now = time.time()
        if entry.is_expired(now):
            # Delete expired entry
            await self.delete(key)
            return None

        return entry

    async def set(
        self,
        key: str,
        entry: CacheEntry,
        namespace: str = "default",
    ) -> None:
        """Store entry in cache.

        Args:
            key: Cache key.
            entry: Cache entry to store.
            namespace: Namespace for grouping entries.

        """
        if not self._connection:
            return

        value_json = json.dumps(entry.value)

        async with self._lock:
            await self._connection.execute(
                """
                INSERT OR REPLACE INTO cache_entries
                (key, value, content_type, created_at, ttl_seconds, namespace)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    value_json,
                    entry.content_type.value,
                    entry.created_at,
                    entry.ttl_seconds,
                    namespace,
                ),
            )
            await self._connection.commit()

    async def delete(self, key: str) -> None:
        """Delete entry from cache.

        Args:
            key: Cache key to delete.

        """
        if not self._connection:
            return

        async with self._lock:
            await self._connection.execute(
                "DELETE FROM cache_entries WHERE key = ?",
                (key,),
            )
            await self._connection.commit()

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key to check.

        Returns:
            True if key exists and is not expired.

        """
        entry = await self.get(key)
        return entry is not None

    async def clear(self) -> None:
        """Remove all entries from cache."""
        if not self._connection:
            return

        async with self._lock:
            await self._connection.execute("DELETE FROM cache_entries")
            await self._connection.commit()

    async def invalidate_namespace(self, namespace: str) -> int:
        """Remove all entries in a namespace.

        Args:
            namespace: Namespace to invalidate.

        Returns:
            Number of entries deleted.

        """
        if not self._connection:
            return 0

        async with self._lock:
            cursor = await self._connection.execute(
                "DELETE FROM cache_entries WHERE namespace = ?",
                (namespace,),
            )
            await self._connection.commit()
            count = cursor.rowcount
            await cursor.close()

        return count

    async def _cleanup_expired(self) -> int:
        """Remove expired entries from database.

        Returns:
            Number of entries deleted.

        """
        if not self._connection:
            return 0

        now = time.time()

        async with self._lock:
            cursor = await self._connection.execute(
                """
                DELETE FROM cache_entries
                WHERE ttl_seconds IS NOT NULL
                AND (created_at + ttl_seconds) < ?
                """,
                (now,),
            )
            await self._connection.commit()
            count = cursor.rowcount
            await cursor.close()

        return count

    async def _cleanup_loop(self) -> None:
        """Background task that periodically cleans up expired entries."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                if self._running:
                    await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue on error to keep cleanup running
                continue

    async def __aenter__(self) -> SQLiteCache:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.stop()
