"""Stampede protection utilities for cache backends.

This module provides singleflight functionality to prevent thundering herd
problems when multiple concurrent requests miss the cache for the same key.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any


class Singleflight:
    """Singleflight coordinator for deduplicating concurrent operations.

    When multiple callers request the same key simultaneously, only one
    operation executes while others wait for the result.
    """

    def __init__(self) -> None:
        """Initialize singleflight coordinator."""
        self._inflight: dict[str, asyncio.Future[Any]] = {}
        self._lock = asyncio.Lock()

    async def do[T](
        self,
        key: str,
        fn: Callable[[], Awaitable[T]],
    ) -> T:
        """Execute function with singleflight protection.

        Args:
            key: Unique key for this operation.
            fn: Async function to execute if not already in-flight.

        Returns:
            Result of fn().

        """
        async with self._lock:
            if key in self._inflight:
                future = self._inflight[key]
                return await future

            future: asyncio.Future[T] = asyncio.Future()
            self._inflight[key] = future

        try:
            result = await fn()
            future.set_result(result)
            return result
        except Exception as error:
            future.set_exception(error)
            raise
        finally:
            async with self._lock:
                self._inflight.pop(key, None)
