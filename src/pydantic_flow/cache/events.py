"""Cache-related streaming events.

This module defines events related to cache operations,
including hits, misses, writes, and errors.
"""

from __future__ import annotations

from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.base import ProgressType


class CacheHit(ProgressItem):
    """Cache hit occurred during node execution.

    Attributes:
        node_id: Node that had the cache hit.
        key: Cache key that was hit.
        backend: Name of cache backend used.
        ttl_remaining: Remaining TTL in seconds, or None if no TTL.

    """

    type: ProgressType = ProgressType.CACHE_HIT
    key: str
    backend: str
    ttl_remaining: float | None = None


class CacheMiss(ProgressItem):
    """Cache miss occurred during node execution.

    Attributes:
        node_id: Node that had the cache miss.
        key: Cache key that missed.
        backend: Name of cache backend used.

    """

    type: ProgressType = ProgressType.CACHE_MISS
    key: str
    backend: str


class CacheWrite(ProgressItem):
    """Cache write completed.

    Attributes:
        node_id: Node that wrote to cache.
        key: Cache key written.
        backend: Name of cache backend used.
        value_size_bytes: Size of cached value in bytes.

    """

    type: ProgressType = ProgressType.CACHE_WRITE
    key: str
    backend: str
    value_size_bytes: int


class CacheError(ProgressItem):
    """Cache operation error occurred.

    Attributes:
        node_id: Node where error occurred.
        error: Error message.
        operation: Operation that failed (get/set/delete).
        key: Cache key involved, if applicable.

    """

    type: ProgressType = ProgressType.CACHE_ERROR
    error: str
    operation: str
    key: str | None = None
