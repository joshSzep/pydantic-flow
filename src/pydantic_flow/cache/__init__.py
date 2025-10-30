"""First-class caching layer for pydantic-flow.

This module provides a unified caching system for LLM responses, embeddings,
and other node outputs with pluggable backends, TTL support, and observability.
"""

from pydantic_flow.cache.base import CacheBackend
from pydantic_flow.cache.base import CacheContentType
from pydantic_flow.cache.base import CacheEntry
from pydantic_flow.cache.base import CacheKeyStrategy
from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.cache.base import CacheScope
from pydantic_flow.cache.events import CacheError
from pydantic_flow.cache.events import CacheHit
from pydantic_flow.cache.events import CacheMiss
from pydantic_flow.cache.events import CacheWrite
from pydantic_flow.cache.memory import InMemoryCache
from pydantic_flow.cache.redis import RedisCache
from pydantic_flow.cache.sqlite import SQLiteCache

__all__ = [
    "CacheBackend",
    "CacheContentType",
    "CacheEntry",
    "CacheError",
    "CacheHit",
    "CacheKeyStrategy",
    "CacheMiss",
    "CachePolicy",
    "CacheScope",
    "CacheWrite",
    "InMemoryCache",
    "RedisCache",
    "SQLiteCache",
]
