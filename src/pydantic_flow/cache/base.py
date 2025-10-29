"""Base cache types and abstractions.

This module defines the core interfaces, types, and policies for the caching layer.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from datetime import timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class CacheContentType(StrEnum):
    """Type of content stored in cache.

    Attributes:
        LLM_COMPLETION: Final LLM completion result (text or structured).
        EMBEDDING_VECTOR: Vector embedding output.
        STREAM_EVENTS: Captured stream event transcript for replay.

    """

    LLM_COMPLETION = "llm_completion"
    EMBEDDING_VECTOR = "embedding_vector"
    STREAM_EVENTS = "stream_events"


class CacheKeyStrategy(StrEnum):
    """Strategy for generating cache keys.

    Attributes:
        DEFAULT: Standard deterministic key including all inputs.
        MINIMAL: Minimal key with only essential parameters.
        CUSTOM: User-provided key material.

    """

    DEFAULT = "default"
    MINIMAL = "minimal"
    CUSTOM = "custom"


class CacheScope(BaseModel):
    """Scope for cache keys.

    Can be GLOBAL (shared across all flows) or NAMESPACE (isolated by string).
    """

    model_config = {"frozen": True}

    type: str = Field(default="global")
    namespace: str | None = Field(default=None)

    @classmethod
    def GLOBAL(cls) -> CacheScope:
        """Create a global scope."""
        return cls(type="global")

    @classmethod
    def NAMESPACE(cls, namespace: str) -> CacheScope:
        """Create a namespaced scope.

        Args:
            namespace: The namespace identifier.

        Returns:
            CacheScope with the specified namespace.

        """
        return cls(type="namespace", namespace=namespace)

    def prefix(self) -> str:
        """Get the prefix for cache keys.

        Returns:
            Prefix string for keys in this scope.

        """
        if self.type == "namespace" and self.namespace:
            return f"ns:{self.namespace}"
        return "global"


class CacheEntry(BaseModel):
    """A cached value with metadata.

    Attributes:
        value: The cached data (can be any type).
        content_type: Type of content stored.
        created_at: Unix timestamp when entry was created.
        ttl_seconds: Time-to-live in seconds, or None for no expiration.

    """

    model_config = {"arbitrary_types_allowed": True}

    value: Any
    content_type: CacheContentType
    created_at: float
    ttl_seconds: int | None = None

    def is_expired(self, current_time: float) -> bool:
        """Check if this entry has expired.

        Args:
            current_time: Current Unix timestamp.

        Returns:
            True if expired, False otherwise.

        """
        if self.ttl_seconds is None:
            return False
        return (current_time - self.created_at) > self.ttl_seconds

    def ttl_remaining(self, current_time: float) -> float | None:
        """Get remaining TTL in seconds.

        Args:
            current_time: Current Unix timestamp.

        Returns:
            Remaining seconds or None if no TTL set.

        """
        if self.ttl_seconds is None:
            return None
        remaining = self.ttl_seconds - (current_time - self.created_at)
        return max(0.0, remaining)


class CachePolicy(BaseModel):
    """Policy for caching node outputs.

    Attributes:
        enabled: Whether caching is enabled.
        ttl: Time-to-live for cached entries.
        key_strategy: Strategy for generating cache keys.
        scope: Scope for cache keys (global or namespaced).
        store_streams: Whether to capture and replay event streams.
        bypass: Skip cache lookup and always execute.
        extra_key_material: Additional data to include in cache key.
        node_version: Optional version string to invalidate cache on changes.

    """

    enabled: bool = True
    ttl: timedelta | None = Field(default_factory=lambda: timedelta(hours=12))
    key_strategy: CacheKeyStrategy = CacheKeyStrategy.DEFAULT
    scope: CacheScope = Field(default_factory=CacheScope.GLOBAL)
    store_streams: bool = False
    bypass: bool = False
    extra_key_material: dict[str, Any] | None = None
    node_version: str | None = None

    def ttl_seconds(self) -> int | None:
        """Get TTL in seconds.

        Returns:
            TTL in seconds or None if no TTL set.

        """
        if self.ttl is None:
            return None
        return int(self.ttl.total_seconds())


class CacheBackend(ABC):
    """Abstract base class for cache backends.

    Implementations must provide async methods for getting, setting,
    and deleting cache entries.
    """

    @abstractmethod
    async def get(self, key: str) -> CacheEntry | None:
        """Retrieve a cache entry by key.

        Args:
            key: The cache key.

        Returns:
            CacheEntry if found and not expired, None otherwise.

        """
        ...

    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> None:
        """Store a cache entry.

        Args:
            key: The cache key.
            entry: The entry to store.

        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a cache entry by key.

        Args:
            key: The cache key to delete.

        """
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: The cache key to check.

        Returns:
            True if key exists and is not expired, False otherwise.

        """
        ...

    async def start(self, *args: Any, **kwargs: Any) -> None:  # noqa: B027
        """Start the cache backend and initialize resources.

        This method is called before the cache is used. Backends can override
        to perform initialization such as:
        - Opening database connections
        - Starting background cleanup tasks
        - Initializing connection pools
        - Verifying connectivity

        Backends may accept additional args/kwargs for initialization.
        Check specific backend documentation for supported parameters.

        The default implementation is a no-op.

        Args:
            *args: Backend-specific positional arguments.
            **kwargs: Backend-specific keyword arguments.

        """
        pass

    async def stop(self, *args: Any, **kwargs: Any) -> None:  # noqa: B027
        """Stop the cache backend and cleanup resources.

        This method is called when the cache is no longer needed. Backends
        can override to perform cleanup such as:
        - Closing database connections
        - Stopping background tasks
        - Flushing pending writes
        - Releasing resources

        Backends may accept additional args/kwargs for cleanup.
        Check specific backend documentation for supported parameters.

        The default implementation is a no-op.

        Args:
            *args: Backend-specific positional arguments.
            **kwargs: Backend-specific keyword arguments.

        """
        pass

    async def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all keys in a namespace.

        Args:
            namespace: The namespace to invalidate.

        Returns:
            Number of keys invalidated.

        Raises:
            NotImplementedError: If backend doesn't support namespace operations.

        """
        msg = f"{self.__class__.__name__} does not support namespace invalidation"
        raise NotImplementedError(msg)
