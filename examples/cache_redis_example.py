"""Redis cache backend example.

This example demonstrates using RedisCache for distributed caching across
multiple processes or servers. Shows both self-managed and cache-managed
connection patterns.

Note: Requires a running Redis instance (default: localhost:6379).
"""

import asyncio
from datetime import timedelta
import time

from redis.asyncio import Redis

from pydantic_flow.cache import CacheContentType
from pydantic_flow.cache import CacheEntry
from pydantic_flow.cache import RedisCache


# Helper to extract result from stream
async def extract_result_from_stream(stream):
    """Extract final result from async stream of progress items."""
    result = None
    async for item in stream:
        if hasattr(item, "result"):
            result = item.result
    return result


async def example_self_managed() -> None:
    """Pattern 1: Self-Managed Connection.

    User creates and manages the Redis client lifecycle.
    Gives full control over connection parameters and reuse.
    """
    print("=== Pattern 1: Self-Managed Connection ===\n")

    # Create Redis client
    redis_client = Redis(
        host="localhost",
        port=6379,
        decode_responses=False,  # Keep as bytes for compression
        socket_connect_timeout=5,
    )

    # Test Redis connection
    try:
        # Try a simple operation to verify connection
        await redis_client.set("pf:test", b"ok", ex=1)
        await redis_client.delete("pf:test")
        print("✓ Connected to Redis")
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        print("  Make sure Redis is running: redis-server")
        await redis_client.aclose()
        return

    # Create Redis cache backend with existing client
    cache = RedisCache(
        redis=redis_client,  # Pass existing client
        key_prefix="pf_example_self",
        compression_threshold=1024,  # Compress values > 1KB
        lock_ttl_ms=5000,  # Distributed lock TTL
    )

    try:
        print("RedisCache initialized (self-managed mode)")
        print("No start() needed - client already connected\n")

        # Store some LLM responses
        llm_entry = CacheEntry(
            value={"text": "The capital of France is Paris."},
            content_type=CacheContentType.LLM_COMPLETION,
            created_at=time.time(),
            ttl_seconds=int(timedelta(hours=24).total_seconds()),
        )
        await cache.set("llm:capital:france", llm_entry)
        print("✓ Stored LLM response in Redis")

        # Retrieve from cache
        retrieved = await cache.get("llm:capital:france")
        if retrieved:
            print(f"✓ Retrieved from cache: {retrieved.value}")

        # Delete specific keys
        await cache.delete("llm:capital:france")
        print("✓ Deleted cache entry")

        print("\n✓ Self-managed pattern complete")
        print("  User has full control over Redis client")
        print("  Client can be reused across multiple caches\n")

    finally:
        # User is responsible for closing the connection
        await redis_client.aclose()
        print("✓ Redis connection closed (by user)\n")


async def example_cache_managed() -> None:
    """Pattern 2: Cache-Managed Connection.

    Cache creates and manages the Redis client lifecycle automatically.
    Simpler API, good for single-cache scenarios.
    """
    print("=== Pattern 2: Cache-Managed Connection ===\n")

    # Create cache without a Redis client
    cache = RedisCache(
        key_prefix="pf_example_managed",
        compression_threshold=1024,
        lock_ttl_ms=5000,
    )

    try:
        # Cache creates and connects to Redis in start()
        print("Calling cache.start() to connect to Redis...")
        await cache.start(
            host="localhost",
            port=6379,
            socket_timeout=5.0,
        )
        print("✓ Cache created and connected to Redis\n")

        # Store a large embedding (will be compressed)
        large_vector = [0.1] * 2000  # 2000 floats > 1KB
        embedding_entry = CacheEntry(
            value={"vector": large_vector, "dimension": len(large_vector)},
            content_type=CacheContentType.EMBEDDING_VECTOR,
            created_at=time.time(),
            ttl_seconds=int(timedelta(days=7).total_seconds()),
        )
        await cache.set("embed:large_doc", embedding_entry)
        print("✓ Stored large embedding (compressed)")

        # Check existence
        exists = await cache.exists("embed:large_doc")
        print(f"✓ Embedding exists: {exists}")

        # Store entries with namespace pattern
        llm_entry = CacheEntry(
            value={"text": "Sample response"},
            content_type=CacheContentType.LLM_COMPLETION,
            created_at=time.time(),
            ttl_seconds=int(timedelta(hours=1).total_seconds()),
        )
        await cache.set("ns:test:key1", llm_entry)
        await cache.set("ns:test:key2", llm_entry)
        print("✓ Stored entries with 'test' namespace")

        # Namespace invalidation
        deleted = await cache.invalidate_namespace("test")
        print(f"✓ Invalidated 'test' namespace: {deleted} entries deleted")

        print("\n✓ Cache-managed pattern complete")
        print("  Cache handles Redis client lifecycle")
        print("  Simpler API for single-cache scenarios\n")

    except ConnectionError as e:
        print(f"✗ Failed to connect: {e}")
        print("  Make sure Redis is running: redis-server")
        return

    finally:
        # Cache closes its own connection
        print("Calling cache.stop() to cleanup...")
        await cache.stop(flush=False)  # Set flush=True to clear all keys
        print("✓ Cache closed connection (automatic)\n")


async def main() -> None:
    """Run both example patterns."""
    print("\n" + "=" * 60)
    print("Redis Cache Backend - Connection Patterns")
    print("=" * 60 + "\n")

    # Pattern 1: Self-managed connection
    await example_self_managed()

    # Pattern 2: Cache-managed connection
    await example_cache_managed()

    print("=" * 60)
    print("Summary:")
    print("  • Self-managed: Full control, reusable client")
    print("  • Cache-managed: Simpler API, automatic lifecycle")
    print("  • Both support distributed caching features")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
