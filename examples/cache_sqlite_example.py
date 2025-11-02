"""SQLite cache backend example.

This example demonstrates using SQLiteCache for persistent local caching.
"""

import asyncio
from datetime import timedelta
import time

from pydantic_flow.cache import CacheContentType
from pydantic_flow.cache import CacheEntry
from pydantic_flow.cache import SQLiteCache


# Helper to extract result from stream
async def extract_result_from_stream(stream):
    """Extract final result from async stream of progress items."""
    result = None
    async for item in stream:
        if hasattr(item, "result"):
            result = item.result
    return result


async def main() -> None:
    """Demonstrate SQLite cache with persistence."""
    # Create SQLite cache with local file
    async with SQLiteCache(
        db_path=".example-cache.db",
        cleanup_interval=60.0,  # Clean up every minute
    ) as cache:
        print("SQLiteCache initialized with local database")

        # Store some LLM responses
        llm_entry = CacheEntry(
            value={"text": "The capital of France is Paris."},
            content_type=CacheContentType.LLM_COMPLETION,
            created_at=time.time(),
            ttl_seconds=int(timedelta(hours=24).total_seconds()),
        )
        await cache.set("llm:capital:france", llm_entry, namespace="geography")
        print("✓ Stored LLM response in 'geography' namespace")

        # Store embedding
        embedding_entry = CacheEntry(
            value={"vector": [0.1, 0.2, 0.3, 0.4], "dimension": 4},
            content_type=CacheContentType.EMBEDDING_VECTOR,
            created_at=time.time(),
            ttl_seconds=int(timedelta(days=7).total_seconds()),
        )
        await cache.set("embed:doc123", embedding_entry, namespace="vectors")
        print("✓ Stored embedding in 'vectors' namespace")

        # Retrieve from cache
        retrieved = await cache.get("llm:capital:france")
        if retrieved:
            print(f"✓ Retrieved from cache: {retrieved.value}")

        # Check existence
        exists = await cache.exists("embed:doc123")
        print(f"✓ Embedding exists: {exists}")

        # Add more geography entries
        await cache.set("geo:country:usa", llm_entry, namespace="geography")
        await cache.set("geo:country:uk", llm_entry, namespace="geography")

        # Namespace invalidation
        deleted = await cache.invalidate_namespace("geography")
        print(f"✓ Invalidated 'geography' namespace: {deleted} entries deleted")

        # Verify vectors namespace still intact
        vectors_exist = await cache.exists("embed:doc123")
        print(f"✓ Vectors namespace unaffected: {vectors_exist}")

        # Verify persistence works
        print("\n--- Cache persists across sessions ---")
        print("Cache data is stored in .example-cache.db")
        print("Re-opening the cache will restore all entries.")


if __name__ == "__main__":
    asyncio.run(main())
