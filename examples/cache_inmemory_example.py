"""InMemoryCache backend example with AgentNode.

This example demonstrates using InMemoryCache for fast, ephemeral caching
within a single process. Perfect for development and testing.

Note: This is a simplified example showing cache configuration. In production,
caching is automatically handled by the Flow when a cache_backend is configured.
"""

import asyncio
from datetime import timedelta

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_flow.cache import CachePolicy
from pydantic_flow.cache import InMemoryCache
from pydantic_flow.nodes import AgentNode


class Question(BaseModel):
    """Input model for questions."""

    text: str


async def main() -> None:
    """Demonstrate InMemoryCache configuration with AgentNode."""
    # Create in-memory cache with LRU eviction
    cache = InMemoryCache(
        max_entries=100,
        cleanup_interval=60.0,
    )
    await cache.start()

    try:
        print("=== InMemoryCache Configuration Example ===\n")

        # Create a simple agent
        agent: Agent[None, str] = Agent(
            "openai:gpt-4o-mini",
            system_prompt="You are a helpful assistant that provides concise answers.",
        )

        # Create a cacheable node with cache policy
        cached_node = AgentNode[Question, str](
            agent=agent,
            prompt_template="{text}",
            name="cached_assistant",
            cache_policy=CachePolicy(
                enabled=True,
                ttl=timedelta(hours=1),
            ),
        )

        print("✓ AgentNode configured with cache policy")
        print(f"✓ InMemoryCache created: max_entries={cache._max_entries}")
        if cached_node.cache_policy:
            print(f"✓ Cache policy: enabled={cached_node.cache_policy.enabled}")
            print(f"                TTL={cached_node.cache_policy.ttl}")
            print(
                f"                store_streams={cached_node.cache_policy.store_streams}"
            )

        print("\n--- Usage in Production ---")
        print("When integrated with a Flow:")
        print("  1. Flow is configured with cache_backend=cache")
        print("  2. Nodes with cache_policy are automatically cached")
        print("  3. Cache events (CacheHit, CacheMiss) appear in stream")
        print("  4. Identical inputs skip execution and return cached results")

        print("\n--- Cache Statistics ---")
        cache_size = len(cache._cache)
        print(f"✓ Current cache size: {cache_size} entries")
        print(f"✓ Max capacity: {cache._max_entries} entries")

        # Test cache clearing
        await cache.clear()
        print(f"✓ After clear: {len(cache._cache)} entries")

        print("\n--- InMemoryCache Benefits ---")
        print("✓ Fastest cache backend (no I/O)")
        print("✓ LRU eviction prevents memory growth")
        print("✓ TTL with background cleanup")
        print("✓ Perfect for development and testing")
        print("✓ Zero external dependencies")

        print("\n--- Example Flow Integration ---")
        print("""
from pydantic_flow import Flow

flow = Flow[Question, Answer](
    input_type=Question,
    output_type=Answer,
    cache_backend=cache,  # Enable caching
)
flow.add_nodes(cached_node)

# First call - cache miss, executes node
result1 = await flow.run(Question(text="What is 2+2?"))

# Second call - cache hit, returns cached result
result2 = await flow.run(Question(text="What is 2+2?"))
        """)

    finally:
        await cache.stop()
        print("✓ Cache stopped and cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
