"""Cache configuration example.

This example demonstrates cache configuration with backend and policy settings.
"""

import asyncio
from datetime import timedelta

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_flow.cache import CachePolicy
from pydantic_flow.cache import InMemoryCache
from pydantic_flow.flow import Flow
from pydantic_flow.nodes import AgentNode


# Helper to extract result from stream
async def extract_result_from_stream(stream):
    """Extract final result from async stream of progress items."""
    result = None
    async for item in stream:
        if hasattr(item, "result"):
            result = item.result
    return result


class Question(BaseModel):
    """Input model for questions."""

    query: str


class Answer(BaseModel):
    """Output model for answers."""

    response: str


async def main() -> None:
    """Demonstrate cache configuration."""
    print("=== Cache Configuration Example ===\n")

    fast_agent = Agent("test", system_prompt="Fast node agent")
    expensive_agent = Agent("test", system_prompt="Expensive node agent")
    variable_agent = Agent("test", system_prompt="Variable node agent")

    print("1. Creating cache backend and policy:\n")

    cache_backend = InMemoryCache(max_entries=1000)
    default_policy = CachePolicy(enabled=True, ttl=timedelta(hours=1))

    print("   ✓ Backend: InMemoryCache with 1000 max entries")
    print("   ✓ Default policy: 1 hour TTL")

    print("\n2. Creating Flow with cache configuration:\n")

    _ = Flow[Question, Answer](
        input_type=Question,
        output_type=Answer,
        cache_backend=cache_backend,
        default_cache_policy=default_policy,
    )

    print("   ✓ Flow initialized with cache configuration")

    print("\n3. Creating nodes with cache policies:\n")

    expensive_policy = CachePolicy(enabled=True, ttl=timedelta(days=7))
    _ = AgentNode[Question, str](
        agent=expensive_agent,
        prompt_template="Analyze: {query}",
        name="expensive_analysis",
        cache_policy=expensive_policy,
    )
    print("   ✓ expensive_analysis: 7-day TTL")

    fast_policy = CachePolicy(enabled=True, ttl=timedelta(minutes=5))
    _ = AgentNode[Question, str](
        agent=fast_agent,
        prompt_template="Quick check: {query}",
        name="fast_check",
        cache_policy=fast_policy,
    )
    print("   ✓ fast_check: 5-minute TTL")

    _ = AgentNode[Question, str](
        agent=variable_agent,
        prompt_template="Process: {query}",
        name="variable_process",
    )
    print("   ✓ variable_process: Uses default 1-hour TTL")

    print("\n=== Cache Benefits ===\n")
    print("✓ Configurable backend (InMemory, SQLite, Redis)")
    print("✓ Per-node cache policies with TTL support")
    print("✓ Default policy for consistency")
    print("✓ Type-safe configuration with Pydantic")

    print("\n=== Cache Stats ===\n")
    print(f"Backend type: {cache_backend.__class__.__name__}")
    print(f"Default policy enabled: {default_policy.enabled}")
    print(f"Default TTL: {default_policy.ttl}")


if __name__ == "__main__":
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
