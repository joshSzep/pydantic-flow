"""Integration test demonstrating complete cache stack."""

import asyncio

import pytest

from pydantic_flow.cache import CachePolicy
from pydantic_flow.cache import CacheScope
from pydantic_flow.cache import InMemoryCache
from pydantic_flow.cache.key import build_llm_cache_key
from pydantic_flow.cache.middleware import maybe_cached_execute


@pytest.mark.asyncio
async def test_full_cache_integration() -> None:
    """Test complete caching flow from key generation to middleware."""
    cache = InMemoryCache(max_entries=100)
    await cache.start()

    try:
        policy = CachePolicy(
            enabled=True,
            scope=CacheScope.GLOBAL(),
            ttl=None,
        )

        messages = [{"role": "user", "content": "Hello, world!"}]

        call_count = 0

        async def expensive_computation() -> str:
            """Simulate expensive computation."""
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return "Computed result"

        result1, events1 = await maybe_cached_execute(
            node_name="test_node",
            inputs={"messages": messages},
            exec_fn=expensive_computation,
            backend=cache,
            policy=policy,
        )

        assert call_count == 1
        assert result1 == "Computed result"
        assert len(events1) == 2
        assert events1[0].type == "cache_miss"
        assert events1[1].type == "cache_write"

        result2, events2 = await maybe_cached_execute(
            node_name="test_node",
            inputs={"messages": messages},
            exec_fn=expensive_computation,
            backend=cache,
            policy=policy,
        )

        assert call_count == 1
        assert result2 == "Computed result"
        assert len(events2) == 1
        assert events2[0].type == "cache_hit"

        size = await cache.size()
        assert size == 1

    finally:
        await cache.stop()


@pytest.mark.asyncio
async def test_cache_with_namespaces() -> None:
    """Test namespace isolation."""
    cache = InMemoryCache()
    await cache.start()

    try:
        policy_prod = CachePolicy(scope=CacheScope.NAMESPACE("production"))
        policy_dev = CachePolicy(scope=CacheScope.NAMESPACE("development"))

        async def compute() -> str:
            return "result"

        result1, _ = await maybe_cached_execute(
            "node1",
            {"input": "data"},
            compute,
            cache,
            policy_prod,
        )
        result2, _ = await maybe_cached_execute(
            "node1",
            {"input": "data"},
            compute,
            cache,
            policy_dev,
        )

        assert result1 == result2

        size = await cache.size()
        assert size == 2

        deleted = await cache.invalidate_namespace("production")
        assert deleted == 1

        size = await cache.size()
        assert size == 1

    finally:
        await cache.stop()


@pytest.mark.asyncio
async def test_cache_key_determinism() -> None:
    """Test that identical inputs produce identical cache keys."""
    messages = [{"role": "user", "content": "test"}]

    key1 = build_llm_cache_key(
        "openai",
        "gpt-4",
        messages,
        temperature=0.7,
        seed=42,
    )

    key2 = build_llm_cache_key(
        "openai",
        "gpt-4",
        messages,
        temperature=0.7,
        seed=42,
    )

    assert key1 == key2

    key3 = build_llm_cache_key(
        "openai",
        "gpt-4",
        messages,
        temperature=0.8,
        seed=42,
    )

    assert key1 != key3
