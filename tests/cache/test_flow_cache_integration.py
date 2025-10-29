"""Integration tests for Flow with cache backend.

Tests cache integration at the stepper engine level.
"""

from datetime import timedelta

from pydantic import BaseModel
import pytest

from pydantic_flow.cache.base import CacheContentType
from pydantic_flow.cache.base import CacheEntry
from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.cache.memory import InMemoryCache
from pydantic_flow.core.errors import FlowError
from pydantic_flow.flow.flow import Flow


class SimpleInput(BaseModel):
    """Simple input model."""

    value: int


class SimpleOutput(BaseModel):
    """Simple output model."""

    doubled: int


@pytest.mark.asyncio
async def test_flow_cache_delete_without_backend():
    """Test cache_delete raises error when no backend configured."""
    flow = Flow[SimpleInput, SimpleOutput](
        input_type=SimpleInput,
        output_type=SimpleOutput,
    )

    with pytest.raises(FlowError, match="No cache backend configured"):
        await flow.cache_delete("some_key")


@pytest.mark.asyncio
async def test_flow_cache_invalidate_without_backend():
    """Test cache_invalidate raises error when no backend configured."""
    flow = Flow[SimpleInput, SimpleOutput](
        input_type=SimpleInput,
        output_type=SimpleOutput,
    )

    with pytest.raises(FlowError, match="No cache backend configured"):
        await flow.cache_invalidate("some_namespace")


@pytest.mark.asyncio
async def test_flow_cache_delete_with_backend():
    """Test cache_delete with configured backend."""
    cache = InMemoryCache()
    await cache.start()

    try:
        flow = Flow[SimpleInput, SimpleOutput](
            input_type=SimpleInput,
            output_type=SimpleOutput,
            cache_backend=cache,
        )

        # Add a cache entry
        entry = CacheEntry(
            value="test_value",
            content_type=CacheContentType.LLM_COMPLETION,
            created_at=0.0,
            ttl_seconds=None,
        )
        await cache.set("test_key", entry)

        # Verify entry exists
        result = await cache.get("test_key")
        assert result is not None

        # Delete via flow
        await flow.cache_delete("test_key")

        # Verify entry is gone
        result = await cache.get("test_key")
        assert result is None

    finally:
        await cache.stop()


@pytest.mark.asyncio
async def test_flow_cache_invalidate_with_backend():
    """Test cache_invalidate with configured backend."""
    cache = InMemoryCache()
    await cache.start()

    try:
        flow = Flow[SimpleInput, SimpleOutput](
            input_type=SimpleInput,
            output_type=SimpleOutput,
            cache_backend=cache,
        )

        # Add cache entries in a namespace
        entry = CacheEntry(
            value="test_value",
            content_type=CacheContentType.LLM_COMPLETION,
            created_at=0.0,
            ttl_seconds=None,
        )

        # Use the correct prefix format that InMemoryCache expects
        await cache.set("pf:ns:test_namespace:key1", entry)
        await cache.set("pf:ns:test_namespace:key2", entry)
        await cache.set("pf:ns:other_namespace:key1", entry)

        # Invalidate test_namespace
        count = await flow.cache_invalidate("test_namespace")
        assert count == 2

        # Verify test_namespace entries are gone
        result1 = await cache.get("pf:ns:test_namespace:key1")
        result2 = await cache.get("pf:ns:test_namespace:key2")
        assert result1 is None
        assert result2 is None

        # Verify other_namespace entry still exists
        result3 = await cache.get("pf:ns:other_namespace:key1")
        assert result3 is not None

    finally:
        await cache.stop()


@pytest.mark.asyncio
async def test_flow_initializes_with_cache_backend():
    """Test Flow accepts cache_backend and default_cache_policy."""
    cache = InMemoryCache()
    policy = CachePolicy(enabled=True, ttl=timedelta(hours=1))

    flow = Flow[SimpleInput, SimpleOutput](
        input_type=SimpleInput,
        output_type=SimpleOutput,
        cache_backend=cache,
        default_cache_policy=policy,
    )

    assert flow._cache_backend is cache
    assert flow._default_cache_policy is policy


@pytest.mark.asyncio
async def test_flow_compile_passes_cache_to_stepper():
    """Test that compile() passes cache backend to StepperEngine."""
    cache = InMemoryCache()
    policy = CachePolicy(enabled=True)

    flow = Flow[SimpleInput, SimpleOutput](
        input_type=SimpleInput,
        output_type=SimpleOutput,
        cache_backend=cache,
        default_cache_policy=policy,
    )

    # Need at least one node and entry point for stepper mode
    from pydantic_flow.nodes.base import BaseNode
    from pydantic_flow.streaming.events import StreamEnd

    class TestNode(BaseNode[SimpleInput, int]):
        """Test node."""

        async def astream(self, input_data: SimpleInput):
            """Stream result."""
            yield StreamEnd(node_id=self.name)

    node = TestNode(name="test")
    flow.add_nodes(node)
    flow.set_entry_nodes("test")

    compiled = flow.compile()

    # Verify stepper engine has cache backend
    if compiled.use_stepper and compiled.engine is not None:
        assert compiled.engine.cache_backend is cache
        assert compiled.engine.default_cache_policy is policy
