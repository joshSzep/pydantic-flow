"""Integration tests for cache middleware with real flow execution.

These tests exercise the cache middleware through actual Flow execution
to improve coverage of cache integration code paths.
"""

from datetime import timedelta

from pydantic import BaseModel
import pytest

from pydantic_flow import Flow
from pydantic_flow import ToolNode
from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.cache.memory import InMemoryCache


class Input(BaseModel):
    """Test input model."""

    value: int


class Output(BaseModel):
    """Test output model."""

    result: int


class FlowOutput(BaseModel):
    """Test flow output model."""

    compute: Output


@pytest.mark.asyncio
async def test_middleware_cache_with_backend():
    """Test flow execution with cache backend configured."""
    cache = InMemoryCache()

    def compute(inp: Input) -> Output:
        return Output(result=inp.value * 2)

    node = ToolNode[Input, Output](
        tool_func=compute,
        name="compute",
    )

    policy = CachePolicy(
        enabled=True,
        ttl=timedelta(seconds=60),
    )

    flow = Flow[Input, FlowOutput](
        input_type=Input,
        output_type=FlowOutput,
        cache_backend=cache,
        default_cache_policy=policy,
    )
    flow.add_nodes(node)

    # Should execute successfully with cache configured
    result = await flow.run(Input(value=5))
    assert result.compute.result == 10


@pytest.mark.asyncio
async def test_middleware_without_cache_backend():
    """Test flow execution without cache backend."""
    call_count = 0

    def compute(inp: Input) -> Output:
        nonlocal call_count
        call_count += 1
        return Output(result=inp.value * 2)

    node = ToolNode[Input, Output](
        tool_func=compute,
        name="compute",
    )

    flow = Flow[Input, FlowOutput](
        input_type=Input,
        output_type=FlowOutput,
        # No cache backend
    )
    flow.add_nodes(node)

    # Both executions should call the function
    result1 = await flow.run(Input(value=5))
    assert result1.compute.result == 10
    assert call_count == 1

    result2 = await flow.run(Input(value=5))
    assert result2.compute.result == 10
    assert call_count == 2  # No caching


@pytest.mark.asyncio
async def test_middleware_cache_operations():
    """Test cache operations via flow methods."""
    cache = InMemoryCache()

    def compute(inp: Input) -> Output:
        return Output(result=inp.value * 2)

    node = ToolNode[Input, Output](
        tool_func=compute,
        name="compute",
    )

    flow = Flow[Input, FlowOutput](
        input_type=Input,
        output_type=FlowOutput,
        cache_backend=cache,
    )
    flow.add_nodes(node)

    # Test cache_delete and cache_invalidate methods exist
    await flow.cache_delete("test_key")
    deleted_count = await flow.cache_invalidate("test_namespace")
    assert deleted_count == 0  # No keys in that namespace
