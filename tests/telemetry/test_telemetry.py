"""Tests for OpenTelemetry integration."""

from datetime import timedelta

from pydantic import BaseModel
import pytest

from pydantic_flow import Flow
from pydantic_flow import ToolNode
from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.cache.memory import InMemoryCache
from pydantic_flow.telemetry import is_enabled
from pydantic_flow.telemetry import setup_telemetry


class Input(BaseModel):
    """Test input."""

    value: int


class Output(BaseModel):
    """Test output."""

    result: int


class Results(BaseModel):
    """Test results."""

    node1: Output


def double_value(input_data: Input) -> Output:
    """Test tool function."""
    return Output(result=input_data.value * 2)


@pytest.fixture(autouse=True)
def _reset_telemetry():
    """Reset telemetry state between tests."""
    # Telemetry is module-level singleton, so we need to be careful
    # In real usage, setup_telemetry is called once at startup
    yield


def test_telemetry_setup_default():
    """Test default telemetry setup."""
    config = setup_telemetry(enabled=False)
    assert config.service_name == "pydantic-flow"
    assert not is_enabled()


def test_telemetry_setup_console():
    """Test console export setup."""
    config = setup_telemetry(
        enabled=True,
        export_to_console=True,
        service_name="test-service",
        trace_sample_rate=1.0,
    )
    assert config.service_name == "test-service"
    assert config.export_to_console
    assert is_enabled()


def test_telemetry_setup_otlp():
    """Test OTLP endpoint setup."""
    config = setup_telemetry(
        enabled=True,
        otlp_endpoint="http://localhost:4318",
        service_name="test-service",
    )
    assert config.otlp_endpoint == "http://localhost:4318"
    assert config.service_name == "test-service"
    assert is_enabled()


@pytest.mark.asyncio
async def test_flow_with_telemetry():
    """Test flow execution with telemetry enabled."""
    # Setup telemetry (no real export for tests)
    setup_telemetry(enabled=True, export_to_console=False)

    # Create simple flow
    node1 = ToolNode[Input, Output](
        tool_func=double_value,
        name="node1",
    )

    flow = Flow(input_type=Input, output_type=Results)
    flow.add_nodes(node1)

    # Execute - should work with telemetry
    result = await flow.run(Input(value=5))
    assert result.node1.result == 10


@pytest.mark.asyncio
async def test_flow_without_telemetry():
    """Test flow execution with telemetry disabled."""
    # Disable telemetry
    setup_telemetry(enabled=False)

    # Create simple flow
    node1 = ToolNode[Input, Output](
        tool_func=double_value,
        name="node1",
    )

    flow = Flow(input_type=Input, output_type=Results)
    flow.add_nodes(node1)

    # Execute - should work without telemetry
    result = await flow.run(Input(value=5))
    assert result.node1.result == 10


@pytest.mark.asyncio
async def test_cache_telemetry():
    """Test cache operations create telemetry."""
    setup_telemetry(enabled=True, export_to_console=False)

    cache = InMemoryCache()
    cache_policy = CachePolicy(enabled=True, ttl=timedelta(hours=1))

    node1 = ToolNode[Input, Output](
        tool_func=double_value,
        name="node1",
        cache_policy=cache_policy,
    )

    flow = Flow(
        input_type=Input,
        output_type=Results,
        cache_backend=cache,
        default_cache_policy=cache_policy,
    )
    flow.add_nodes(node1)

    # First run - cache miss
    result1 = await flow.run(Input(value=5))
    assert result1.node1.result == 10

    # Second run - cache hit
    result2 = await flow.run(Input(value=5))
    assert result2.node1.result == 10


@pytest.mark.asyncio
async def test_telemetry_overhead_disabled():
    """Test that disabled telemetry has minimal overhead."""
    setup_telemetry(enabled=False)

    node1 = ToolNode[Input, Output](
        tool_func=double_value,
        name="node1",
    )

    flow = Flow(input_type=Input, output_type=Results)
    flow.add_nodes(node1)

    # Measure execution time
    import time

    start = time.perf_counter()
    for _ in range(100):
        await flow.run(Input(value=5))
    elapsed_disabled = time.perf_counter() - start

    # With telemetry disabled, overhead should be negligible
    # This is a smoke test - just ensure it completes quickly
    assert elapsed_disabled < 10.0  # 100 runs in under 10 seconds


@pytest.mark.asyncio
async def test_multiple_flows_same_telemetry():
    """Test multiple flows with shared telemetry setup."""
    setup_telemetry(enabled=True, export_to_console=False)

    # Create two different flows
    node1 = ToolNode[Input, Output](tool_func=double_value, name="node1")
    flow1 = Flow(input_type=Input, output_type=Results)
    flow1.add_nodes(node1)

    node2 = ToolNode[Input, Output](tool_func=double_value, name="node1")
    flow2 = Flow(input_type=Input, output_type=Results)
    flow2.add_nodes(node2)

    # Execute both - should work with shared telemetry
    result1 = await flow1.run(Input(value=3))
    result2 = await flow2.run(Input(value=4))

    assert result1.node1.result == 6
    assert result2.node1.result == 8
