"""Simplified Telemetry Demo: Shows OpenTelemetry integration with cache instrumentation.

Run this to see:
- flow_run spans with flow.id and run.id attributes
- node_run spans nested under flows
- cache_lookup spans showing miss then hit
- Console output of all traces
"""

import asyncio
from datetime import timedelta

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow.cache import CachePolicy
from pydantic_flow.cache import InMemoryCache
from pydantic_flow.nodes import ToolNode
from pydantic_flow.telemetry import setup_telemetry


# Helper to extract result from stream
async def extract_result_from_stream(stream):
    """Extract final result from async stream of progress items."""
    result = None
    async for item in stream:
        if hasattr(item, "result"):
            result = item.result
    return result


class Query(BaseModel):
    """User query."""

    question: str


class Answer(BaseModel):
    """Answer to query."""

    response: str
    confidence: float


def answer_query(query: Query) -> Answer:
    """Simple function that answers queries."""
    return Answer(
        response=f"The answer to '{query.question}' is 42",
        confidence=0.95,
    )


async def main() -> None:
    """Run the telemetry demo."""
    print("🔍 Pydantic-Flow Telemetry Demo")
    print("=" * 60)

    # Setup telemetry with console output for visibility
    print("\n📊 Setting up telemetry...")
    setup_telemetry(
        service_name="telemetry-demo",
        export_to_console=True,
        trace_sample_rate=1.0,
    )
    print("✅ Telemetry configured\n")

    # Create flow with cache enabled
    print("🏗️  Building flow...")
    cache = InMemoryCache()

    answer_node = ToolNode[Query, Answer](
        tool_func=answer_query,
        name="answer",
        cache_policy=CachePolicy(enabled=True, ttl=timedelta(hours=1)),
    )

    flow = Flow(
        input_type=Query,
        output_type=Answer,
        cache_backend=cache,
    )
    flow.add_nodes(answer_node)
    print("✅ Flow built\n")

    query = Query(question="What is the meaning of life?")

    # First execution - cache miss
    print("▶️  First execution (cache miss)...")
    result1 = await extract_result_from_stream(flow.astream(query))
    print(f"✅ Result: {result1.response}\n")

    # Second execution - cache hit
    print("▶️  Second execution (cache hit)...")
    result2 = await extract_result_from_stream(flow.astream(query))
    print(f"✅ Result: {result2.response}\n")

    print("=" * 60)
    print("📊 Check the JSON trace output above!")
    print("\nYou should see:")
    print("  • flow_run spans with pflow.flow.id and pflow.run.id")
    print("  • outcome='success' attribute")
    print("  • pflow.execution.mode='dag'")
    print("\n💡 To export to OTLP collector:")
    print("  setup_telemetry(otlp_endpoint='http://localhost:4318')")


if __name__ == "__main__":
    asyncio.run(main())
