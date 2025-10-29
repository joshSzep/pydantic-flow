"""Telemetry demo showing tracing and metrics for pydantic-flow.

This example demonstrates:
1. Setting up OpenTelemetry with minimal configuration
2. Viewing traces that map Flow → Node → Events
3. Metrics for duration, cache hits, and resource usage
4. HITL interruption tracking
"""

import asyncio
from datetime import timedelta

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow import PromptConfig
from pydantic_flow import PromptNode
from pydantic_flow import ToolNode
from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.cache.memory import InMemoryCache
from pydantic_flow.telemetry import setup_telemetry


class Query(BaseModel):
    """Input query."""

    question: str


class Research(BaseModel):
    """Research results."""

    findings: str
    sources: list[str]


class Answer(BaseModel):
    """Final answer."""

    answer: str
    confidence: float


class Results(BaseModel):
    """Flow results."""

    research: Research
    final_answer: Answer


def conduct_research(query: Query) -> Research:
    """Simulate research tool."""
    return Research(
        findings=f"Research findings for: {query.question}",
        sources=["source1.com", "source2.org"],
    )


async def main():
    """Run the telemetry demo."""
    print("🔍 Pydantic-Flow Telemetry Demo")
    print("=" * 60)

    # Setup 1: Console output (for development)
    print("\n📊 Setting up telemetry (console output)...")
    setup_telemetry(
        service_name="telemetry-demo",
        export_to_console=True,
        trace_sample_rate=1.0,
    )

    # Uncomment to export to OTLP endpoint instead:
    # setup_telemetry(
    #     service_name="telemetry-demo",
    #     otlp_endpoint="http://localhost:4318",
    #     trace_sample_rate=1.0
    # )

    print("✅ Telemetry configured\n")

    # Create a simple flow with research and answer nodes
    print("🏗️  Building flow with 2 nodes...")

    # Setup cache to demonstrate cache instrumentation
    cache = InMemoryCache()
    cache_policy = CachePolicy(
        enabled=True,
        ttl=timedelta(hours=1),
    )

    research_node = ToolNode[Query, Research](
        tool_func=conduct_research,
        name="research",
        cache_policy=cache_policy,
    )

    answer_node = PromptNode[Research, Answer](
        prompt="Based on this research: {findings}, provide a concise answer.",
        config=PromptConfig(model="test", result_type=Answer),
        input=research_node.output,
        name="answer",
        cache_policy=cache_policy,
    )

    flow = Flow(
        input_type=Query,
        output_type=Results,
        cache_backend=cache,
        default_cache_policy=cache_policy,
    )
    flow.add_nodes(research_node, answer_node)

    print("✅ Flow built with cache enabled\n")

    # First run - no cache
    print("▶️  First execution (cache miss expected)...")
    query = Query(question="What is machine learning?")
    result1 = await flow.run(query)
    print(f"✅ Got answer: {result1.final_answer.answer[:50]}...\n")

    # Second run - should hit cache
    print("▶️  Second execution (cache hit expected)...")
    result2 = await flow.run(query)
    print(f"✅ Got cached answer: {result2.final_answer.answer[:50]}...\n")

    print("=" * 60)
    print("🎉 Demo complete!")
    print("\nWhat to look for in traces:")
    print("  • FlowRun span covering entire execution")
    print("  • NodeRun spans for each node (research, answer)")
    print("  • Stream events (start, end, tool calls)")
    print("  • Cache operations (lookup, hit/miss, write)")
    print("\nWhat to look for in metrics:")
    print("  • pflow.flow.runs counter (should be 2)")
    print("  • pflow.node.executions counter (should be 4)")
    print("  • pflow.cache.hits counter (should have hits on 2nd run)")
    print("  • pflow.flow.duration.ms histogram")
    print("  • pflow.node.duration.ms histogram")
    print("\nTo view in Grafana/Jaeger:")
    print("  1. Start OTLP collector (e.g., Jaeger all-in-one)")
    print("  2. Run with otlp_endpoint='http://localhost:4318'")
    print("  3. Open Jaeger UI at http://localhost:16686")
    print("  4. Search for service 'telemetry-demo'")


if __name__ == "__main__":
    asyncio.run(main())
