"""Advanced example demonstrating real-world FlowNode memory mode use cases.

This example shows practical scenarios where different memory modes are useful:
1. Parallel research tasks with ISOLATED mode
2. Background context enrichment with READONLY mode
3. Sequential conversation with SHARED mode
"""

import asyncio

from pydantic import BaseModel
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import SystemPromptPart

from pydantic_flow import Flow
from pydantic_flow import MemoryConfig
from pydantic_flow.memory import MemoryMode
from pydantic_flow.nodes import FlowNode
from pydantic_flow.nodes import ToolNode


# Helper to extract result from stream
async def extract_result_from_stream(stream):
    """Extract final result from async stream of progress items."""
    result = None
    async for item in stream:
        if hasattr(item, "result"):
            result = item.result
    return result


# Data models
class Query(BaseModel):
    """User query."""

    question: str


class FindingsText(BaseModel):
    """Text findings."""

    text: str


class ResearchResult(BaseModel):
    """Result from research."""

    findings: FindingsText


class EnrichedContext(BaseModel):
    """Enriched context data."""

    context: str


class SummaryResult(BaseModel):
    """Summary of findings."""

    summary: str


class ParallelResearchOutput(BaseModel):
    """Output from parallel research."""

    topic_a: ResearchResult
    topic_b: ResearchResult


class EnrichmentOutput(BaseModel):
    """Output from context enrichment."""

    enriched: EnrichedContext


class FinalReport(BaseModel):
    """Final report."""

    result: SummaryResult


# Tool functions
async def research_topic_a(query: Query) -> ResearchResult:
    """Research topic A."""
    return ResearchResult(
        findings=FindingsText(text=f"Topic A findings for: {query.question}")
    )


async def research_topic_b(query: Query) -> ResearchResult:
    """Research topic B."""
    return ResearchResult(
        findings=FindingsText(text=f"Topic B findings for: {query.question}")
    )


async def enrich_context(query: Query) -> EnrichedContext:
    """Enrich context data."""
    return EnrichedContext(context=f"Enriched context: {query.question}")


async def summarize_findings(research: ParallelResearchOutput) -> SummaryResult:
    """Summarize research findings."""
    combined = f"{research.topic_a.findings} | {research.topic_b.findings}"
    return SummaryResult(summary=f"Summary: {combined}")


async def scenario_parallel_research():
    """Scenario: Parallel research tasks that shouldn't see each other's work.

    Use case: Running multiple independent research sub-flows in parallel
    where each should maintain its own conversation history without
    cross-contamination.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 1: Parallel Independent Research")
    print("=" * 70)
    print("Use case: Multiple research tasks that need isolation")
    print("Memory mode: ISOLATED (prevents cross-contamination)\n")

    # Create parent orchestrator flow
    parent_flow = Flow[Query, ParallelResearchOutput](
        input_type=Query,
        output_type=ParallelResearchOutput,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )

    # Add initial context to parent memory
    if parent_flow._conversation_memory:
        parent_flow._conversation_memory.append(
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content="Research coordinator: Starting parallel research"
                    )
                ]
            )
        )

    # Create sub-flow A with ISOLATED memory
    sub_flow_a = Flow[Query, ResearchResult](
        input_type=Query,
        output_type=ResearchResult,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )
    researcher_a = ToolNode[Query, ResearchResult](
        tool_func=research_topic_a, name="findings"
    )
    sub_flow_a.add_nodes(researcher_a)
    # Flows execute directly - no compilation needed

    # Create sub-flow B with ISOLATED memory
    sub_flow_b = Flow[Query, ResearchResult](
        input_type=Query,
        output_type=ResearchResult,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )
    researcher_b = ToolNode[Query, ResearchResult](
        tool_func=research_topic_b, name="findings"
    )
    sub_flow_b.add_nodes(researcher_b)
    # Flows execute directly - no compilation needed

    # Wrap sub-flows with ISOLATED mode
    flow_node_a = FlowNode[Query, ResearchResult](
        flow=sub_flow_a,
        name="topic_a",
        memory_mode=MemoryMode.ISOLATED,
        seed_isolated_memory=False,  # Each starts fresh
    )

    flow_node_b = FlowNode[Query, ResearchResult](
        flow=sub_flow_b,
        name="topic_b",
        memory_mode=MemoryMode.ISOLATED,
        seed_isolated_memory=False,  # Each starts fresh
    )

    parent_flow.add_nodes(flow_node_a, flow_node_b)
    # Flows execute directly - no compilation needed

    # Execute
    result = await extract_result_from_stream(
        parent_flow.astream(Query(question="AI safety"))
    )

    print(f"Topic A: {result.topic_a.findings.text}")
    print(f"Topic B: {result.topic_b.findings.text}")

    if parent_flow._conversation_memory:
        print(f"\nParent memory: {len(parent_flow._conversation_memory)} message")
        print("✓ Each research task had isolated memory")
        print("✓ No cross-contamination between parallel tasks")


async def scenario_context_enrichment():
    """Scenario: Background context enrichment without polluting main memory.

    Use case: Enriching context by querying external sources without adding
    those queries to the main conversation history.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: Background Context Enrichment")
    print("=" * 70)
    print("Use case: Enrich context without polluting main conversation")
    print("Memory mode: READONLY (can read but not modify parent)\n")

    # Create parent flow
    parent_flow = Flow[Query, EnrichmentOutput](
        input_type=Query,
        output_type=EnrichmentOutput,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )

    # Add user conversation to parent memory
    if parent_flow._conversation_memory:
        parent_flow._conversation_memory.append(
            ModelRequest(
                parts=[SystemPromptPart(content="User: Tell me about machine learning")]
            )
        )
        parent_flow._conversation_memory.append(
            ModelRequest(
                parts=[SystemPromptPart(content="Assistant: ML is a subset of AI...")]
            )
        )

    # Create enrichment sub-flow
    enrichment_flow = Flow[Query, EnrichedContext](
        input_type=Query,
        output_type=EnrichedContext,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )
    enricher = ToolNode[Query, EnrichedContext](
        tool_func=enrich_context, name="context"
    )
    enrichment_flow.add_nodes(enricher)
    # Flows execute directly - no compilation needed

    # Wrap with READONLY mode
    enrichment_node = FlowNode[Query, EnrichedContext](
        flow=enrichment_flow,
        name="enriched",
        memory_mode=MemoryMode.READONLY,  # Can read history but not modify
    )

    parent_flow.add_nodes(enrichment_node)
    # Flows execute directly - no compilation needed

    print("Parent memory before enrichment: 2 messages")

    # Execute
    result = await extract_result_from_stream(
        parent_flow.astream(Query(question="deep learning"))
    )

    print(f"Enriched: {result.enriched.context}")

    if parent_flow._conversation_memory:
        mem_count = len(parent_flow._conversation_memory)
        print(f"\nParent memory after enrichment: {mem_count} messages")
        print("✓ Enrichment sub-flow read conversation history")
        print("✓ But didn't add its own queries to main conversation")
        print("✓ Main conversation remains clean and focused")


async def scenario_sequential_conversation():
    """Scenario: Sequential conversation flow with full memory sharing.

    Use case: Building a conversation where each step needs full context
    from all previous steps.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: Sequential Conversation with Shared Memory")
    print("=" * 70)
    print("Use case: Multi-step conversation needing full context")
    print("Memory mode: SHARED (default, full memory sharing)\n")

    # Create parent flow
    parent_flow = Flow[Query, FinalReport](
        input_type=Query,
        output_type=FinalReport,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )

    # Create research sub-flow
    research_flow = Flow[Query, ParallelResearchOutput](
        input_type=Query,
        output_type=ParallelResearchOutput,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )

    researcher_a = ToolNode[Query, ResearchResult](
        tool_func=research_topic_a, name="topic_a"
    )
    researcher_b = ToolNode[Query, ResearchResult](
        tool_func=research_topic_b, name="topic_b"
    )
    research_flow.add_nodes(researcher_a, researcher_b)
    # Flows execute directly - no compilation needed

    # Create summary sub-flow
    summary_flow = Flow[ParallelResearchOutput, SummaryResult](
        input_type=ParallelResearchOutput,
        output_type=SummaryResult,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )
    summarizer = ToolNode[ParallelResearchOutput, SummaryResult](
        tool_func=summarize_findings, name="summary"
    )
    summary_flow.add_nodes(summarizer)
    # Flows execute directly - no compilation needed

    # Wrap with SHARED mode (default)
    research_node = FlowNode[Query, ParallelResearchOutput](
        flow=research_flow,
        name="research",
        memory_mode=MemoryMode.SHARED,  # Shares parent memory
    )

    summary_node = FlowNode[ParallelResearchOutput, SummaryResult](
        flow=summary_flow,
        input=research_node.output,
        name="result",
        memory_mode=MemoryMode.SHARED,  # Also shares parent memory
    )

    parent_flow.add_nodes(research_node, summary_node)  # type: ignore[arg-type]
    # Flows execute directly - no compilation needed

    # Execute
    result = await extract_result_from_stream(
        parent_flow.astream(Query(question="neural networks"))
    )

    print(f"Final summary: {result.result.summary}")

    if parent_flow._conversation_memory:
        mem_count = len(parent_flow._conversation_memory)
        print(f"\nParent memory: {mem_count} message(s)")
        print("✓ All steps shared same conversation memory")
        print("✓ Each step had full context from previous steps")
        print("✓ Enables coherent multi-step conversations")


async def main():
    """Run all scenarios."""
    print("\n" + "=" * 70)
    print("ADVANCED MEMORY MODES: Real-World Use Cases")
    print("=" * 70)

    await scenario_parallel_research()
    await scenario_context_enrichment()
    await scenario_sequential_conversation()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. ISOLATED: Use for parallel independent tasks")
    print("   - Prevents cross-contamination")
    print("   - Each task has clean conversation context")
    print()
    print("2. READONLY: Use for background enrichment")
    print("   - Read context without modifying it")
    print("   - Keep main conversation clean")
    print()
    print("3. SHARED: Use for sequential conversations")
    print("   - Full context sharing across steps")
    print("   - Coherent multi-step workflows")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
