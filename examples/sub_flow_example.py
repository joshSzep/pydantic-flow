"""Sub-flow composition example using FlowNode.

This example demonstrates how to build hierarchical workflows by composing
smaller flows into larger ones using FlowNode. This enables building complex
workflows from simpler, reusable sub-flows.
"""

import asyncio

from pydantic import BaseModel

from pflow import Flow
from pflow import FlowNode
from pflow import ToolNode


# Input/Output Models
class UserQuery(BaseModel):
    """User input for the complete workflow."""

    topic: str
    style: str = "professional"


class ResearchData(BaseModel):
    """Output from research sub-flow."""

    facts: str
    sources: str


class ResearchResults(BaseModel):
    """Results from research sub-flow."""

    research: ResearchData


class SummaryData(BaseModel):
    """Output from summary generation."""

    title: str
    content: str
    word_count: int


class SummaryResults(BaseModel):
    """Results from summary sub-flow."""

    summary: SummaryData


class FinalReport(BaseModel):
    """Final workflow output combining all sub-flows."""

    research_flow: ResearchResults
    summary_flow: SummaryResults


class NestedFlowResults(BaseModel):
    """Results from the complete nested workflow."""

    complete_workflow: FinalReport


# Helper functions for demonstration
def mock_research_api(query: UserQuery) -> ResearchData:
    """Mock research API that gathers information."""
    return ResearchData(
        facts=f"Key facts about {query.topic}: Fact 1, Fact 2, Fact 3",
        sources=f"Sources for {query.topic}: Source A, Source B, Source C",
    )


def parse_research_output(raw_data: str) -> ResearchData:
    """Parse raw research data into structured format."""
    lines = raw_data.split("\n")
    facts = lines[0] if lines else "No facts found"
    sources = lines[1] if len(lines) > 1 else "No sources found"

    return ResearchData(facts=facts, sources=sources)


def generate_summary(research_results: ResearchResults) -> SummaryData:
    """Generate a summary from research data."""
    research = research_results.research
    word_count = len(research.facts.split()) + len(research.sources.split())

    return SummaryData(
        title="Summary of Research",
        content=f"Summary: {research.facts[:100]}...",
        word_count=word_count,
    )


async def create_research_sub_flow() -> Flow[UserQuery, ResearchResults]:
    """Create a sub-flow that handles research gathering."""
    # Create the research flow
    research_flow = Flow(input_type=UserQuery, output_type=ResearchResults)

    # Add research API call
    research_node = ToolNode[UserQuery, ResearchData](
        tool_func=mock_research_api,
        name="research",
    )

    research_flow.add_nodes(research_node)
    return research_flow


async def create_summary_sub_flow() -> Flow[ResearchResults, SummaryResults]:
    """Create a sub-flow that handles summary generation."""
    # Create the summary flow
    summary_flow = Flow(input_type=ResearchResults, output_type=SummaryResults)

    # Add summary generation
    summary_node = ToolNode[ResearchResults, SummaryData](
        tool_func=generate_summary,
        name="summary",
    )

    summary_flow.add_nodes(summary_node)
    return summary_flow


async def demonstrate_simple_sub_flow():
    """Demonstrate using a single sub-flow within a parent flow."""
    print("=== Simple Sub-flow Example ===")

    # Create research sub-flow
    research_sub_flow = await create_research_sub_flow()

    # Create parent flow that uses the research sub-flow
    parent_flow = Flow(input_type=UserQuery, output_type=FinalReport)

    # Wrap sub-flow as a node
    research_flow_node = FlowNode[UserQuery, ResearchResults](
        flow=research_sub_flow,
        name="research_flow",
    )

    # Create summary sub-flow and wrap it too
    summary_sub_flow = await create_summary_sub_flow()
    summary_flow_node = FlowNode[ResearchResults, SummaryResults](
        flow=summary_sub_flow,
        input=research_flow_node.output,  # Chain flows together
        name="summary_flow",
    )

    # Add both flow nodes to parent
    parent_flow.add_nodes(research_flow_node, summary_flow_node)

    # Execute the complete workflow
    query = UserQuery(topic="artificial intelligence", style="technical")
    result = await parent_flow.run(query)

    print(f"Research Facts: {result.research_flow.research.facts}")
    print(f"Summary Title: {result.summary_flow.summary.title}")
    print(f"Word Count: {result.summary_flow.summary.word_count}")
    print()


async def demonstrate_nested_sub_flows():
    """Demonstrate deeply nested sub-flows."""
    print("=== Nested Sub-flows Example ===")

    # Level 1: Basic research flow
    level1_flow = await create_research_sub_flow()
    print(f"Level 1 Flow: {level1_flow}")

    # Level 2: Wrapper flow that contains the research flow
    level2_flow = Flow(input_type=UserQuery, output_type=FinalReport)

    # Wrap level 1 in a FlowNode
    level1_node = FlowNode[UserQuery, ResearchResults](
        flow=level1_flow,
        name="research_flow",
    )

    # Add summary flow to level 2
    summary_sub_flow = await create_summary_sub_flow()
    summary_node = FlowNode[ResearchResults, SummaryResults](
        flow=summary_sub_flow,
        input=level1_node.output,
        name="summary_flow",
    )

    level2_flow.add_nodes(level1_node, summary_node)
    print(f"Level 2 Flow: {level2_flow}")

    # Level 3: Top-level flow that wraps everything
    level3_flow = Flow(input_type=UserQuery, output_type=NestedFlowResults)

    level2_node = FlowNode[UserQuery, FinalReport](
        flow=level2_flow,
        name="complete_workflow",
    )

    level3_flow.add_nodes(level2_node)
    print(f"Level 3 Flow: {level3_flow}")

    # Execute the deeply nested workflow
    query = UserQuery(topic="machine learning", style="academic")
    result = await level3_flow.run(query)

    # Verify the nested structure worked
    nested_result = result.complete_workflow
    research_facts = nested_result.research_flow.research.facts[:50]
    print(f"Nested Result - Research: {research_facts}...")
    print(f"Nested Result - Summary: {nested_result.summary_flow.summary.title}")
    print()


async def demonstrate_reusable_sub_flows():
    """Demonstrate reusing the same sub-flow in multiple contexts."""
    print("=== Reusable Sub-flows Example ===")

    # Create a reusable research sub-flow
    reusable_research_flow = await create_research_sub_flow()

    # Use it in multiple parent flows
    flow_a = Flow(input_type=UserQuery, output_type=FinalReport)
    flow_b = Flow(input_type=UserQuery, output_type=FinalReport)

    # Same sub-flow, different contexts
    research_node_a = FlowNode[UserQuery, ResearchResults](
        flow=reusable_research_flow,
        name="research_flow",
    )

    research_node_b = FlowNode[UserQuery, ResearchResults](
        flow=reusable_research_flow,
        name="research_flow",
    )

    # Add summary flows to both
    summary_flow = await create_summary_sub_flow()

    summary_node_a = FlowNode[ResearchResults, SummaryResults](
        flow=summary_flow,
        input=research_node_a.output,
        name="summary_flow",
    )

    summary_node_b = FlowNode[ResearchResults, SummaryResults](
        flow=summary_flow,
        input=research_node_b.output,
        name="summary_flow",
    )

    flow_a.add_nodes(research_node_a, summary_node_a)
    flow_b.add_nodes(research_node_b, summary_node_b)

    # Execute both flows with different inputs
    query_a = UserQuery(topic="quantum computing", style="beginner")
    query_b = UserQuery(topic="blockchain technology", style="expert")

    result_a = await flow_a.run(query_a)
    result_b = await flow_b.run(query_b)

    print("Flow A Topic: quantum computing")
    print(f"Flow A Research: {result_a.research_flow.research.facts[:50]}...")

    print("Flow B Topic: blockchain technology")
    print(f"Flow B Research: {result_b.research_flow.research.facts[:50]}...")
    print()


async def main():
    """Run all sub-flow composition examples."""
    print("🔄 pflow Sub-flow Composition Examples\n")

    await demonstrate_simple_sub_flow()
    await demonstrate_nested_sub_flows()
    await demonstrate_reusable_sub_flows()

    print("✅ Sub-flow composition examples completed!")
    print("\nKey Benefits of FlowNode:")
    print("- Hierarchical composition of complex workflows")
    print("- Reusable sub-flows across different contexts")
    print("- Type-safe flow composition with full IDE support")
    print("- Encapsulated sub-workflow logic with clean interfaces")


if __name__ == "__main__":
    asyncio.run(main())
