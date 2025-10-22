"""Hierarchical Flow Example: AI Content Creation Pipeline.

This example demonstrates a sophisticated hierarchy of flows for an AI-powered
content creation system. It shows how to build complex workflows from simpler,
reusable sub-flows using FlowNode.

The pipeline consists of:
1. Research Flow - Gathers information and validates sources
2. Content Planning Flow - Creates outlines and structure
3. Writing Flow - Generates content with review cycles
4. Publishing Flow - Formats and optimizes for different platforms
5. Master Content Pipeline - Orchestrates the entire process
"""

import asyncio
from datetime import datetime

from pydantic import BaseModel
from pydantic import Field

from pflow import Flow
from pflow import FlowNode
from pflow import ToolNode

# ================================
# Data Models for Content Pipeline
# ================================


class ContentRequest(BaseModel):
    """Initial request for content creation."""

    topic: str
    target_audience: str
    content_type: str = "blog_post"  # blog_post, article, social_media
    tone: str = "professional"  # professional, casual, technical
    word_count_target: int = 1000


class ResearchData(BaseModel):
    """Raw research information."""

    facts: list[str]
    sources: list[str]
    key_insights: list[str]
    credibility_score: float = Field(ge=0.0, le=1.0)


# Research quality threshold for validation
RESEARCH_QUALITY_THRESHOLD = 0.8


class ValidatedResearch(BaseModel):
    """Research validated for accuracy and relevance."""

    verified_facts: list[str]
    trusted_sources: list[str]
    main_insights: list[str]
    research_quality: str  # high, medium, low


class ResearchResults(BaseModel):
    """Complete research phase output."""

    research_data: ResearchData
    validated_research: ValidatedResearch


class ContentOutline(BaseModel):
    """Structured content outline."""

    title: str
    introduction: str
    main_sections: list[str]
    conclusion: str
    estimated_word_count: int


class ContentStrategy(BaseModel):
    """Strategic content planning."""

    target_keywords: list[str]
    content_pillars: list[str]
    engagement_hooks: list[str]
    call_to_action: str


class PlanningResults(BaseModel):
    """Complete planning phase output."""

    content_outline: ContentOutline
    content_strategy: ContentStrategy


class DraftContent(BaseModel):
    """Initial content draft."""

    title: str
    content: str
    word_count: int
    readability_score: float


class ReviewedContent(BaseModel):
    """Content after review and editing."""

    final_title: str
    final_content: str
    final_word_count: int
    quality_score: float
    revision_notes: list[str]


class WritingResults(BaseModel):
    """Complete writing phase output."""

    draft_content: DraftContent
    reviewed_content: ReviewedContent


class FormattedContent(BaseModel):
    """Content formatted for specific platform."""

    platform: str
    formatted_content: str
    metadata: dict
    seo_optimized: bool


class PublishingAssets(BaseModel):
    """Supporting assets for publishing."""

    social_media_posts: list[str]
    email_snippet: str
    meta_description: str
    featured_image_prompt: str


class PublishingResults(BaseModel):
    """Complete publishing phase output."""

    formatted_content: FormattedContent
    publishing_assets: PublishingAssets


class ContentCreationResults(BaseModel):
    """Final output of the entire content creation pipeline."""

    research_phase: ResearchResults
    planning_phase: PlanningResults
    writing_phase: WritingResults
    publishing_phase: PublishingResults


# =============================================
# Mock Functions (Replace with Real AI/APIs)
# =============================================


def gather_research(request: ContentRequest) -> ResearchData:
    """Mock research gathering function."""
    return ResearchData(
        facts=[
            f"Key fact about {request.topic} for {request.target_audience}",
            f"Important trend in {request.topic} industry",
            f"Statistical data relevant to {request.topic}",
        ],
        sources=[
            f"https://authority-source-{request.topic.replace(' ', '-')}.com",
            f"https://research-journal-{request.topic.replace(' ', '-')}.org",
            f"https://industry-report-{request.topic.replace(' ', '-')}.net",
        ],
        key_insights=[
            f"Primary insight about {request.topic}",
            f"Secondary insight for {request.target_audience}",
        ],
        credibility_score=0.85,
    )


def validate_research(research_data: ResearchData) -> ValidatedResearch:
    """Mock research validation function."""
    is_high_quality = research_data.credibility_score > RESEARCH_QUALITY_THRESHOLD
    return ValidatedResearch(
        verified_facts=research_data.facts[:2],  # Keep top facts
        trusted_sources=research_data.sources[:2],  # Keep top sources
        main_insights=research_data.key_insights,
        research_quality="high" if is_high_quality else "medium",
    )


def create_outline(research_results: ResearchResults) -> ContentOutline:
    """Mock content outline creation."""
    topic_words = research_results.validated_research.verified_facts[0].split()[:3]
    return ContentOutline(
        title=f"The Complete Guide to {' '.join(topic_words)}",
        introduction=f"Introduction covering {' '.join(topic_words)}",
        main_sections=[
            f"Understanding {topic_words[0]}",
            f"Best Practices for {topic_words[1]}",
            f"Advanced {topic_words[2]} Techniques",
            "Implementation and Results",
        ],
        conclusion="Summary and next steps",
        estimated_word_count=1200,
    )


def develop_strategy(research_results: ResearchResults) -> ContentStrategy:
    """Mock content strategy development."""
    return ContentStrategy(
        target_keywords=["primary keyword", "secondary keyword", "long tail keyword"],
        content_pillars=["Education", "Problem-solving", "Industry insights"],
        engagement_hooks=["Question hook", "Statistic hook", "Story hook"],
        call_to_action="Learn more about our solutions",
    )


def write_draft(planning_results: PlanningResults) -> DraftContent:
    """Mock content writing function."""
    intro = planning_results.content_outline.introduction
    return DraftContent(
        title=planning_results.content_outline.title,
        content=f"Draft content based on outline: {intro}...",
        word_count=1150,
        readability_score=0.75,
    )


def review_content(draft_content: DraftContent) -> ReviewedContent:
    """Mock content review and editing."""
    return ReviewedContent(
        final_title=f"Revised: {draft_content.title}",
        final_content=f"Edited version: {draft_content.content[:100]}... [enhanced]",
        final_word_count=draft_content.word_count + 50,
        quality_score=0.9,
        revision_notes=["Improved clarity", "Enhanced SEO", "Added examples"],
    )


def format_content(writing_results: WritingResults) -> FormattedContent:
    """Mock content formatting for publishing."""
    final_content = writing_results.reviewed_content.final_content
    return FormattedContent(
        platform="blog",
        formatted_content=f"HTML formatted: {final_content}",
        metadata={
            "author": "AI Content System",
            "publish_date": datetime.now().isoformat(),
            "category": "Technology",
        },
        seo_optimized=True,
    )


def create_assets(writing_results: WritingResults) -> PublishingAssets:
    """Mock publishing assets creation."""
    final_title = writing_results.reviewed_content.final_title
    return PublishingAssets(
        social_media_posts=[
            f"Check out our latest post: {final_title[:50]}...",
            f"New insights on {final_title.split()[-1]}!",
        ],
        email_snippet=f"Latest from our blog: {final_title}",
        meta_description=f"Comprehensive guide covering {final_title.lower()}",
        featured_image_prompt=f"Professional image representing {final_title}",
    )


# =======================================
# Sub-flow Definitions (Level 1 Flows)
# =======================================


async def create_research_flow() -> Flow[ContentRequest, ResearchResults]:
    """Create a research sub-flow that gathers and validates information."""
    # Research flow: ContentRequest → ResearchResults
    research_flow = Flow(input_type=ContentRequest, output_type=ResearchResults)

    # Step 1: Gather raw research data
    research_node = ToolNode[ContentRequest, ResearchData](
        tool_func=gather_research, name="research_data"
    )

    # Step 2: Validate research for accuracy
    validation_node = ToolNode[ResearchData, ValidatedResearch](
        tool_func=validate_research,
        input=research_node.output,
        name="validated_research",
    )

    research_flow.add_nodes(research_node, validation_node)
    return research_flow


async def create_planning_flow() -> Flow[ResearchResults, PlanningResults]:
    """Create a planning sub-flow that develops content strategy."""
    # Planning flow: ResearchResults → PlanningResults
    planning_flow = Flow(input_type=ResearchResults, output_type=PlanningResults)

    # Step 1: Create content outline
    outline_node = ToolNode[ResearchResults, ContentOutline](
        tool_func=create_outline, name="content_outline"
    )

    # Step 2: Develop content strategy
    strategy_node = ToolNode[ResearchResults, ContentStrategy](
        tool_func=develop_strategy, name="content_strategy"
    )

    planning_flow.add_nodes(outline_node, strategy_node)
    return planning_flow


async def create_writing_flow() -> Flow[PlanningResults, WritingResults]:
    """Create a writing sub-flow with draft and review cycles."""
    # Writing flow: PlanningResults → WritingResults
    writing_flow = Flow(input_type=PlanningResults, output_type=WritingResults)

    # Step 1: Write initial draft
    draft_node = ToolNode[PlanningResults, DraftContent](
        tool_func=write_draft, name="draft_content"
    )

    # Step 2: Review and edit content
    review_node = ToolNode[DraftContent, ReviewedContent](
        tool_func=review_content, input=draft_node.output, name="reviewed_content"
    )

    writing_flow.add_nodes(draft_node, review_node)
    return writing_flow


async def create_publishing_flow() -> Flow[WritingResults, PublishingResults]:
    """Create a publishing sub-flow for formatting and asset creation."""
    # Publishing flow: WritingResults → PublishingResults
    publishing_flow = Flow(input_type=WritingResults, output_type=PublishingResults)

    # Step 1: Format content for platform
    formatting_node = ToolNode[WritingResults, FormattedContent](
        tool_func=format_content, name="formatted_content"
    )

    # Step 2: Create publishing assets
    assets_node = ToolNode[WritingResults, PublishingAssets](
        tool_func=create_assets, name="publishing_assets"
    )

    publishing_flow.add_nodes(formatting_node, assets_node)
    return publishing_flow


# =======================================
# Master Pipeline (Level 2 Flow)
# =======================================


async def create_content_pipeline() -> Flow[ContentRequest, ContentCreationResults]:
    """Create the master content creation pipeline using sub-flows."""
    # Master pipeline: ContentRequest → ContentCreationResults
    master_pipeline = Flow(
        input_type=ContentRequest, output_type=ContentCreationResults
    )

    # Create all sub-flows
    research_flow = await create_research_flow()
    planning_flow = await create_planning_flow()
    writing_flow = await create_writing_flow()
    publishing_flow = await create_publishing_flow()

    # Wrap sub-flows as FlowNodes
    research_flow_node = FlowNode[ContentRequest, ResearchResults](
        flow=research_flow, name="research_phase"
    )

    planning_flow_node = FlowNode[ResearchResults, PlanningResults](
        flow=planning_flow, input=research_flow_node.output, name="planning_phase"
    )

    writing_flow_node = FlowNode[PlanningResults, WritingResults](
        flow=writing_flow, input=planning_flow_node.output, name="writing_phase"
    )

    publishing_flow_node = FlowNode[WritingResults, PublishingResults](
        flow=publishing_flow, input=writing_flow_node.output, name="publishing_phase"
    )

    # Add all sub-flow nodes to master pipeline
    master_pipeline.add_nodes(
        research_flow_node, planning_flow_node, writing_flow_node, publishing_flow_node
    )

    return master_pipeline


# ======================================
# Example Execution and Demonstrations
# ======================================


async def demonstrate_individual_flows():
    """Demonstrate individual sub-flows working in isolation."""
    print("=== Individual Sub-flow Demonstrations ===\n")

    # Test research flow
    print("1. Testing Research Flow")
    research_flow = await create_research_flow()
    request = ContentRequest(
        topic="artificial intelligence",
        target_audience="business executives",
        content_type="blog_post",
    )
    research_result = await research_flow.run(request)
    print(f"   Research Quality: {research_result.validated_research.research_quality}")
    print(
        f"   Verified Facts: {len(research_result.validated_research.verified_facts)}"
    )
    print()

    # Test planning flow
    print("2. Testing Planning Flow")
    planning_flow = await create_planning_flow()
    planning_result = await planning_flow.run(research_result)
    print(f"   Content Title: {planning_result.content_outline.title}")
    print(f"   Sections: {len(planning_result.content_outline.main_sections)}")
    print()

    # Test writing flow
    print("3. Testing Writing Flow")
    writing_flow = await create_writing_flow()
    writing_result = await writing_flow.run(planning_result)
    print(f"   Final Word Count: {writing_result.reviewed_content.final_word_count}")
    print(f"   Quality Score: {writing_result.reviewed_content.quality_score}")
    print()

    # Test publishing flow
    print("4. Testing Publishing Flow")
    publishing_flow = await create_publishing_flow()
    publishing_result = await publishing_flow.run(writing_result)
    print(f"   Platform: {publishing_result.formatted_content.platform}")
    print(f"   SEO Optimized: {publishing_result.formatted_content.seo_optimized}")
    print()


async def demonstrate_master_pipeline():
    """Demonstrate the complete hierarchical pipeline."""
    print("=== Master Content Creation Pipeline ===\n")

    # Create the master pipeline
    master_pipeline = await create_content_pipeline()
    print(f"Pipeline Structure: {master_pipeline}")
    print(f"Execution Order: {master_pipeline.get_execution_order()}")
    print()

    # Execute complete pipeline
    content_request = ContentRequest(
        topic="sustainable technology trends",
        target_audience="technology professionals",
        content_type="article",
        tone="professional",
        word_count_target=1500,
    )

    print("Executing complete content creation pipeline...")
    results = await master_pipeline.run(content_request)

    # Display results from each phase
    print("\n📊 Pipeline Results Summary:")
    research_quality = results.research_phase.validated_research.research_quality
    print(f"✅ Research Phase: {research_quality} quality")
    print(f"✅ Planning Phase: {results.planning_phase.content_outline.title}")
    word_count = results.writing_phase.reviewed_content.final_word_count
    print(f"✅ Writing Phase: {word_count} words")
    platform = results.publishing_phase.formatted_content.platform
    print(f"✅ Publishing Phase: {platform} ready")

    print("\n📈 Content Metrics:")
    sources_count = len(results.research_phase.validated_research.trusted_sources)
    print(f"   • Research Sources: {sources_count}")
    sections_count = len(results.planning_phase.content_outline.main_sections)
    print(f"   • Content Sections: {sections_count}")
    quality_score = results.writing_phase.reviewed_content.quality_score
    print(f"   • Quality Score: {quality_score}")
    posts_count = len(results.publishing_phase.publishing_assets.social_media_posts)
    print(f"   • Social Posts Generated: {posts_count}")


async def demonstrate_pipeline_reusability():
    """Demonstrate reusing sub-flows for different content types."""
    print("\n=== Pipeline Reusability Demonstration ===\n")

    # Create different content requests
    blog_request = ContentRequest(
        topic="cloud computing basics",
        target_audience="beginners",
        content_type="blog_post",
        tone="casual",
    )

    technical_request = ContentRequest(
        topic="microservices architecture patterns",
        target_audience="software architects",
        content_type="technical_guide",
        tone="technical",
    )

    # Reuse the same pipeline for different content types
    pipeline = await create_content_pipeline()

    print("Creating blog content...")
    blog_results = await pipeline.run(blog_request)

    print("Creating technical guide...")
    technical_results = await pipeline.run(technical_request)

    print("\n📝 Content Comparison:")
    print(f"Blog Post: '{blog_results.planning_phase.content_outline.title}'")
    blog_word_count = blog_results.writing_phase.reviewed_content.final_word_count
    print(f"   - Word Count: {blog_word_count}")
    blog_quality = blog_results.writing_phase.reviewed_content.quality_score
    print(f"   - Quality: {blog_quality}")

    tech_title = technical_results.planning_phase.content_outline.title
    print(f"Technical Guide: '{tech_title}'")
    tech_word_count = technical_results.writing_phase.reviewed_content.final_word_count
    print(f"   - Word Count: {tech_word_count}")
    tech_quality = technical_results.writing_phase.reviewed_content.quality_score
    print(f"   - Quality: {tech_quality}")


async def main():
    """Run all hierarchical flow demonstrations."""
    print("🏗️  Hierarchical Flow Architecture Example")
    print("=" * 50)
    print("This example demonstrates a sophisticated content creation pipeline")
    print("built using hierarchical flows with FlowNode composition.\n")

    print("Architecture Overview:")
    print("Level 1: Individual sub-flows (Research, Planning, Writing, Publishing)")
    print("Level 2: Master pipeline orchestrating all sub-flows")
    print("Level 3: Reusable components across different content types\n")

    # Run all demonstrations
    await demonstrate_individual_flows()
    await demonstrate_master_pipeline()
    await demonstrate_pipeline_reusability()

    print("\n✨ Key Benefits Demonstrated:")
    print("• Hierarchical Composition: Complex workflows from simple components")
    print("• Reusability: Same sub-flows work across different content types")
    print("• Type Safety: Full type checking across all flow boundaries")
    print("• Testability: Each sub-flow can be tested independently")
    print("• Maintainability: Clear separation of concerns and responsibilities")
    print("• Scalability: Easy to add new phases or modify existing ones")


if __name__ == "__main__":
    asyncio.run(main())
