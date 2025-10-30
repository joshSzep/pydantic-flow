"""Markdown technical docs example using heading splitter and MMR.

This example shows how to split technical documentation by headings
and use MMR to get diverse results across different sections.
"""

from pydantic_flow.rag import Document
from pydantic_flow.rag import LexicalReranker
from pydantic_flow.rag import MarkdownHeadingSplitter
from pydantic_flow.rag import Metadata
from pydantic_flow.rag import SplitConfig
from pydantic_flow.rag import mmr_select


def main() -> None:
    """Run markdown splitting demo."""
    print("=" * 80)
    print("Markdown Technical Documentation RAG Demo")
    print("=" * 80)

    markdown_doc = """# API Authentication

Authentication is required for all API endpoints.
Use an API key in the Authorization header.

## Getting an API Key

Visit your account dashboard to generate an API key.
Each key can be scoped to specific permissions.

### Key Rotation

Rotate your keys regularly for security.
Old keys remain valid for 24 hours after rotation.

## Making Authenticated Requests

Include the header: `Authorization: Bearer YOUR_KEY`
All requests without authentication will return 401.

# Rate Limiting

API requests are rate limited per account.
Free tier allows 100 requests per hour.

## Rate Limit Headers

Response headers include rate limit information.
Check `X-RateLimit-Remaining` for remaining requests.

### Exceeding Rate Limits

Requests beyond the limit return 429 status.
Wait for the reset time indicated in headers.

# Error Handling

APIs return standard HTTP status codes.
Error responses include detailed messages.

## Common Error Codes

- 400: Bad Request - Invalid parameters
- 401: Unauthorized - Missing or invalid API key
- 403: Forbidden - Insufficient permissions
- 404: Not Found - Resource doesn't exist
- 429: Too Many Requests - Rate limit exceeded
- 500: Internal Server Error - Server-side issue

## Error Response Format

Errors return JSON with error code and message.
Use these for debugging and user feedback.
"""

    doc = Document(
        id="api_docs",
        content=markdown_doc,
        metadata=Metadata(source="docs", extra={"type": "api_reference"}),
    )

    print("\n1. MARKDOWN SPLITTING: Splitting by headings")
    print("-" * 80)

    splitter = MarkdownHeadingSplitter()
    config = SplitConfig(
        splitter_type="markdown",
        max_chars=500,
        overlap=50,
        min_chunk_chars=20,
    )

    chunks = splitter.split(doc.content, doc.id, config)

    print(f"Split into {len(chunks)} chunks")
    print("\nChunk examples with heading paths:")
    for i, chunk in enumerate(chunks[:3], 1):
        path = " > ".join(chunk.metadata.heading_path)
        preview = chunk.text.strip()[:60]
        print(f"\n{i}. Path: {path}")
        print(f"   Text: {preview}...")

    print("\n2. QUERYING: Finding relevant sections")
    print("-" * 80)

    query = "authentication API key header"
    print(f"Query: '{query}'")

    reranker = LexicalReranker()
    scored = reranker.score(query, chunks)

    print("\nTop 3 results:")
    for i, scored_chunk in enumerate(scored[:3], 1):
        path = " > ".join(scored_chunk.chunk.metadata.heading_path)
        preview = scored_chunk.chunk.text.strip()[:60]
        print(f"\n{i}. Score: {scored_chunk.score:.3f}")
        print(f"   Path: {path}")
        print(f"   Text: {preview}...")

    print("\n3. DIVERSIFICATION: Getting diverse sections with MMR")
    print("-" * 80)

    final = mmr_select(scored, k=4, lambda_mult=0.4)

    print(f"Selected {len(final)} diverse sections:")
    for i, scored_chunk in enumerate(final, 1):
        path = " > ".join(scored_chunk.chunk.metadata.heading_path)
        preview = scored_chunk.chunk.text.strip()[:60]
        print(f"\n{i}. Score: {scored_chunk.score:.3f}")
        print(f"   Path: {path}")
        print(f"   Text: {preview}...")

    print("\n" + "=" * 80)
    print("Notice how results span different documentation sections!")
    print("=" * 80)


if __name__ == "__main__":
    main()
