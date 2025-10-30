"""Comprehensive RAG pipeline example.

Demonstrates all splitters, rerankers, and diversification options.
"""

from pydantic_flow.rag import LexicalReranker
from pydantic_flow.rag import MarkdownHeadingSplitter
from pydantic_flow.rag import SentenceSplitter
from pydantic_flow.rag import SplitConfig
from pydantic_flow.rag import TokenSplitter
from pydantic_flow.rag import diversify_by_source
from pydantic_flow.rag import mmr_select
from pydantic_flow.rag.rerankers import ScoredChunk
from pydantic_flow.rag.splitters import ChunkMetadata
from pydantic_flow.rag.splitters import DocumentChunk


def demo_splitters() -> None:
    """Demonstrate all three splitter types."""
    print("\n" + "=" * 80)
    print("SPLITTER COMPARISON")
    print("=" * 80)

    text = "First sentence. Second sentence. Third sentence. " * 5

    print("\n1. Token Splitter (approximate token count)")
    print("-" * 80)
    token_splitter = TokenSplitter()
    token_config = SplitConfig(max_tokens=20, overlap=5, min_chunk_chars=10)
    token_chunks = token_splitter.split(text, "doc1", token_config)
    print(f"Created {len(token_chunks)} chunks")
    for i, chunk in enumerate(token_chunks[:2], 1):
        print(
            f"  Chunk {i}: {len(chunk.text)} chars, ~{chunk.metadata.token_count} tokens"
        )

    print("\n2. Sentence Splitter (preserves sentence boundaries)")
    print("-" * 80)
    sentence_splitter = SentenceSplitter()
    sentence_config = SplitConfig(max_chars=100, overlap=20, min_chunk_chars=10)
    sentence_chunks = sentence_splitter.split(text, "doc1", sentence_config)
    print(f"Created {len(sentence_chunks)} chunks")
    for i, chunk in enumerate(sentence_chunks[:2], 1):
        print(f"  Chunk {i}: {chunk.text[:50]}...")

    markdown_text = """# Title
Content under title.

## Section
Content under section."""

    print("\n3. Markdown Splitter (splits by headings)")
    print("-" * 80)
    md_splitter = MarkdownHeadingSplitter()
    md_config = SplitConfig(max_chars=200, overlap=0, min_chunk_chars=5)
    md_chunks = md_splitter.split(markdown_text, "doc1", md_config)
    print(f"Created {len(md_chunks)} chunks")
    for i, chunk in enumerate(md_chunks, 1):
        path = " > ".join(chunk.metadata.heading_path)
        print(f"  Chunk {i}: Path=[{path}]")


def demo_reranking() -> None:
    """Demonstrate lexical reranking."""
    print("\n" + "=" * 80)
    print("LEXICAL RERANKING")
    print("=" * 80)

    docs = [
        ("Reset your password by clicking forgot password", "doc1", 0),
        ("Create a new account on the signup page", "doc2", 0),
        ("Password reset email not received", "doc3", 0),
        ("Update account settings in your profile", "doc4", 0),
    ]

    chunks = [
        DocumentChunk(
            id=source_id,
            text=text,
            metadata=ChunkMetadata(source_id=source_id, chunk_index=idx),
        )
        for text, source_id, idx in docs
    ]

    query = "reset password email"
    print(f"\nQuery: '{query}'")
    print("\nOriginal order:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}. {chunk.text}")

    reranker = LexicalReranker()
    scored = reranker.score(query, chunks)

    print("\nAfter reranking:")
    for i, scored_chunk in enumerate(scored, 1):
        print(f"  {i}. Score: {scored_chunk.score:.3f} - {scored_chunk.chunk.text}")


def demo_mmr() -> None:
    """Demonstrate MMR diversification."""
    print("\n" + "=" * 80)
    print("MMR DIVERSIFICATION")
    print("=" * 80)

    chunks = [
        ScoredChunk(
            chunk=DocumentChunk(
                id=f"doc{i}",
                text=text,
                metadata=ChunkMetadata(source_id=f"doc{i}", chunk_index=0),
            ),
            score=1.0 - (i * 0.1),
            rank=i,
        )
        for i, text in enumerate([
            "Python is a programming language",
            "Python is a popular coding language",
            "Java is also a programming language",
            "Python has simple syntax",
            "JavaScript runs in browsers",
        ])
    ]

    print("\nOriginal order (by relevance score):")
    for i, sc in enumerate(chunks, 1):
        print(f"  {i}. Score: {sc.score:.2f} - {sc.chunk.text}")

    final = mmr_select(chunks, k=3, lambda_mult=0.3)

    print("\nAfter MMR (lambda=0.3, more diverse):")
    for i, sc in enumerate(final, 1):
        print(f"  {i}. Score: {sc.score:.2f} - {sc.chunk.text}")


def demo_source_diversification() -> None:
    """Demonstrate source-aware diversification."""
    print("\n" + "=" * 80)
    print("SOURCE-AWARE DIVERSIFICATION")
    print("=" * 80)

    chunks = [
        ScoredChunk(
            chunk=DocumentChunk(
                id=f"chunk{i}",
                text=f"Content from source {source}",
                metadata=ChunkMetadata(source_id=source, chunk_index=i),
            ),
            score=1.0 - (i * 0.05),
            rank=i,
        )
        for i, source in enumerate([
            "doc1",
            "doc1",
            "doc1",
            "doc2",
            "doc2",
            "doc2",
            "doc3",
            "doc3",
        ])
    ]

    print("\nOriginal (3 from doc1, 3 from doc2, 2 from doc3):")
    for i, sc in enumerate(chunks, 1):
        print(f"  {i}. {sc.chunk.metadata.source_id} - Score: {sc.score:.2f}")

    final = diversify_by_source(chunks, k=6, max_per_source=2)

    print("\nAfter source diversification (max 2 per source):")
    for i, sc in enumerate(final, 1):
        print(f"  {i}. {sc.chunk.metadata.source_id} - Score: {sc.score:.2f}")


def main() -> None:
    """Run all RAG demos."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RAG PIPELINE DEMO")
    print("=" * 80)

    demo_splitters()
    demo_reranking()
    demo_mmr()
    demo_source_diversification()

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
