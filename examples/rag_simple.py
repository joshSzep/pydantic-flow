"""RAG example: Ingest markdown and query with streaming.

This example demonstrates:
1. Loading markdown files from filesystem
2. Generating embeddings
3. Upserting to HNSW in-memory vector store
4. Querying with streaming retrieval
"""

import asyncio
from pathlib import Path
import tempfile

from pydantic_flow.rag.embeddings.base import EmbeddingProvider
from pydantic_flow.rag.loaders.fs import FSLoader
from pydantic_flow.rag.nodes.retriever import QueryInput
from pydantic_flow.rag.nodes.retriever import VectorRetrieverNode
from pydantic_flow.rag.retrievers.vector import VectorRetriever
from pydantic_flow.rag.vectors.hnsw import HNSWMemoryStore
from pydantic_flow.streaming.events import RetrievalItem


class MockEmbeddings(EmbeddingProvider):
    """Mock embedding provider for demonstration."""

    def __init__(self, dim: int = 128):
        """Initialize mock embeddings."""
        self.dimension = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings based on text length."""
        return [[hash(text) % 100 / 100.0] * self.dimension for text in texts]

    def dim(self) -> int:
        """Return dimension."""
        return self.dimension


async def main():
    """Run the RAG example."""
    print("=== RAG Example: Markdown Ingestion and Query ===\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        (temp_path / "doc1.md").write_text(
            "# Introduction\n\nPydantic-flow is a type-safe AI agent framework."
        )
        (temp_path / "doc2.md").write_text(
            "# Features\n\nStreaming-native APIs with async first design."
        )
        (temp_path / "doc3.md").write_text(
            "# Architecture\n\nBuilt on pydantic-ai with batteries included."
        )

        print(f"1. Loading documents from {temp_path}")
        loader = FSLoader(path=temp_path, chunk_size=100, chunk_overlap=20)
        documents = await loader.load()
        print(f"   Loaded {len(documents)} document chunks\n")

        print("2. Initializing embedding provider and vector store")
        embeddings = MockEmbeddings(dim=128)
        store = HNSWMemoryStore(dim=128)
        print(f"   Using HNSW with dimension {embeddings.dim()}\n")

        print("3. Generating embeddings and upserting to vector store")
        items = []
        for doc in documents:
            text_embeddings = await embeddings.embed([doc.content])
            items.append((doc.id, text_embeddings[0], doc))

        await store.upsert(items)
        print(f"   Upserted {len(items)} documents\n")

        print("4. Creating retriever and node")
        retriever = VectorRetriever(
            embedding_provider=embeddings,
            vector_store=store,
            default_k=2,
        )

        node = VectorRetrieverNode(
            retriever=retriever,
            name="rag-retriever",
        )
        print("   VectorRetrieverNode created\n")

        print("5. Querying with streaming")
        query = QueryInput(query="What is pydantic-flow?", k=2)

        print(f"   Query: '{query.query}'")
        print("   Streaming results:\n")

        async for item in node.astream(query):
            if isinstance(item, RetrievalItem):
                print(f"   📄 Retrieved: {item.item_id}")
                print(f"      Content: {item.content[:80]}...")
                print()

        print("\n6. Using non-streaming run() method")
        result = await node.run(query)
        print(f"   Retrieved {len(result.documents)} documents")
        for i, doc_dict in enumerate(result.documents):
            print(f"   {i + 1}. {doc_dict['id']}: {doc_dict['content'][:60]}...")

    print("\n✅ Example complete!")


if __name__ == "__main__":
    asyncio.run(main())
