"""RAG (Retrieval-Augmented Generation) adapters for pydantic-flow.

This package provides type-safe, streaming-native RAG components including:
- Document models and metadata
- Embedding providers (OpenAI, Cohere, HuggingFace, Ollama)
- Vector stores (HNSW, PGVector)
- Retrievers for semantic search
- Loaders for various data sources
- Nodes for integrating RAG into flows
"""

from pydantic_flow.rag.docs.types import Document
from pydantic_flow.rag.docs.types import Metadata
from pydantic_flow.rag.embeddings.base import EmbeddingProvider
from pydantic_flow.rag.embeddings.cohere import CohereEmbeddings
from pydantic_flow.rag.embeddings.huggingface import HuggingFaceEmbeddings
from pydantic_flow.rag.embeddings.ollama import OllamaEmbeddings
from pydantic_flow.rag.embeddings.openai import OpenAIEmbeddings
from pydantic_flow.rag.loaders.base import Loader
from pydantic_flow.rag.loaders.fs import FSLoader
from pydantic_flow.rag.loaders.web import WebLoader
from pydantic_flow.rag.nodes.embedding import EmbeddingNode
from pydantic_flow.rag.nodes.retriever import VectorRetrieverNode
from pydantic_flow.rag.retrievers.base import Retriever
from pydantic_flow.rag.retrievers.vector import VectorRetriever
from pydantic_flow.rag.vectors.base import SearchResult
from pydantic_flow.rag.vectors.base import VectorStore
from pydantic_flow.rag.vectors.hnsw import HNSWMemoryStore
from pydantic_flow.rag.vectors.pgvector import PostgresPGVectorStore

__all__ = [
    "CohereEmbeddings",
    "Document",
    "EmbeddingNode",
    "EmbeddingProvider",
    "FSLoader",
    "HNSWMemoryStore",
    "HuggingFaceEmbeddings",
    "Loader",
    "Metadata",
    "OllamaEmbeddings",
    "OpenAIEmbeddings",
    "PostgresPGVectorStore",
    "Retriever",
    "SearchResult",
    "VectorRetriever",
    "VectorRetrieverNode",
    "VectorStore",
    "WebLoader",
]
