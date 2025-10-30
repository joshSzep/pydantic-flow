"""RAG (Retrieval-Augmented Generation) adapters for pydantic-flow.

This package provides type-safe, streaming-native RAG components including:
- Document models and metadata
- Embedding providers (OpenAI, Cohere, HuggingFace, Ollama)
- Vector stores (HNSW, PGVector)
- Retrievers for semantic search
- Loaders for various data sources
- Nodes for integrating RAG into flows
- Document splitters (Token, Sentence, Markdown)
- Rerankers (Lexical, Cohere)
- Diversification (MMR, source-aware)
"""

from pydantic_flow.rag.diversify import DiversifyConfig
from pydantic_flow.rag.diversify import diversify_by_source
from pydantic_flow.rag.diversify import mmr_select
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
from pydantic_flow.rag.nodes.enhanced_retriever import EnhancedRetrieverNode
from pydantic_flow.rag.nodes.retriever import VectorRetrieverNode
from pydantic_flow.rag.rerankers import CohereReranker
from pydantic_flow.rag.rerankers import LexicalReranker
from pydantic_flow.rag.rerankers import RerankConfig
from pydantic_flow.rag.rerankers import Reranker
from pydantic_flow.rag.rerankers import ScoredChunk
from pydantic_flow.rag.retrievers.base import Retriever
from pydantic_flow.rag.retrievers.vector import VectorRetriever
from pydantic_flow.rag.splitters import ChunkMetadata
from pydantic_flow.rag.splitters import DocumentChunk
from pydantic_flow.rag.splitters import MarkdownHeadingSplitter
from pydantic_flow.rag.splitters import SentenceSplitter
from pydantic_flow.rag.splitters import SimpleTokenCounter
from pydantic_flow.rag.splitters import SplitConfig
from pydantic_flow.rag.splitters import Splitter
from pydantic_flow.rag.splitters import TokenCounter
from pydantic_flow.rag.splitters import TokenSplitter
from pydantic_flow.rag.vectors.base import SearchResult
from pydantic_flow.rag.vectors.base import VectorStore
from pydantic_flow.rag.vectors.hnsw import HNSWMemoryStore
from pydantic_flow.rag.vectors.pgvector import PostgresPGVectorStore

__all__ = [
    "ChunkMetadata",
    "CohereEmbeddings",
    "CohereReranker",
    "DiversifyConfig",
    "Document",
    "DocumentChunk",
    "EmbeddingNode",
    "EmbeddingProvider",
    "EnhancedRetrieverNode",
    "FSLoader",
    "HNSWMemoryStore",
    "HuggingFaceEmbeddings",
    "LexicalReranker",
    "Loader",
    "MarkdownHeadingSplitter",
    "Metadata",
    "OllamaEmbeddings",
    "OpenAIEmbeddings",
    "PostgresPGVectorStore",
    "RerankConfig",
    "Reranker",
    "Retriever",
    "ScoredChunk",
    "SearchResult",
    "SentenceSplitter",
    "SimpleTokenCounter",
    "SplitConfig",
    "Splitter",
    "TokenCounter",
    "TokenSplitter",
    "VectorRetriever",
    "VectorRetrieverNode",
    "VectorStore",
    "WebLoader",
    "diversify_by_source",
    "mmr_select",
]
