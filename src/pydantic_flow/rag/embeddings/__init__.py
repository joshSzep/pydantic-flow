"""Embedding providers."""

from pydantic_flow.rag.embeddings.base import EmbeddingProvider
from pydantic_flow.rag.embeddings.cohere import CohereEmbeddings
from pydantic_flow.rag.embeddings.huggingface import HuggingFaceEmbeddings
from pydantic_flow.rag.embeddings.ollama import OllamaEmbeddings
from pydantic_flow.rag.embeddings.openai import OpenAIEmbeddings

__all__ = [
    "CohereEmbeddings",
    "EmbeddingProvider",
    "HuggingFaceEmbeddings",
    "OllamaEmbeddings",
    "OpenAIEmbeddings",
]
