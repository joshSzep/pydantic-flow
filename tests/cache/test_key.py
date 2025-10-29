"""Tests for cache key builders."""

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.cache.base import CacheScope
from pydantic_flow.cache.key import build_cache_key
from pydantic_flow.cache.key import build_embedding_cache_key
from pydantic_flow.cache.key import build_llm_cache_key


def test_build_cache_key_deterministic() -> None:
    """Cache keys should be deterministic for same inputs."""
    policy = CachePolicy(scope=CacheScope.GLOBAL())

    key1 = build_cache_key(
        "test_node",
        {"input": "data"},
        policy,
    )
    key2 = build_cache_key(
        "test_node",
        {"input": "data"},
        policy,
    )

    assert key1 == key2
    assert key1.startswith("pf:global:")


def test_build_cache_key_different_inputs() -> None:
    """Different inputs should produce different keys."""
    policy = CachePolicy(scope=CacheScope.GLOBAL())

    key1 = build_cache_key("test_node", {"input": "data1"}, policy)
    key2 = build_cache_key("test_node", {"input": "data2"}, policy)

    assert key1 != key2


def test_build_cache_key_with_namespace() -> None:
    """Namespaced keys should include namespace prefix."""
    policy = CachePolicy(scope=CacheScope.NAMESPACE("production"))

    key = build_cache_key("test_node", {"input": "data"}, policy)

    assert key.startswith("pf:ns:production:")


def test_build_cache_key_with_version() -> None:
    """Node version should affect cache key."""
    policy1 = CachePolicy(node_version="v1")
    policy2 = CachePolicy(node_version="v2")

    key1 = build_cache_key("test_node", {"input": "data"}, policy1)
    key2 = build_cache_key("test_node", {"input": "data"}, policy2)

    assert key1 != key2


def test_build_llm_cache_key_basic() -> None:
    """LLM cache keys should be deterministic."""
    messages = [{"role": "user", "content": "hello"}]

    key1 = build_llm_cache_key("openai", "gpt-4", messages)
    key2 = build_llm_cache_key("openai", "gpt-4", messages)

    assert key1 == key2


def test_build_llm_cache_key_different_model() -> None:
    """Different models should produce different keys."""
    messages = [{"role": "user", "content": "hello"}]

    key1 = build_llm_cache_key("openai", "gpt-4", messages)
    key2 = build_llm_cache_key("openai", "gpt-3.5-turbo", messages)

    assert key1 != key2


def test_build_llm_cache_key_different_temperature() -> None:
    """Different temperatures should produce different keys."""
    messages = [{"role": "user", "content": "hello"}]

    key1 = build_llm_cache_key("openai", "gpt-4", messages, temperature=0.7)
    key2 = build_llm_cache_key("openai", "gpt-4", messages, temperature=0.9)

    assert key1 != key2


def test_build_llm_cache_key_with_seed() -> None:
    """Same seed should produce same key."""
    messages = [{"role": "user", "content": "hello"}]

    key1 = build_llm_cache_key("openai", "gpt-4", messages, seed=42)
    key2 = build_llm_cache_key("openai", "gpt-4", messages, seed=42)
    key3 = build_llm_cache_key("openai", "gpt-4", messages, seed=43)

    assert key1 == key2
    assert key1 != key3


def test_build_llm_cache_key_with_tools() -> None:
    """Tools should affect cache key."""
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"name": "calculator", "schema": {"type": "object"}}]

    key1 = build_llm_cache_key("openai", "gpt-4", messages, tools=tools)
    key2 = build_llm_cache_key("openai", "gpt-4", messages)

    assert key1 != key2


def test_build_embedding_cache_key_basic() -> None:
    """Embedding cache keys should be deterministic."""
    key1 = build_embedding_cache_key("openai", "text-embedding-3-small", "hello")
    key2 = build_embedding_cache_key("openai", "text-embedding-3-small", "hello")

    assert key1 == key2


def test_build_embedding_cache_key_different_text() -> None:
    """Different text should produce different keys."""
    key1 = build_embedding_cache_key("openai", "text-embedding-3-small", "hello")
    key2 = build_embedding_cache_key("openai", "text-embedding-3-small", "world")

    assert key1 != key2


def test_build_embedding_cache_key_with_dimension() -> None:
    """Dimension should affect cache key."""
    key1 = build_embedding_cache_key(
        "openai", "text-embedding-3-small", "hello", dimension=512
    )
    key2 = build_embedding_cache_key(
        "openai", "text-embedding-3-small", "hello", dimension=1024
    )

    assert key1 != key2


def test_build_embedding_cache_key_with_normalize() -> None:
    """Normalize flag should affect cache key."""
    key1 = build_embedding_cache_key(
        "openai", "text-embedding-3-small", "hello", normalize=True
    )
    key2 = build_embedding_cache_key(
        "openai", "text-embedding-3-small", "hello", normalize=False
    )

    assert key1 != key2
