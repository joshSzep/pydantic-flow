"""Tests for cache key generation."""

from __future__ import annotations

from pydantic import BaseModel

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.cache.base import CacheScope
from pydantic_flow.cache.key import build_cache_key
from pydantic_flow.cache.key import build_embedding_cache_key
from pydantic_flow.cache.key import build_llm_cache_key
from pydantic_flow.cache.key import compute_node_code_fingerprint


class SampleInput(BaseModel):
    """Sample input model for testing."""

    value: int
    name: str


class SampleNode:
    """Sample node for testing fingerprint."""

    def execute(self) -> str:
        """Execute node."""
        return "result"


def test_build_cache_key_basic() -> None:
    """Test basic cache key generation."""
    policy = CachePolicy(scope=CacheScope.GLOBAL())
    key = build_cache_key("test_node", {"input": "value"}, policy)

    assert key.startswith("pf:global:")
    assert len(key.split(":")) == 3


def test_build_cache_key_with_version() -> None:
    """Test cache key with node version."""
    policy = CachePolicy(scope=CacheScope.NAMESPACE("flow"), node_version="v2")
    key = build_cache_key("test_node", {"input": "value"}, policy)

    assert key.startswith("pf:ns:flow:")


def test_build_cache_key_with_extra_material() -> None:
    """Test cache key with extra key material."""
    policy = CachePolicy(
        scope=CacheScope.NAMESPACE("run"),
        extra_key_material={"custom": "data"},
    )
    key = build_cache_key("test_node", {"input": "value"}, policy)

    assert key.startswith("pf:ns:run:")


def test_build_cache_key_with_context() -> None:
    """Test cache key with execution context."""
    policy = CachePolicy(scope=CacheScope.GLOBAL())
    key = build_cache_key(
        "test_node",
        {"input": "value"},
        policy,
        context={"flow_id": "test"},
    )

    assert key.startswith("pf:global:")


def test_build_cache_key_with_pydantic_input() -> None:
    """Test cache key with Pydantic model input."""
    policy = CachePolicy(scope=CacheScope.GLOBAL())
    input_data = SampleInput(value=42, name="test")

    key = build_cache_key("test_node", {"input": input_data}, policy)

    assert key.startswith("pf:global:")


def test_build_cache_key_deterministic() -> None:
    """Test that same inputs produce same key."""
    policy = CachePolicy(scope=CacheScope.GLOBAL())

    key1 = build_cache_key("test_node", {"a": 1, "b": 2}, policy)
    key2 = build_cache_key("test_node", {"a": 1, "b": 2}, policy)

    assert key1 == key2


def test_build_cache_key_different_inputs() -> None:
    """Test that different inputs produce different keys."""
    policy = CachePolicy(scope=CacheScope.GLOBAL())

    key1 = build_cache_key("test_node", {"a": 1}, policy)
    key2 = build_cache_key("test_node", {"a": 2}, policy)

    assert key1 != key2


def test_build_llm_cache_key_basic() -> None:
    """Test basic LLM cache key generation."""
    key = build_llm_cache_key(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert key.startswith("pf:global:")


def test_build_llm_cache_key_with_system_prompt() -> None:
    """Test LLM cache key with system prompt."""
    key = build_llm_cache_key(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        system_prompt="You are a helpful assistant",
    )

    assert key.startswith("pf:global:")


def test_build_llm_cache_key_with_temperature() -> None:
    """Test LLM cache key with temperature."""
    key = build_llm_cache_key(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
    )

    assert key.startswith("pf:global:")


def test_build_llm_cache_key_with_top_p() -> None:
    """Test LLM cache key with top_p."""
    key = build_llm_cache_key(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        top_p=0.9,
    )

    assert key.startswith("pf:global:")


def test_build_llm_cache_key_with_seed() -> None:
    """Test LLM cache key with seed."""
    key = build_llm_cache_key(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        seed=42,
    )

    assert key.startswith("pf:global:")


def test_build_llm_cache_key_with_tools() -> None:
    """Test LLM cache key with tools."""
    tools = [
        {"name": "get_weather", "description": "Get weather"},
        {"name": "get_time", "description": "Get time"},
    ]

    key = build_llm_cache_key(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        tools=tools,
    )

    assert key.startswith("pf:global:")


def test_build_llm_cache_key_with_tool_mode() -> None:
    """Test LLM cache key with tool mode."""
    key = build_llm_cache_key(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        tool_mode="auto",
    )

    assert key.startswith("pf:global:")


def test_build_llm_cache_key_with_policy() -> None:
    """Test LLM cache key with custom policy."""
    policy = CachePolicy(scope=CacheScope.NAMESPACE("flow"), node_version="v1")

    key = build_llm_cache_key(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        policy=policy,
    )

    assert key.startswith("pf:ns:flow:")


def test_build_llm_cache_key_with_environment() -> None:
    """Test LLM cache key with environment."""
    key = build_llm_cache_key(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        environment="production",
    )

    assert key.startswith("pf:global:")


def test_build_llm_cache_key_deterministic() -> None:
    """Test that same LLM params produce same key."""
    key1 = build_llm_cache_key(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
    )
    key2 = build_llm_cache_key(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
    )

    assert key1 == key2


def test_build_llm_cache_key_different_messages() -> None:
    """Test that different messages produce different keys."""
    key1 = build_llm_cache_key(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
    )
    key2 = build_llm_cache_key(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "Goodbye"}],
    )

    assert key1 != key2


def test_build_embedding_cache_key_basic() -> None:
    """Test basic embedding cache key generation."""
    key = build_embedding_cache_key(
        provider="openai",
        model="text-embedding-3-small",
        text="Hello world",
    )

    assert key.startswith("pf:global:")


def test_build_embedding_cache_key_with_dimension() -> None:
    """Test embedding cache key with dimension."""
    key = build_embedding_cache_key(
        provider="openai",
        model="text-embedding-3-small",
        text="Hello world",
        dimension=512,
    )

    assert key.startswith("pf:global:")


def test_build_embedding_cache_key_with_normalize() -> None:
    """Test embedding cache key with normalize."""
    key = build_embedding_cache_key(
        provider="openai",
        model="text-embedding-3-small",
        text="Hello world",
        normalize=True,
    )

    assert key.startswith("pf:global:")


def test_build_embedding_cache_key_with_chunking_version() -> None:
    """Test embedding cache key with chunking version."""
    key = build_embedding_cache_key(
        provider="openai",
        model="text-embedding-3-small",
        text="Hello world",
        chunking_version="v2",
    )

    assert key.startswith("pf:global:")


def test_build_embedding_cache_key_with_policy() -> None:
    """Test embedding cache key with custom policy."""
    policy = CachePolicy(scope=CacheScope.NAMESPACE("run"), node_version="v1")

    key = build_embedding_cache_key(
        provider="openai",
        model="text-embedding-3-small",
        text="Hello world",
        policy=policy,
    )

    assert key.startswith("pf:ns:run:")


def test_build_embedding_cache_key_deterministic() -> None:
    """Test that same text produces same key."""
    key1 = build_embedding_cache_key(
        provider="openai",
        model="text-embedding-3-small",
        text="Hello world",
    )
    key2 = build_embedding_cache_key(
        provider="openai",
        model="text-embedding-3-small",
        text="Hello world",
    )

    assert key1 == key2


def test_build_embedding_cache_key_different_text() -> None:
    """Test that different text produces different keys."""
    key1 = build_embedding_cache_key(
        provider="openai",
        model="text-embedding-3-small",
        text="Hello",
    )
    key2 = build_embedding_cache_key(
        provider="openai",
        model="text-embedding-3-small",
        text="Goodbye",
    )

    assert key1 != key2


def test_compute_node_code_fingerprint() -> None:
    """Test computing node code fingerprint."""
    node = SampleNode()
    fingerprint = compute_node_code_fingerprint(node)

    assert fingerprint is not None
    assert isinstance(fingerprint, str)
    assert len(fingerprint) > 0


def test_compute_node_code_fingerprint_deterministic() -> None:
    """Test that same node produces same fingerprint."""
    node1 = SampleNode()
    node2 = SampleNode()

    fp1 = compute_node_code_fingerprint(node1)
    fp2 = compute_node_code_fingerprint(node2)

    assert fp1 == fp2


def test_compute_node_code_fingerprint_builtin() -> None:
    """Test fingerprint for built-in types returns None."""
    fingerprint = compute_node_code_fingerprint(int)

    assert fingerprint is None
