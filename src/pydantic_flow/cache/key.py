"""Cache key builders for different node types.

This module provides deterministic key generation for LLM, embedding,
and other cacheable operations.
"""

from __future__ import annotations

import inspect
from typing import Any

from pydantic import BaseModel

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.cache.hashing import hash_bytes
from pydantic_flow.cache.hashing import hash_json


def build_cache_key(
    node_name: str,
    inputs: dict[str, Any],
    policy: CachePolicy,
    context: dict[str, Any] | None = None,
) -> str:
    """Build a cache key for a node execution.

    Args:
        node_name: Name of the node.
        inputs: Input data for the node.
        policy: Cache policy with scope and strategy.
        context: Optional execution context.

    Returns:
        Cache key string in format: pf:{scope}:{hash}

    """
    scope_prefix = policy.scope.prefix()

    key_material = {
        "node": node_name,
        "inputs": _serialize_inputs(inputs),
    }

    if policy.node_version:
        key_material["version"] = policy.node_version

    if policy.extra_key_material:
        key_material["extra"] = policy.extra_key_material

    if context:
        key_material["context"] = context

    key_hash = hash_json(key_material)
    return f"pf:{scope_prefix}:{key_hash}"


def build_llm_cache_key(  # noqa: PLR0913
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_mode: str | None = None,
    policy: CachePolicy | None = None,
    environment: str | None = None,
) -> str:
    """Build a cache key for LLM completion.

    Includes all parameters that affect the output:
    - Provider and model
    - Messages and system prompt
    - Sampling parameters (temp, top_p, seed)
    - Tool schemas and mode
    - Policy version and extra material
    - Environment label

    Args:
        provider: LLM provider name.
        model: Model identifier.
        messages: List of message dicts.
        system_prompt: System prompt text.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        seed: Random seed for reproducibility.
        tools: Tool schemas.
        tool_mode: Tool invocation mode.
        policy: Cache policy with scope and version.
        environment: Environment label (dev/stage/prod).

    Returns:
        Cache key string.

    """
    policy = policy or CachePolicy()
    scope_prefix = policy.scope.prefix()

    key_material: dict[str, Any] = {
        "type": "llm_completion",
        "provider": provider,
        "model": model,
        "messages": messages,
    }

    if system_prompt:
        key_material["system_prompt"] = system_prompt

    if temperature is not None:
        key_material["temperature"] = temperature

    if top_p is not None:
        key_material["top_p"] = top_p

    if seed is not None:
        key_material["seed"] = seed

    if tools:
        tool_hashes = [hash_json(tool) for tool in tools]
        key_material["tools"] = sorted(tool_hashes)

    if tool_mode:
        key_material["tool_mode"] = tool_mode

    if policy.node_version:
        key_material["version"] = policy.node_version

    if policy.extra_key_material:
        key_material["extra"] = policy.extra_key_material

    if environment:
        key_material["environment"] = environment

    key_hash = hash_json(key_material)
    return f"pf:{scope_prefix}:{key_hash}"


def build_embedding_cache_key(  # noqa: PLR0913
    provider: str,
    model: str,
    text: str,
    dimension: int | None = None,
    normalize: bool = False,
    policy: CachePolicy | None = None,
    chunking_version: str | None = None,
) -> str:
    """Build a cache key for embedding generation.

    Args:
        provider: Embedding provider name.
        model: Model identifier.
        text: Input text to embed.
        dimension: Output dimension.
        normalize: Whether to normalize vectors.
        policy: Cache policy with scope and version.
        chunking_version: Version tag for text preprocessing.

    Returns:
        Cache key string.

    """
    policy = policy or CachePolicy()
    scope_prefix = policy.scope.prefix()

    text_hash = hash_bytes(text.encode("utf-8"))

    key_material: dict[str, Any] = {
        "type": "embedding",
        "provider": provider,
        "model": model,
        "text_hash": text_hash,
    }

    if dimension:
        key_material["dimension"] = dimension

    if normalize:
        key_material["normalize"] = normalize

    if chunking_version:
        key_material["chunking_version"] = chunking_version

    if policy.node_version:
        key_material["version"] = policy.node_version

    if policy.extra_key_material:
        key_material["extra"] = policy.extra_key_material

    key_hash = hash_json(key_material)
    return f"pf:{scope_prefix}:{key_hash}"


def compute_node_code_fingerprint(node: Any) -> str | None:
    """Compute a fingerprint of node implementation code.

    This can be used as part of the cache key to invalidate
    caches when node logic changes.

    Args:
        node: The node instance.

    Returns:
        Hash of node source code, or None if source unavailable.

    """
    try:
        source = inspect.getsource(node.__class__)
        return hash_bytes(source.encode("utf-8"))
    except TypeError, OSError:
        return None


def _serialize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Serialize input data for hashing.

    Args:
        inputs: Raw input dictionary.

    Returns:
        Serialized inputs suitable for JSON hashing.

    """
    serialized = {}
    for key, value in inputs.items():
        if isinstance(value, BaseModel):
            serialized[key] = value.model_dump()
        elif hasattr(value, "__dict__"):
            serialized[key] = {
                k: v for k, v in value.__dict__.items() if not k.startswith("_")
            }
        else:
            serialized[key] = value
    return serialized
