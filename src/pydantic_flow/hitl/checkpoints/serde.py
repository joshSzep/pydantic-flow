"""Serialization and hashing utilities for checkpoints.

Provides canonical JSON serialization with sorted keys and content hashing
for checkpoint verification.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope


def serialize_checkpoint(envelope: CheckpointEnvelope) -> str:
    """Serialize checkpoint envelope to canonical JSON.

    Uses sorted keys and UTF-8 encoding for deterministic output.

    Args:
        envelope: The checkpoint envelope to serialize.

    Returns:
        JSON string with sorted keys.

    """
    data = envelope.model_dump(mode="json")
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


def deserialize_checkpoint(json_str: str) -> CheckpointEnvelope:
    """Deserialize checkpoint envelope from JSON.

    Args:
        json_str: JSON string representation.

    Returns:
        Parsed checkpoint envelope.

    Raises:
        ValueError: If JSON is invalid or doesn't match schema.

    """
    data = json.loads(json_str)
    return CheckpointEnvelope.model_validate(data)


def compute_content_hash(envelope: CheckpointEnvelope) -> str:
    """Compute SHA-256 hash of checkpoint and metadata.

    Hashes the checkpoint and metadata fields in canonical form
    for verification and deduplication.

    Args:
        envelope: The checkpoint envelope.

    Returns:
        Hex-encoded SHA-256 hash.

    """
    content: dict[str, Any] = {
        "checkpoint": envelope.checkpoint.model_dump(mode="json"),
        "metadata": envelope.metadata or {},
    }
    canonical_json = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def verify_content_hash(envelope: CheckpointEnvelope) -> bool:
    """Verify that the envelope's content hash matches its content.

    Args:
        envelope: The checkpoint envelope to verify.

    Returns:
        True if hash matches or is None, False otherwise.

    """
    if envelope.content_hash is None:
        return True
    computed = compute_content_hash(envelope)
    return computed == envelope.content_hash
