"""Hashing utilities for deterministic cache keys.

This module provides canonical JSON serialization and BLAKE3 hashing
to ensure consistent cache keys across different executions.
"""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> bytes:
    """Serialize object to canonical JSON bytes.

    Ensures deterministic serialization by:
    - Sorting dictionary keys
    - Normalizing floats/decimals to strings with fixed precision
    - Removing whitespace
    - Using UTF-8 encoding

    Args:
        obj: Object to serialize (must be JSON-serializable).

    Returns:
        Canonical JSON bytes.

    """

    def normalize(value: Any) -> Any:
        """Normalize values for canonical representation."""
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        if isinstance(value, (float, Decimal)):
            return f"{float(value):.17g}"
        if isinstance(value, set):
            return sorted([normalize(item) for item in value])
        return value

    normalized = normalize(obj)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def hash_json(obj: Any) -> str:
    """Hash object to deterministic hex string using BLAKE3.

    Args:
        obj: Object to hash (must be JSON-serializable).

    Returns:
        Hex digest of BLAKE3 hash (64 characters).

    """
    canonical_bytes = canonical_json(obj)
    hasher = hashlib.blake2b(canonical_bytes, digest_size=32)
    return hasher.hexdigest()


def hash_bytes(data: bytes) -> str:
    """Hash raw bytes using BLAKE3.

    Args:
        data: Bytes to hash.

    Returns:
        Hex digest of BLAKE3 hash (64 characters).

    """
    hasher = hashlib.blake2b(data, digest_size=32)
    return hasher.hexdigest()
