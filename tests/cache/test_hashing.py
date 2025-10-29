"""Tests for cache hashing utilities."""

from decimal import Decimal

from pydantic_flow.cache.hashing import canonical_json
from pydantic_flow.cache.hashing import hash_bytes
from pydantic_flow.cache.hashing import hash_json


def test_canonical_json_sorts_keys() -> None:
    """Canonical JSON should sort dictionary keys."""
    obj1 = {"z": 1, "a": 2, "m": 3}
    obj2 = {"a": 2, "m": 3, "z": 1}

    assert canonical_json(obj1) == canonical_json(obj2)


def test_canonical_json_normalizes_floats() -> None:
    """Canonical JSON should normalize floats consistently."""
    obj1 = {"value": 3.14}
    obj2 = {"value": 3.14000000}

    assert canonical_json(obj1) == canonical_json(obj2)


def test_canonical_json_handles_decimals() -> None:
    """Canonical JSON should convert Decimal to normalized floats."""
    obj1 = {"value": Decimal("3.14")}
    obj2 = {"value": 3.14}

    assert canonical_json(obj1) == canonical_json(obj2)


def test_canonical_json_sorts_nested() -> None:
    """Canonical JSON should sort nested dictionaries."""
    obj1 = {"outer": {"z": 1, "a": 2}}
    obj2 = {"outer": {"a": 2, "z": 1}}

    assert canonical_json(obj1) == canonical_json(obj2)


def test_canonical_json_handles_lists() -> None:
    """Canonical JSON should preserve list order."""
    obj1 = {"items": [3, 1, 2]}
    obj2 = {"items": [3, 1, 2]}
    obj3 = {"items": [1, 2, 3]}

    assert canonical_json(obj1) == canonical_json(obj2)
    assert canonical_json(obj1) != canonical_json(obj3)


def test_hash_json_deterministic() -> None:
    """hash_json should produce same hash for equivalent objects."""
    obj = {"model": "gpt-4", "temperature": 0.7, "messages": ["hello"]}

    hash1 = hash_json(obj)
    hash2 = hash_json(obj)

    assert hash1 == hash2
    assert len(hash1) == 64


def test_hash_json_different_for_different_objects() -> None:
    """hash_json should produce different hashes for different objects."""
    obj1 = {"model": "gpt-4", "temperature": 0.7}
    obj2 = {"model": "gpt-4", "temperature": 0.8}

    hash1 = hash_json(obj1)
    hash2 = hash_json(obj2)

    assert hash1 != hash2


def test_hash_bytes_deterministic() -> None:
    """hash_bytes should produce same hash for same bytes."""
    data = b"test data"

    hash1 = hash_bytes(data)
    hash2 = hash_bytes(data)

    assert hash1 == hash2
    assert len(hash1) == 64


def test_hash_bytes_different_for_different_data() -> None:
    """hash_bytes should produce different hashes for different bytes."""
    data1 = b"test data 1"
    data2 = b"test data 2"

    hash1 = hash_bytes(data1)
    hash2 = hash_bytes(data2)

    assert hash1 != hash2
