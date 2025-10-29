"""Tests for cache hashing utilities."""

from decimal import Decimal

from pydantic_flow.cache.hashing import canonical_json
from pydantic_flow.cache.hashing import hash_bytes
from pydantic_flow.cache.hashing import hash_json


def test_canonical_json_simple_dict() -> None:
    """Test canonical JSON for simple dict."""
    obj = {"name": "Alice", "age": 30}
    result = canonical_json(obj)

    assert isinstance(result, bytes)
    assert b"age" in result
    assert b"name" in result


def test_canonical_json_sorted_keys() -> None:
    """Test that dict keys are sorted."""
    obj = {"z": 1, "a": 2, "m": 3}
    result = canonical_json(obj)
    text = result.decode("utf-8")

    assert text.index('"a"') < text.index('"m"') < text.index('"z"')


def test_canonical_json_nested_dict() -> None:
    """Test canonical JSON with nested dicts."""
    obj = {"outer": {"inner": {"deep": "value"}}}
    result = canonical_json(obj)

    assert b"outer" in result
    assert b"inner" in result
    assert b"deep" in result


def test_canonical_json_list() -> None:
    """Test canonical JSON with list."""
    obj = {"items": [1, 2, 3]}
    result = canonical_json(obj)

    assert b"[1,2,3]" in result


def test_canonical_json_tuple() -> None:
    """Test canonical JSON converts tuples to lists."""
    obj = {"coords": (1, 2, 3)}
    result = canonical_json(obj)

    assert b"[1,2,3]" in result


def test_canonical_json_float() -> None:
    """Test canonical JSON normalizes floats."""
    obj = {"value": 3.14159}
    result = canonical_json(obj)

    assert isinstance(result, bytes)
    assert b"3.14159" in result or b"3.1415" in result


def test_canonical_json_decimal() -> None:
    """Test canonical JSON converts Decimal to string."""
    obj = {"price": Decimal("19.99")}
    result = canonical_json(obj)

    assert b"19.99" in result or b"19.989" in result


def test_canonical_json_set() -> None:
    """Test canonical JSON converts sets to sorted lists."""
    obj = {"tags": {3, 1, 2}}
    result = canonical_json(obj)

    assert b"[1,2,3]" in result


def test_canonical_json_deterministic() -> None:
    """Test that identical objects produce identical JSON."""
    obj = {"name": "Bob", "age": 25, "active": True}
    result1 = canonical_json(obj)
    result2 = canonical_json(obj)

    assert result1 == result2


def test_canonical_json_key_order_independence() -> None:
    """Test that different key orders produce identical JSON."""
    obj1 = {"name": "Alice", "age": 30}
    obj2 = {"age": 30, "name": "Alice"}
    result1 = canonical_json(obj1)
    result2 = canonical_json(obj2)

    assert result1 == result2


def test_hash_json_basic() -> None:
    """Test basic JSON hashing."""
    obj = {"key": "value"}
    result = hash_json(obj)

    assert isinstance(result, str)
    assert len(result) == 64


def test_hash_json_deterministic() -> None:
    """Test that identical objects produce identical hashes."""
    obj = {"name": "Charlie", "age": 35, "role": "engineer"}
    hash1 = hash_json(obj)
    hash2 = hash_json(obj)

    assert hash1 == hash2


def test_hash_json_key_order_independence() -> None:
    """Test that key order doesn't affect hash."""
    obj1 = {"name": "Dana", "age": 28}
    obj2 = {"age": 28, "name": "Dana"}
    hash1 = hash_json(obj1)
    hash2 = hash_json(obj2)

    assert hash1 == hash2


def test_hash_json_different_values() -> None:
    """Test that different values produce different hashes."""
    obj1 = {"key": "value1"}
    obj2 = {"key": "value2"}
    hash1 = hash_json(obj1)
    hash2 = hash_json(obj2)

    assert hash1 != hash2


def test_hash_json_complex_object() -> None:
    """Test hashing complex nested object."""
    obj = {
        "user": {"name": "Eve", "roles": ["admin", "user"]},
        "metadata": {"created": "2024-01-01", "active": True},
    }
    result = hash_json(obj)

    assert isinstance(result, str)
    assert len(result) == 64


def test_hash_bytes_basic() -> None:
    """Test basic byte hashing."""
    data = b"Hello, world!"
    result = hash_bytes(data)

    assert isinstance(result, str)
    assert len(result) == 64


def test_hash_bytes_deterministic() -> None:
    """Test that identical bytes produce identical hashes."""
    data = b"test data"
    hash1 = hash_bytes(data)
    hash2 = hash_bytes(data)

    assert hash1 == hash2


def test_hash_bytes_different_data() -> None:
    """Test that different bytes produce different hashes."""
    hash1 = hash_bytes(b"data1")
    hash2 = hash_bytes(b"data2")

    assert hash1 != hash2


def test_hash_bytes_empty() -> None:
    """Test hashing empty bytes."""
    result = hash_bytes(b"")

    assert isinstance(result, str)
    assert len(result) == 64
