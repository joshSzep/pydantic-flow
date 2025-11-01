"""Tests for TypedSerializer with msgpack type preservation."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime

from pydantic import BaseModel

from pydantic_flow.checkpoints.serialization import TypedSerializer
from pydantic_flow.checkpoints.serialization import compress
from pydantic_flow.checkpoints.serialization import decompress


class SimpleModel(BaseModel):
    """Simple test model."""

    name: str
    age: int


class NestedModel(BaseModel):
    """Model with nested Pydantic model."""

    simple: SimpleModel
    items: list[str]


class DeeplyNestedModel(BaseModel):
    """Model with deep nesting."""

    nested: NestedModel
    timestamp: datetime
    metadata: dict[str, int]


def test_serialize_simple_model():
    """Test serialization of simple Pydantic model."""
    model = SimpleModel(name="Alice", age=30)
    data = TypedSerializer.serialize(model)

    assert isinstance(data, bytes)
    assert len(data) > 0


def test_deserialize_simple_model():
    """Test deserialization with type preservation."""
    model = SimpleModel(name="Bob", age=25)
    data = TypedSerializer.serialize(model)
    restored = TypedSerializer.deserialize(data)

    assert isinstance(restored, SimpleModel)
    assert restored.name == "Bob"
    assert restored.age == 25
    assert restored == model


def test_serialize_nested_model():
    """Test serialization of nested Pydantic models."""
    model = NestedModel(
        simple=SimpleModel(name="Charlie", age=35),
        items=["foo", "bar"],
    )
    data = TypedSerializer.serialize(model)
    restored = TypedSerializer.deserialize(data)

    assert isinstance(restored, NestedModel)
    assert isinstance(restored.simple, SimpleModel)
    assert restored.simple.name == "Charlie"
    assert restored.simple.age == 35
    assert restored.items == ["foo", "bar"]


def test_serialize_deeply_nested_model():
    """Test serialization of deeply nested models."""
    now = datetime.now(UTC)
    model = DeeplyNestedModel(
        nested=NestedModel(
            simple=SimpleModel(name="Dave", age=40),
            items=["x", "y", "z"],
        ),
        timestamp=now,
        metadata={"count": 42, "version": 1},
    )
    data = TypedSerializer.serialize(model)
    restored = TypedSerializer.deserialize(data)

    assert isinstance(restored, DeeplyNestedModel)
    assert isinstance(restored.nested, NestedModel)
    assert isinstance(restored.nested.simple, SimpleModel)
    assert restored.nested.simple.name == "Dave"
    assert restored.timestamp == now
    assert restored.metadata == {"count": 42, "version": 1}


def test_serialize_datetime():
    """Test datetime serialization."""
    now = datetime.now(UTC)
    data = TypedSerializer.serialize(now)
    restored = TypedSerializer.deserialize(data)

    assert isinstance(restored, datetime)
    assert restored == now


def test_serialize_dict_of_models():
    """Test serialization of dict containing models."""
    data_dict = {
        "alice": SimpleModel(name="Alice", age=30),
        "bob": SimpleModel(name="Bob", age=25),
    }
    data = TypedSerializer.serialize(data_dict)
    restored = TypedSerializer.deserialize(data)

    assert isinstance(restored, dict)
    assert isinstance(restored["alice"], SimpleModel)
    assert isinstance(restored["bob"], SimpleModel)
    assert restored["alice"].name == "Alice"
    assert restored["bob"].name == "Bob"


def test_serialize_list_of_models():
    """Test serialization of list containing models."""
    models = [
        SimpleModel(name="Alice", age=30),
        SimpleModel(name="Bob", age=25),
    ]
    data = TypedSerializer.serialize(models)
    restored = TypedSerializer.deserialize(data)

    assert isinstance(restored, list)
    assert len(restored) == 2
    assert isinstance(restored[0], SimpleModel)
    assert isinstance(restored[1], SimpleModel)
    assert restored[0].name == "Alice"
    assert restored[1].name == "Bob"


def test_serialize_primitives():
    """Test serialization of primitive types."""
    primitives = {
        "string": "hello",
        "int": 42,
        "float": 3.14,
        "bool": True,
        "none": None,
        "list": [1, 2, 3],
    }
    data = TypedSerializer.serialize(primitives)
    restored = TypedSerializer.deserialize(data)

    assert restored == primitives


def test_compression():
    """Test gzip compression utilities."""
    data = b"Hello World" * 100
    compressed = compress(data, level=6)

    assert len(compressed) < len(data)

    decompressed = decompress(compressed)
    assert decompressed == data


def test_compression_levels():
    """Test different compression levels."""
    data = b"Test data" * 1000

    compressed_fast = compress(data, level=1)
    compressed_default = compress(data, level=6)
    compressed_best = compress(data, level=9)

    assert len(compressed_fast) >= len(compressed_default)
    assert len(compressed_default) >= len(compressed_best)

    assert decompress(compressed_fast) == data
    assert decompress(compressed_default) == data
    assert decompress(compressed_best) == data


def test_serialization_size_vs_json():
    """Test that msgpack is smaller than JSON."""
    import json

    model = NestedModel(
        simple=SimpleModel(name="Test" * 50, age=100),
        items=["item" + str(i) for i in range(100)],
    )

    msgpack_data = TypedSerializer.serialize(model)
    json_data = json.dumps(model.model_dump(mode="json")).encode()

    assert len(msgpack_data) < len(json_data)


def test_roundtrip_preserves_equality():
    """Test that serialize/deserialize roundtrip preserves equality."""
    models = [
        SimpleModel(name="Test", age=25),
        NestedModel(
            simple=SimpleModel(name="Nested", age=30),
            items=["a", "b", "c"],
        ),
        DeeplyNestedModel(
            nested=NestedModel(
                simple=SimpleModel(name="Deep", age=35),
                items=["x"],
            ),
            timestamp=datetime.now(UTC),
            metadata={"key": 123},
        ),
    ]

    for model in models:
        data = TypedSerializer.serialize(model)
        restored = TypedSerializer.deserialize(data)
        assert restored == model


def test_schema_hash_stability():
    """Test that schema hashing is stable."""
    hash1 = TypedSerializer._hash_schema(SimpleModel)
    hash2 = TypedSerializer._hash_schema(SimpleModel)

    assert hash1 == hash2


def test_class_cache():
    """Test that class import caching works."""
    model1 = SimpleModel(name="First", age=25)
    data1 = TypedSerializer.serialize(model1)

    TypedSerializer._class_cache.clear()
    restored1 = TypedSerializer.deserialize(data1)

    model2 = SimpleModel(name="Second", age=30)
    data2 = TypedSerializer.serialize(model2)
    restored2 = TypedSerializer.deserialize(data2)

    assert isinstance(restored1, SimpleModel)
    assert isinstance(restored2, SimpleModel)
