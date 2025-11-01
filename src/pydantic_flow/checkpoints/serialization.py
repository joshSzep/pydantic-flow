"""Type-safe serialization with msgpack and type preservation.

This module provides a serializer that preserves Pydantic model types during
serialization using msgpack extension types.
"""

from __future__ import annotations

from datetime import datetime
import gzip
import hashlib
import importlib
import json
from typing import Any
from typing import ClassVar

import msgpack
from pydantic import BaseModel


class TypedSerializer:
    """msgpack serializer that preserves Pydantic type information.

    Uses msgpack extension types to encode type metadata alongside data,
    enabling complete type reconstruction on deserialization.
    """

    TYPE_PYDANTIC = 1
    TYPE_DATETIME = 2
    TYPE_DELETED_KEY = 3

    _class_cache: ClassVar[dict[str, type[BaseModel]]] = {}

    @classmethod
    def serialize(cls, obj: Any) -> bytes:
        """Serialize with type preservation.

        Args:
            obj: Object to serialize (supports Pydantic models, primitives, etc.).

        Returns:
            Serialized msgpack bytes.

        """
        return msgpack.packb(obj, default=cls._encode_hook, use_bin_type=True)

    @classmethod
    def deserialize(cls, data: bytes) -> Any:
        """Deserialize with type reconstruction.

        Args:
            data: msgpack bytes to deserialize.

        Returns:
            Reconstructed object with preserved types.

        """
        return msgpack.unpackb(
            data, ext_hook=cls._decode_ext_hook, raw=False, strict_map_key=False
        )

    @classmethod
    def _encode_hook(cls, obj: Any) -> Any:
        """Encode objects using msgpack extension types.

        Args:
            obj: Object to encode.

        Returns:
            msgpack ExtType or raises TypeError.

        Raises:
            TypeError: If object type is not supported.

        """
        if isinstance(obj, BaseModel):
            type_path = f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"

            # Get fields without converting nested BaseModels to dicts
            model_data = {k: getattr(obj, k) for k in obj.__class__.model_fields}

            # Use default= to handle nested objects recursively
            inner = msgpack.packb(
                {
                    "t": type_path,
                    "d": model_data,
                    "h": cls._hash_schema(obj.__class__)[:8],
                },
                default=cls._encode_hook,
                use_bin_type=True,
            )

            return msgpack.ExtType(cls.TYPE_PYDANTIC, inner)

        if isinstance(obj, datetime):
            inner = obj.isoformat().encode()
            return msgpack.ExtType(cls.TYPE_DATETIME, inner)

        # Handle DeletedKey sentinel
        from pydantic_flow.checkpoints.types import DeletedKey

        if isinstance(obj, DeletedKey):
            return msgpack.ExtType(cls.TYPE_DELETED_KEY, b"")

        msg = f"Cannot serialize type {type(obj)}"
        raise TypeError(msg)

    @classmethod
    def _decode_ext_hook(cls, code: int, data: bytes) -> Any:
        """Decode msgpack extension types.

        Args:
            code: Extension type code.
            data: Extension data bytes.

        Returns:
            Decoded object or ExtType if unknown.

        """
        if code == cls.TYPE_PYDANTIC:
            obj = msgpack.unpackb(data, ext_hook=cls._decode_ext_hook, raw=False)
            type_path = obj["t"]

            if type_path not in cls._class_cache:
                cls._class_cache[type_path] = cls._import_class(type_path)

            model_class = cls._class_cache[type_path]
            return model_class.model_validate(obj["d"])

        if code == cls.TYPE_DATETIME:
            return datetime.fromisoformat(data.decode())

        if code == cls.TYPE_DELETED_KEY:
            from pydantic_flow.checkpoints.types import DELETED_KEY

            return DELETED_KEY

        return msgpack.ExtType(code, data)

    @staticmethod
    def _hash_schema(model_class: type[BaseModel]) -> str:
        """Compute schema hash for validation.

        Args:
            model_class: Pydantic model class.

        Returns:
            Hex-encoded SHA-256 hash of schema.

        """
        # Handle BaseModel itself (no schema available)
        if model_class is BaseModel:
            return "basemodel_generic"

        schema_json = model_class.model_json_schema()
        canonical = json.dumps(schema_json, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _import_class(type_path: str) -> type[BaseModel]:
        """Dynamically import class from module path.

        Args:
            type_path: Fully qualified class path (module.ClassName).

        Returns:
            Imported class.

        Raises:
            ImportError: If module or class not found.

        """
        module_path, class_name = type_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


def compress(data: bytes, level: int = 6) -> bytes:
    """Compress data with gzip.

    Args:
        data: Data to compress.
        level: Compression level (1=fast, 9=best, 6=default).

    Returns:
        Compressed bytes.

    """
    return gzip.compress(data, compresslevel=level)


def decompress(data: bytes) -> bytes:
    """Decompress gzip data.

    Args:
        data: Compressed data.

    Returns:
        Decompressed bytes.

    """
    return gzip.decompress(data)
