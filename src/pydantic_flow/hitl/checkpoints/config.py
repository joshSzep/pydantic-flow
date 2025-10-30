"""Configuration for checkpoint stores.

Provides a unified configuration interface and factory for creating stores.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel
from pydantic import Field

from pydantic_flow.hitl.checkpoints.flatfile import FlatFileCheckpointStore
from pydantic_flow.hitl.checkpoints.flatfile import FlatFileCheckpointStoreConfig
from pydantic_flow.hitl.checkpoints.interface import CheckpointStore
from pydantic_flow.hitl.checkpoints.memory import InMemoryCheckpointStore
from pydantic_flow.hitl.checkpoints.postgres import PostgresCheckpointStore
from pydantic_flow.hitl.checkpoints.postgres import PostgresCheckpointStoreConfig
from pydantic_flow.hitl.checkpoints.redis import RedisCheckpointStore
from pydantic_flow.hitl.checkpoints.redis import RedisCheckpointStoreConfig
from pydantic_flow.hitl.checkpoints.s3 import S3CheckpointStore
from pydantic_flow.hitl.checkpoints.s3 import S3CheckpointStoreConfig
from pydantic_flow.hitl.checkpoints.sqlite import SQLiteCheckpointStore
from pydantic_flow.hitl.checkpoints.sqlite import SQLiteCheckpointStoreConfig


class CheckpointStoreType(Enum):
    """Available checkpoint store backends."""

    MEMORY = "memory"
    FLATFILE = "flatfile"
    SQLITE = "sqlite"
    REDIS = "redis"
    POSTGRES = "postgres"
    S3 = "s3"


class CheckpointStoreConfig(BaseModel):
    """Unified configuration for checkpoint stores.

    Exactly one store configuration must be provided.

    Attributes:
        memory: In-memory store configuration.
        flatfile: Flat-file JSON store configuration.
        sqlite: SQLite store configuration.
        redis: Redis store configuration.
        postgres: Postgres store configuration.
        s3: S3-compatible store configuration.

    """

    memory: Annotated[bool, Field()] | None = None
    flatfile: FlatFileCheckpointStoreConfig | None = None
    sqlite: SQLiteCheckpointStoreConfig | None = None
    redis: RedisCheckpointStoreConfig | None = None
    postgres: PostgresCheckpointStoreConfig | None = None
    s3: S3CheckpointStoreConfig | None = None


def create_checkpoint_store(config: CheckpointStoreConfig) -> CheckpointStore:
    """Create a checkpoint store from configuration.

    Args:
        config: Store configuration with exactly one backend specified.

    Returns:
        Configured checkpoint store instance.

    Raises:
        ValueError: If no backend or multiple backends are specified.

    """
    backends = [
        ("memory", config.memory is not None),
        ("flatfile", config.flatfile is not None),
        ("sqlite", config.sqlite is not None),
        ("redis", config.redis is not None),
        ("postgres", config.postgres is not None),
        ("s3", config.s3 is not None),
    ]

    active_backends = [name for name, active in backends if active]

    if len(active_backends) == 0:
        msg = "No checkpoint store backend specified"
        raise ValueError(msg)

    if len(active_backends) > 1:
        msg = f"Multiple checkpoint store backends specified: {active_backends}"
        raise ValueError(msg)

    backend = active_backends[0]

    if backend == "memory":
        return InMemoryCheckpointStore()
    elif backend == "flatfile":
        return FlatFileCheckpointStore(config.flatfile)  # type: ignore[arg-type]
    elif backend == "sqlite":
        return SQLiteCheckpointStore(config.sqlite)  # type: ignore[arg-type]
    elif backend == "redis":
        return RedisCheckpointStore(config.redis)  # type: ignore[arg-type]
    elif backend == "postgres":
        return PostgresCheckpointStore(config.postgres)  # type: ignore[arg-type]
    elif backend == "s3":
        return S3CheckpointStore(config.s3)  # type: ignore[arg-type]
    else:
        msg = f"Unknown checkpoint store backend: {backend}"
        raise ValueError(msg)
