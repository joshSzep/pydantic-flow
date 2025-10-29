"""Tests for checkpoint configuration."""

from pathlib import Path

import pytest

from pydantic_flow.checkpoints.config import CheckpointStoreConfig
from pydantic_flow.checkpoints.config import create_checkpoint_store
from pydantic_flow.checkpoints.flatfile import FlatFileCheckpointStore
from pydantic_flow.checkpoints.flatfile import FlatFileCheckpointStoreConfig
from pydantic_flow.checkpoints.memory import InMemoryCheckpointStore
from pydantic_flow.checkpoints.postgres import PostgresCheckpointStore
from pydantic_flow.checkpoints.postgres import PostgresCheckpointStoreConfig
from pydantic_flow.checkpoints.redis import RedisCheckpointStore
from pydantic_flow.checkpoints.redis import RedisCheckpointStoreConfig
from pydantic_flow.checkpoints.s3 import S3CheckpointStore
from pydantic_flow.checkpoints.s3 import S3CheckpointStoreConfig
from pydantic_flow.checkpoints.sqlite import SQLiteCheckpointStore
from pydantic_flow.checkpoints.sqlite import SQLiteCheckpointStoreConfig


def test_create_memory_checkpoint_store() -> None:
    """Test creating in-memory checkpoint store."""
    config = CheckpointStoreConfig(memory=True)
    store = create_checkpoint_store(config)

    assert isinstance(store, InMemoryCheckpointStore)


def test_create_flatfile_checkpoint_store() -> None:
    """Test creating flatfile checkpoint store."""
    flatfile_config = FlatFileCheckpointStoreConfig(base_path=Path("/tmp/checkpoints"))
    config = CheckpointStoreConfig(flatfile=flatfile_config)
    store = create_checkpoint_store(config)

    assert isinstance(store, FlatFileCheckpointStore)


def test_create_sqlite_checkpoint_store() -> None:
    """Test creating SQLite checkpoint store."""
    sqlite_config = SQLiteCheckpointStoreConfig(db_path=Path(":memory:"))
    config = CheckpointStoreConfig(sqlite=sqlite_config)
    store = create_checkpoint_store(config)

    assert isinstance(store, SQLiteCheckpointStore)


def test_create_redis_checkpoint_store() -> None:
    """Test creating Redis checkpoint store."""
    redis_config = RedisCheckpointStoreConfig(redis_url="redis://localhost:6379")
    config = CheckpointStoreConfig(redis=redis_config)
    store = create_checkpoint_store(config)

    assert isinstance(store, RedisCheckpointStore)


def test_create_postgres_checkpoint_store() -> None:
    """Test creating Postgres checkpoint store."""
    postgres_config = PostgresCheckpointStoreConfig(
        dsn="postgresql://test:test@localhost:5432/test",
    )
    config = CheckpointStoreConfig(postgres=postgres_config)
    store = create_checkpoint_store(config)

    assert isinstance(store, PostgresCheckpointStore)


def test_create_s3_checkpoint_store() -> None:
    """Test creating S3 checkpoint store."""
    s3_config = S3CheckpointStoreConfig(
        bucket="test-bucket",
        region_name="us-east-1",
    )
    config = CheckpointStoreConfig(s3=s3_config)
    store = create_checkpoint_store(config)

    assert isinstance(store, S3CheckpointStore)


def test_create_checkpoint_store_no_backend_raises() -> None:
    """Test that creating store with no backend raises ValueError."""
    config = CheckpointStoreConfig()

    with pytest.raises(ValueError, match="No checkpoint store backend specified"):
        create_checkpoint_store(config)


def test_create_checkpoint_store_multiple_backends_raises() -> None:
    """Test that creating store with multiple backends raises ValueError."""
    config = CheckpointStoreConfig(
        memory=True,
        flatfile=FlatFileCheckpointStoreConfig(base_path=Path("/tmp")),
    )

    with pytest.raises(ValueError, match="Multiple checkpoint store backends"):
        create_checkpoint_store(config)
