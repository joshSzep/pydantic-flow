"""Tests for SQLiteCheckpointStore implementation."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_flow.hitl.checkpoints.sqlite import SQLiteCheckpointStore
from pydantic_flow.hitl.checkpoints.sqlite import SQLiteCheckpointStoreConfig
from tests.test_checkpoints_conformance import CheckpointStoreConformanceTests


class TestSQLiteCheckpointStoreConformance(CheckpointStoreConformanceTests):
    """Conformance tests for SQLiteCheckpointStore."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> SQLiteCheckpointStore:
        """Create a SQLiteCheckpointStore instance using temp database."""
        config = SQLiteCheckpointStoreConfig(
            db_path=tmp_path / "test_checkpoints.db",
            busy_timeout_ms=30000,
        )
        store = SQLiteCheckpointStore(config)
        await store.healthcheck()
        return store


class TestSQLiteCheckpointStoreSpecific:
    """Tests specific to SQLiteCheckpointStore implementation."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> SQLiteCheckpointStore:
        """Create a SQLiteCheckpointStore instance for testing."""
        config = SQLiteCheckpointStoreConfig(
            db_path=tmp_path / "test_checkpoints.db",
            busy_timeout_ms=30000,
        )
        store = SQLiteCheckpointStore(config)
        await store.healthcheck()
        return store

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        """Test that WAL mode is properly enabled."""
        import aiosqlite

        config = SQLiteCheckpointStoreConfig(
            db_path=tmp_path / "test_wal.db",
        )
        store = SQLiteCheckpointStore(config)
        await store.healthcheck()

        async with aiosqlite.connect(config.db_path) as db:
            cursor = await db.execute("PRAGMA journal_mode")
            result = await cursor.fetchone()
            assert result is not None
            assert result[0].lower() == "wal"

    @pytest.mark.asyncio
    async def test_schema_initialization(self, tmp_path: Path) -> None:
        """Test that database schema is created on first access."""
        import aiosqlite

        config = SQLiteCheckpointStoreConfig(
            db_path=tmp_path / "test_schema.db",
        )
        store = SQLiteCheckpointStore(config)

        assert not config.db_path.exists()

        await store.healthcheck()

        assert config.db_path.exists()

        async with aiosqlite.connect(config.db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='checkpoints'"
            )
            result = await cursor.fetchone()
            assert result is not None
            assert result[0] == "checkpoints"

    @pytest.mark.asyncio
    async def test_indexes_created(self, tmp_path: Path) -> None:
        """Test that indexes are created for query performance."""
        import aiosqlite

        config = SQLiteCheckpointStoreConfig(
            db_path=tmp_path / "test_indexes.db",
        )
        store = SQLiteCheckpointStore(config)
        await store.healthcheck()

        async with aiosqlite.connect(config.db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
            indexes = [row[0] for row in await cursor.fetchall()]

        assert "idx_run_created" in indexes
        assert "idx_run_node_created" in indexes

    @pytest.mark.asyncio
    async def test_concurrent_access(
        self, store: SQLiteCheckpointStore, sample_checkpoint
    ) -> None:
        """Test concurrent database access with busy timeout."""
        import asyncio
        from datetime import UTC
        from datetime import datetime

        from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
        from pydantic_flow.hitl.checkpoints.interface import CheckpointId
        from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
        from pydantic_flow.hitl.checkpoints.interface import RunId

        run_id = RunId("concurrent_test")

        async def save_checkpoint(num: int) -> None:
            envelope = CheckpointEnvelope(
                id=CheckpointId(f"ckpt_{num}"),
                run_id=run_id,
                node_id=f"node_{num}",
                created_at=datetime.now(UTC),
                schema_version=1,
                checkpoint=sample_checkpoint,
            )
            await store.save(envelope)

        await asyncio.gather(*[save_checkpoint(i) for i in range(10)])

        query = CheckpointQuery(run_id=run_id)
        checkpoints, _ = await store.list(query)
        assert len(checkpoints) == 10

    @pytest.mark.asyncio
    async def test_database_file_created_with_parents(self, tmp_path: Path) -> None:
        """Test that database file and parent directories are created."""
        nested_path = tmp_path / "deep" / "nested" / "path" / "checkpoints.db"
        config = SQLiteCheckpointStoreConfig(db_path=nested_path)
        store = SQLiteCheckpointStore(config)

        await store.healthcheck()

        assert nested_path.exists()
        assert nested_path.parent.exists()

    @pytest.mark.asyncio
    async def test_content_hash_stored(
        self, store: SQLiteCheckpointStore, sample_envelope
    ) -> None:
        """Test that content hash is computed and stored."""
        import aiosqlite

        saved = await store.save(sample_envelope)

        assert saved.content_hash is not None

        async with aiosqlite.connect(store.config.db_path) as db:
            cursor = await db.execute(
                "SELECT content_hash FROM checkpoints WHERE checkpoint_id = ?",
                (str(saved.id),),
            )
            result = await cursor.fetchone()
            assert result is not None
            assert result[0] == saved.content_hash
