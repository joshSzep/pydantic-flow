"""Tests for SQLite checkpoint backend (V2)."""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from pydantic_flow.checkpoints import RunId
from pydantic_flow.checkpoints import SnapshotReason
from pydantic_flow.checkpoints import SQLiteCheckpointBackend
from pydantic_flow.checkpoints import SQLiteCheckpointConfig
from pydantic_flow.checkpoints import StateSnapshot
from pydantic_flow.checkpoints.types import generate_snapshot_id


@pytest.fixture
async def backend(tmp_path: Path) -> AsyncGenerator[SQLiteCheckpointBackend]:
    """Create a SQLite backend with temp database."""
    config = SQLiteCheckpointConfig(db_path=tmp_path / "test.db")
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.mark.asyncio
async def test_wal_mode_enabled(tmp_path: Path) -> None:
    """Test that WAL mode is properly enabled."""
    import aiosqlite

    config = SQLiteCheckpointConfig(db_path=tmp_path / "test_wal.db")
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()

    async with aiosqlite.connect(config.db_path) as db:
        cursor = await db.execute("PRAGMA journal_mode")
        result = await cursor.fetchone()
        assert result is not None
        assert result[0].lower() == "wal"

    await backend.close()


@pytest.mark.asyncio
async def test_schema_initialization(tmp_path: Path) -> None:
    """Test that database schema is created on initialization."""
    import aiosqlite

    config = SQLiteCheckpointConfig(db_path=tmp_path / "test_schema.db")
    backend = SQLiteCheckpointBackend(config)

    assert not config.db_path.exists()

    await backend.initialize()

    assert config.db_path.exists()

    async with aiosqlite.connect(config.db_path) as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='state_snapshots'"
        )
        result = await cursor.fetchone()
        assert result is not None
        assert result[0] == "state_snapshots"

    await backend.close()


@pytest.mark.asyncio
async def test_indexes_created(tmp_path: Path) -> None:
    """Test that indexes are created for query performance."""
    import aiosqlite

    config = SQLiteCheckpointConfig(db_path=tmp_path / "test_indexes.db")
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()

    async with aiosqlite.connect(config.db_path) as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in await cursor.fetchall()]

    assert len(indexes) > 0

    await backend.close()


@pytest.mark.asyncio
async def test_concurrent_access(backend: SQLiteCheckpointBackend) -> None:
    """Test concurrent database access with busy timeout."""
    import asyncio

    run_id = RunId("concurrent_test")

    async def save_snapshot(num: int) -> None:
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=num,
            full_state={},
            state_hash=f"hash_{num}",
            next_frontier=[],
            routing_ended=False,
            reason=SnapshotReason.AUTOMATIC,
        )
        await backend.save_state_snapshot(snapshot)

    await asyncio.gather(*[save_snapshot(i) for i in range(10)])

    snapshots = await backend.get_snapshots_range(run_id, 0, 9)
    assert len(snapshots) == 10


@pytest.mark.asyncio
async def test_database_file_created_with_parents(tmp_path: Path) -> None:
    """Test that database file and parent directories are created."""
    nested_path = tmp_path / "deep" / "nested" / "path" / "checkpoints.db"
    config = SQLiteCheckpointConfig(db_path=nested_path)
    backend = SQLiteCheckpointBackend(config)

    assert not nested_path.exists()

    await backend.initialize()

    assert nested_path.exists()
    assert nested_path.parent.exists()

    await backend.close()


@pytest.mark.asyncio
async def test_save_and_retrieve_snapshot(backend: SQLiteCheckpointBackend) -> None:
    """Test saving and retrieving a state snapshot."""
    run_id = RunId("test_run")
    snapshot_id = generate_snapshot_id()

    snapshot = StateSnapshot(
        snapshot_id=snapshot_id,
        run_id=run_id,
        wave_number=0,
        full_state={},
        state_hash="test_hash",
        next_frontier=["node1"],
        routing_ended=False,
        reason=SnapshotReason.HITL_INTERRUPT,
        interrupted_node_id="test_node",
    )

    await backend.save_state_snapshot(snapshot)

    retrieved = await backend.get_state_snapshot(run_id, 0)
    assert retrieved is not None
    assert retrieved.snapshot_id == snapshot_id
    assert retrieved.run_id == run_id
    assert retrieved.reason == SnapshotReason.HITL_INTERRUPT
    assert retrieved.interrupted_node_id == "test_node"
