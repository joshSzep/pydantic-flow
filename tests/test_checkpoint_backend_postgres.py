"""Tests for PostgreSQL checkpoint backend v2."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import patch
import uuid

from pydantic import BaseModel
import pytest

from pydantic_flow.checkpoints.backends.postgres import PostgresCheckpointBackend
from pydantic_flow.checkpoints.backends.postgres import PostgresCheckpointConfig
from pydantic_flow.checkpoints.types import EventSummary
from pydantic_flow.checkpoints.types import ExecutionTrace
from pydantic_flow.checkpoints.types import NodeExecutionTrace
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateRef
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id


class SampleState(BaseModel):
    """Sample state for testing."""

    value: int
    name: str


class MockAsyncPGPool:
    """Mock asyncpg connection pool."""

    def __init__(self) -> None:
        """Initialize mock pool."""
        self.closed = False
        self._data: dict[str, list[dict]] = {
            "state_snapshots": [],
            "execution_traces": [],
            "node_traces": [],
            "run_metadata": [],
        }

    async def close(self) -> None:
        """Close pool."""
        self.closed = True

    def acquire(self) -> MockAsyncPGConnection:
        """Acquire connection."""
        return MockAsyncPGConnection(self._data)

    async def __aenter__(self) -> MockAsyncPGPool:
        """Context manager entry."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Context manager exit."""
        pass


class MockAsyncPGConnection:
    """Mock asyncpg connection."""

    def __init__(self, data: dict[str, list[dict]]) -> None:
        """Initialize mock connection."""
        self._data = data

    async def __aenter__(self) -> MockAsyncPGConnection:
        """Context manager entry."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Context manager exit."""
        pass

    async def execute(self, query: str, *args: object) -> str:  # noqa: PLR0911
        """Execute query."""
        if "CREATE TABLE" in query or "CREATE INDEX" in query:
            return "CREATE"
        elif "INSERT INTO run_metadata" in query:
            self._data["run_metadata"].append({
                "run_id": args[0],
                "flow_id": args[1],
                "started_at": args[2],
                "completed_at": args[3],
                "status": args[4],
                "total_waves": args[5],
                "error_json": args[6],
            })
            return "INSERT 0 1"
        elif "INSERT INTO state_snapshots" in query:
            self._data["state_snapshots"].append({
                "snapshot_id": args[0],
                "run_id": args[1],
                "wave_number": args[2],
                "data_compressed": args[3],
                "state_hash": args[4],
                "trace_id": args[5],
                "created_at": args[6],
            })
            return "INSERT 0 1"
        elif "INSERT INTO execution_traces" in query:
            self._data["execution_traces"].append({
                "trace_id": args[0],
                "run_id": args[1],
                "wave_number": args[2],
                "checkpoint_snapshot_id": args[3],
                "data_compressed": args[4],
                "created_at": args[5],
            })
            return "INSERT 0 1"
        elif "INSERT INTO node_traces" in query:
            self._data["node_traces"].append({
                "log_id": args[0],
                "trace_id": args[1],
                "node_id": args[2],
                "wave_number": args[3],
                "data_compressed": args[4],
                "created_at": args[5],
            })
            return "INSERT 0 1"
        elif "UPDATE state_snapshots" in query:
            for snapshot in self._data["state_snapshots"]:
                if snapshot["snapshot_id"] == args[2]:
                    snapshot["data_compressed"] = args[0]
                    snapshot["trace_id"] = args[1]
            return "UPDATE 1"
        elif "DELETE FROM execution_traces" in query:
            count = len([
                t for t in self._data["execution_traces"] if t["run_id"] == args[0]
            ])
            self._data["execution_traces"] = [
                t for t in self._data["execution_traces"] if t["run_id"] != args[0]
            ]
            return f"DELETE {count}"
        elif "DELETE FROM run_metadata" in query:
            count = len([
                m for m in self._data["run_metadata"] if m["run_id"] == args[0]
            ])
            self._data["run_metadata"] = [
                m for m in self._data["run_metadata"] if m["run_id"] != args[0]
            ]
            return f"DELETE {count}"
        elif "DELETE FROM state_snapshots" in query:
            count = len([
                s for s in self._data["state_snapshots"] if s["run_id"] == args[0]
            ])
            self._data["state_snapshots"] = [
                s for s in self._data["state_snapshots"] if s["run_id"] != args[0]
            ]
            return f"DELETE {count}"
        return "OK"

    async def fetchval(self, query: str, *args: object) -> object:
        """Fetch single value."""
        if "SELECT 1" in query:
            return 1
        elif "SELECT 1 FROM state_snapshots" in query:
            for snapshot in self._data["state_snapshots"]:
                if snapshot["snapshot_id"] == args[0]:
                    return 1
            return None
        return None

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        """Fetch single row."""
        if "FROM run_metadata" in query:
            for metadata in self._data["run_metadata"]:
                if metadata["run_id"] == args[0]:
                    return metadata
        elif "FROM state_snapshots" in query:
            for snapshot in self._data["state_snapshots"]:
                if snapshot["run_id"] == args[0] and snapshot["wave_number"] == args[1]:
                    return snapshot
        elif "FROM execution_traces" in query:
            for trace in self._data["execution_traces"]:
                if trace["run_id"] == args[0] and trace["wave_number"] == args[1]:
                    return trace
        elif "FROM node_traces" in query:
            for node_trace in self._data["node_traces"]:
                if node_trace["log_id"] == args[0]:
                    return node_trace
        return None

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        """Fetch multiple rows."""
        if "FROM state_snapshots" in query:
            results = []
            for snapshot in self._data["state_snapshots"]:
                if (
                    snapshot["run_id"] == args[0]
                    and args[1] <= snapshot["wave_number"] <= args[2]
                ):
                    results.append(snapshot)
            if "DESC" in query:
                results.reverse()
            return results
        elif "FROM run_metadata" in query:
            results = list(self._data["run_metadata"])
            if "started_at DESC" in query:
                results.sort(
                    key=lambda x: x["started_at"],
                    reverse=True,  # type: ignore[arg-type, return-value]
                )
            return results
        return []


@pytest.fixture
async def mock_asyncpg() -> AsyncMock:
    """Mock asyncpg module."""
    mock_pool = MockAsyncPGPool()

    async def create_pool(*args: object, **kwargs: object) -> MockAsyncPGPool:
        return mock_pool

    mock_asyncpg = AsyncMock()
    mock_asyncpg.create_pool = create_pool
    return mock_asyncpg


@pytest.fixture
async def postgres_backend(
    mock_asyncpg: AsyncMock,
) -> AsyncGenerator[PostgresCheckpointBackend]:
    """Create postgres backend with mocked asyncpg."""
    config = PostgresCheckpointConfig(
        connection_string="postgresql://test:test@localhost:5432/test"
    )
    backend = PostgresCheckpointBackend(config)

    with patch("asyncpg.create_pool", mock_asyncpg.create_pool):
        await backend.initialize()

    yield backend
    await backend.close()


@pytest.mark.asyncio
async def test_postgres_healthcheck(
    postgres_backend: PostgresCheckpointBackend,
) -> None:
    """Test postgres healthcheck."""
    health = await postgres_backend.healthcheck()
    assert health is True


@pytest.mark.asyncio
async def test_save_and_get_snapshot(
    postgres_backend: PostgresCheckpointBackend,
) -> None:
    """Test saving and retrieving state snapshots."""
    run_id = generate_run_id()
    snapshot = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=0,
        full_state={"node1": SampleState(value=1, name="test")},
        state_hash="hash123",
        next_frontier=["node2"],
        routing_ended=False,
    )

    await postgres_backend.save_state_snapshot(snapshot)
    retrieved = await postgres_backend.get_state_snapshot(run_id, 0)

    assert retrieved is not None
    assert retrieved.snapshot_id == snapshot.snapshot_id
    assert retrieved.run_id == run_id
    assert retrieved.wave_number == 0
    assert retrieved.full_state is not None
    assert "node1" in retrieved.full_state
    assert retrieved.full_state["node1"].value == 1  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_get_snapshots_range(
    postgres_backend: PostgresCheckpointBackend,
) -> None:
    """Test retrieving range of snapshots."""
    run_id = generate_run_id()

    for i in range(3):
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=i,
            full_state={"node1": SampleState(value=i, name=f"wave_{i}")},
            state_hash=f"hash{i}",
            next_frontier=["node2"],
            routing_ended=False,
        )
        await postgres_backend.save_state_snapshot(snapshot)

    snapshots = await postgres_backend.get_snapshots_range(run_id, 0, 2, "ASC")

    assert len(snapshots) == 3
    assert snapshots[0].wave_number == 0
    assert snapshots[1].wave_number == 1
    assert snapshots[2].wave_number == 2


@pytest.mark.asyncio
async def test_save_and_get_trace(
    postgres_backend: PostgresCheckpointBackend,
) -> None:
    """Test saving and retrieving execution traces."""
    run_id = generate_run_id()
    snapshot_id = generate_snapshot_id()

    snapshot = StateSnapshot(
        snapshot_id=snapshot_id,
        run_id=run_id,
        wave_number=0,
        full_state={"node1": SampleState(value=1, name="test")},
        state_hash="hash123",
        next_frontier=["node2"],
        routing_ended=False,
    )
    await postgres_backend.save_state_snapshot(snapshot)

    node_trace = NodeExecutionTrace(
        log_id="log1",
        node_id="node1",
        wave_number=0,
        snapshot_id=snapshot_id,
        input_ref=StateRef(snapshot_id=snapshot_id, state_key="node1"),
        output_ref=StateRef(snapshot_id=snapshot_id, state_key="node1"),
        event_log_id="log123",
        total_events=5,
        event_summary=EventSummary(
            total_events=5,
            token_count=100,
            tool_call_count=1,
            cache_hits=0,
            tool_calls=["tool1"],
        ),
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        next_nodes=[],
    )

    trace = ExecutionTrace(
        trace_id=str(uuid.uuid4()),
        run_id=run_id,
        wave_number=0,
        checkpoint_snapshot_id=snapshot_id,
        node_traces=[node_trace],
        parallel_batch_id="batch1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )

    await postgres_backend.save_trace(trace)
    retrieved = await postgres_backend.get_trace(run_id, 0)

    assert retrieved is not None
    assert retrieved.trace_id == trace.trace_id
    assert len(retrieved.node_traces) == 1
    assert retrieved.node_traces[0].log_id == "log1"


@pytest.mark.asyncio
async def test_list_runs(
    postgres_backend: PostgresCheckpointBackend,
) -> None:
    """Test listing runs."""
    metadata = RunMetadata(
        run_id=generate_run_id(),
        flow_id="test_flow",
        started_at=datetime.now(UTC),
        status=RunMetadata.Status.COMPLETED,
        total_waves=3,
    )

    await postgres_backend.save_run_metadata(metadata)
    runs = await postgres_backend.list_runs(limit=10)

    assert len(runs) == 1
    assert runs[0].run_id == metadata.run_id


@pytest.mark.asyncio
async def test_delete_run(
    postgres_backend: PostgresCheckpointBackend,
) -> None:
    """Test deleting a run."""
    run_id = generate_run_id()

    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_flow",
        started_at=datetime.now(UTC),
        status=RunMetadata.Status.COMPLETED,
        total_waves=1,
    )
    await postgres_backend.save_run_metadata(metadata)

    snapshot = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=0,
        full_state={},
        state_hash="hash",
        next_frontier=[],
        routing_ended=True,
    )
    await postgres_backend.save_state_snapshot(snapshot)

    await postgres_backend.delete_run(run_id, keep_checkpoints=False)

    retrieved_metadata = await postgres_backend.get_run_metadata(run_id)
    assert retrieved_metadata is None

    retrieved_snapshot = await postgres_backend.get_state_snapshot(run_id, 0)
    assert retrieved_snapshot is None


@pytest.mark.asyncio
async def test_connection_pool_config(mock_asyncpg: AsyncMock) -> None:
    """Test connection pool configuration."""
    config = PostgresCheckpointConfig(
        connection_string="postgresql://test:test@localhost:5432/test",
        min_pool_size=5,
        max_pool_size=20,
        timeout=15.0,
    )
    backend = PostgresCheckpointBackend(config)

    with patch("asyncpg.create_pool", mock_asyncpg.create_pool):
        await backend.initialize()

    assert backend.pool is not None
    pool_closed = backend.pool.closed
    await backend.close()
    assert pool_closed is False


@pytest.mark.asyncio
async def test_update_state_snapshot(
    postgres_backend: PostgresCheckpointBackend,
) -> None:
    """Test updating existing state snapshot."""
    run_id = generate_run_id()
    snapshot_id = generate_snapshot_id()

    snapshot = StateSnapshot(
        snapshot_id=snapshot_id,
        run_id=run_id,
        wave_number=0,
        full_state={"node1": SampleState(value=1, name="test")},
        state_hash="hash123",
        next_frontier=["node2"],
        routing_ended=False,
    )
    await postgres_backend.save_state_snapshot(snapshot)

    snapshot.trace_id = "new_trace_id"
    await postgres_backend.update_state_snapshot(snapshot)

    retrieved = await postgres_backend.get_state_snapshot(run_id, 0)
    assert retrieved is not None
    assert retrieved.trace_id == "new_trace_id"
