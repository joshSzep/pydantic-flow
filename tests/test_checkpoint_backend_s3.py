"""Tests for S3 checkpoint backend v2."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import patch
import uuid

from pydantic import BaseModel
import pytest

from pydantic_flow.checkpoints.backends.s3 import S3CheckpointBackend
from pydantic_flow.checkpoints.backends.s3 import S3CheckpointConfig
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


class MockS3Object:
    """Mock S3 object response."""

    def __init__(self, body: bytes):
        """Initialize mock S3 object."""
        self.body = body

    async def read(self) -> bytes:
        """Read object body."""
        return self.body


class MockS3Client:
    """Mock aioboto3 S3 client."""

    def __init__(self) -> None:
        """Initialize mock client."""
        self._objects: dict[str, bytes] = {}
        self.closed = False

    async def __aenter__(self) -> MockS3Client:
        """Context manager entry."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Context manager exit."""
        self.closed = True

    async def head_bucket(self, Bucket: str) -> dict:
        """Mock head_bucket."""
        return {}

    async def put_object(self, **kwargs: object) -> dict:
        """Mock put_object."""
        key = kwargs["Key"]
        body = kwargs["Body"]

        if isinstance(body, bytes):
            self._objects[key] = body  # type: ignore[index]
        else:
            self._objects[key] = body.encode("utf-8")  # type: ignore[index, union-attr]

        return {"ETag": "mock-etag"}

    async def get_object(self, Bucket: str, Key: str) -> dict:
        """Mock get_object."""
        if Key not in self._objects:
            raise Exception("NoSuchKey")

        return {"Body": MockS3Object(self._objects[Key])}

    async def delete_object(self, Bucket: str, Key: str) -> dict:
        """Mock delete_object."""
        if Key in self._objects:
            del self._objects[Key]
        return {}

    async def list_objects_v2(self, Bucket: str, Prefix: str) -> dict:
        """Mock list_objects_v2."""
        matching_keys = [key for key in self._objects if key.startswith(Prefix)]

        if not matching_keys:
            return {}

        contents = [
            {"Key": key, "LastModified": datetime.now(UTC)} for key in matching_keys
        ]

        return {"Contents": contents}


@pytest.fixture
async def mock_aioboto3() -> AsyncMock:
    """Mock aioboto3 module."""
    mock_client = MockS3Client()

    class MockSession:
        def client(self, *args: object, **kwargs: object) -> MockS3Client:
            return mock_client

    mock_aioboto3 = AsyncMock()
    mock_aioboto3.Session = MockSession
    return mock_aioboto3


@pytest.fixture
async def s3_backend(
    mock_aioboto3: AsyncMock,
) -> AsyncGenerator[S3CheckpointBackend]:
    """Create S3 backend with mocked aioboto3."""
    config = S3CheckpointConfig(bucket="test-bucket", key_prefix="test-prefix")
    backend = S3CheckpointBackend(config)

    with patch("aioboto3.Session", mock_aioboto3.Session):
        await backend.initialize()

    yield backend
    await backend.close()


@pytest.mark.asyncio
async def test_s3_healthcheck(s3_backend: S3CheckpointBackend) -> None:
    """Test S3 healthcheck."""
    health = await s3_backend.healthcheck()
    assert health is True


@pytest.mark.asyncio
async def test_save_and_get_snapshot(
    s3_backend: S3CheckpointBackend,
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

    await s3_backend.save_state_snapshot(snapshot)
    retrieved = await s3_backend.get_state_snapshot(run_id, 0)

    assert retrieved is not None
    assert retrieved.snapshot_id == snapshot.snapshot_id
    assert retrieved.run_id == run_id
    assert retrieved.wave_number == 0
    assert retrieved.full_state is not None
    assert "node1" in retrieved.full_state
    assert retrieved.full_state["node1"].value == 1  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_get_snapshots_range(
    s3_backend: S3CheckpointBackend,
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
        await s3_backend.save_state_snapshot(snapshot)

    snapshots = await s3_backend.get_snapshots_range(run_id, 0, 2, "ASC")

    assert len(snapshots) == 3
    assert snapshots[0].wave_number == 0
    assert snapshots[1].wave_number == 1
    assert snapshots[2].wave_number == 2


@pytest.mark.asyncio
async def test_save_and_get_trace(
    s3_backend: S3CheckpointBackend,
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
    await s3_backend.save_state_snapshot(snapshot)

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

    await s3_backend.save_trace(trace)
    retrieved = await s3_backend.get_trace(run_id, 0)

    assert retrieved is not None
    assert retrieved.trace_id == trace.trace_id
    assert len(retrieved.node_traces) == 1
    assert retrieved.node_traces[0].log_id == "log1"


@pytest.mark.asyncio
async def test_list_runs(s3_backend: S3CheckpointBackend) -> None:
    """Test listing runs."""
    metadata = RunMetadata(
        run_id=generate_run_id(),
        flow_id="test_flow",
        started_at=datetime.now(UTC),
        status=RunMetadata.Status.COMPLETED,
        total_waves=3,
    )

    await s3_backend.save_run_metadata(metadata)
    runs = await s3_backend.list_runs(limit=10)

    assert len(runs) == 1
    assert runs[0].run_id == metadata.run_id


@pytest.mark.asyncio
async def test_delete_run(s3_backend: S3CheckpointBackend) -> None:
    """Test deleting a run."""
    run_id = generate_run_id()

    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_flow",
        started_at=datetime.now(UTC),
        status=RunMetadata.Status.COMPLETED,
        total_waves=1,
    )
    await s3_backend.save_run_metadata(metadata)

    snapshot = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=0,
        full_state={},
        state_hash="hash",
        next_frontier=[],
        routing_ended=True,
    )
    await s3_backend.save_state_snapshot(snapshot)

    await s3_backend.delete_run(run_id, keep_checkpoints=False)

    retrieved_metadata = await s3_backend.get_run_metadata(run_id)
    assert retrieved_metadata is None

    retrieved_snapshot = await s3_backend.get_state_snapshot(run_id, 0)
    assert retrieved_snapshot is None


@pytest.mark.asyncio
async def test_s3_config_options(mock_aioboto3: AsyncMock) -> None:
    """Test S3 configuration options."""
    config = S3CheckpointConfig(
        bucket="test-bucket",
        key_prefix="custom-prefix",
        region_name="us-west-2",
        server_side_encryption=True,
        storage_class="GLACIER",
        compress_level=9,
    )
    backend = S3CheckpointBackend(config)

    with patch("aioboto3.Session", mock_aioboto3.Session):
        await backend.initialize()

    assert backend.config.bucket == "test-bucket"
    assert backend.config.key_prefix == "custom-prefix"
    assert backend.config.server_side_encryption is True
    assert backend.config.storage_class == "GLACIER"

    await backend.close()


@pytest.mark.asyncio
async def test_update_state_snapshot(
    s3_backend: S3CheckpointBackend,
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
    await s3_backend.save_state_snapshot(snapshot)

    snapshot.trace_id = "new_trace_id"
    await s3_backend.update_state_snapshot(snapshot)

    retrieved = await s3_backend.get_state_snapshot(run_id, 0)
    assert retrieved is not None
    assert retrieved.trace_id == "new_trace_id"
