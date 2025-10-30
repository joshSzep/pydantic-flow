"""Tests for S3CheckpointStore implementation with mocked aioboto3."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
from pydantic_flow.hitl.checkpoints.s3 import S3CheckpointStore
from pydantic_flow.hitl.checkpoints.s3 import S3CheckpointStoreConfig
from tests.test_checkpoints_conformance import CheckpointStoreConformanceTests


class MockS3Client:
    """Mock S3 client for testing."""

    class Exceptions:
        """Mock S3 exceptions."""

        class NoSuchKey(Exception):
            """Mock NoSuchKey exception."""

            pass

    def __init__(self) -> None:
        """Initialize mock S3 with in-memory storage."""
        self._objects: dict[str, bytes] = {}
        self._metadata: dict[str, dict[str, object]] = {}
        self.exceptions = self.Exceptions()

    async def put_object(
        self,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str | None = None,
    ) -> dict[str, object]:
        """Put object to S3."""
        self._objects[Key] = Body
        self._metadata[Key] = {
            "LastModified": datetime.now(UTC),
            "Size": len(Body),
        }
        return {"ETag": '"mock-etag"'}

    async def get_object(self, Bucket: str, Key: str) -> dict[str, object]:
        """Get object from S3."""
        if Key not in self._objects:
            raise self.exceptions.NoSuchKey(f"NoSuchKey: {Key}")

        body_mock = AsyncMock()
        body_mock.read = AsyncMock(return_value=self._objects[Key])
        return {"Body": body_mock}

    async def head_object(self, Bucket: str, Key: str) -> dict[str, object]:
        """Check if object exists."""
        if Key not in self._objects:
            raise self.exceptions.NoSuchKey(f"NoSuchKey: {Key}")
        return {"ETag": '"mock-etag"'}

    async def head_bucket(self, Bucket: str) -> dict[str, object]:
        """Check if bucket exists."""
        return {}

    async def delete_object(self, Bucket: str, Key: str) -> dict[str, object]:
        """Delete object from S3."""
        if Key in self._objects:
            del self._objects[Key]
            del self._metadata[Key]
        return {"DeleteMarker": False}

    async def list_objects_v2(
        self,
        Bucket: str,
        Prefix: str | None = None,
        MaxKeys: int | None = None,
        ContinuationToken: str | None = None,
    ) -> dict[str, object]:
        """List objects in S3 with pagination support."""
        # Get all keys matching prefix
        all_keys = sorted(
            k for k in self._objects if not Prefix or k.startswith(Prefix)
        )

        # Apply continuation token (skip to after this key)
        if ContinuationToken:
            try:
                start_index = all_keys.index(ContinuationToken) + 1
                all_keys = all_keys[start_index:]
            except ValueError:
                # Token not found, start from beginning
                pass

        # Apply MaxKeys limit
        is_truncated = False
        next_token = None
        if MaxKeys and len(all_keys) > MaxKeys:
            all_keys = all_keys[:MaxKeys]
            is_truncated = True
            # Next continuation token is the last key we returned
            next_token = all_keys[-1]

        contents = [
            {
                "Key": key,
                "LastModified": self._metadata[key]["LastModified"],
                "Size": self._metadata[key]["Size"],
            }
            for key in all_keys
        ]

        result: dict[str, object] = {
            "Contents": contents if contents else [],
            "KeyCount": len(contents),
            "IsTruncated": is_truncated,
        }

        if next_token:
            result["NextContinuationToken"] = next_token

        return result

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        pass


@pytest.fixture
def mock_s3_client() -> MockS3Client:
    """Create a mock S3 client."""
    return MockS3Client()


@pytest.fixture
def mock_s3_session(mock_s3_client: MockS3Client):
    """Create a mock aioboto3 session."""

    class MockSession:
        def __init__(self):
            self.mock_client = mock_s3_client

        def client(self, service_name, **kwargs):
            return self.mock_client

    return MockSession()


@pytest.fixture
def mock_aioboto3(mock_s3_session):
    """Create a mock aioboto3 module."""
    import sys

    class MockAioboto3:
        def __init__(self, session):
            self.session = session

        def Session(self):
            return self.session

    mock = MockAioboto3(mock_s3_session)
    sys.modules["aioboto3"] = mock  # type: ignore[assignment]
    yield mock
    # Cleanup
    if "aioboto3" in sys.modules:
        del sys.modules["aioboto3"]


class TestS3CheckpointStoreConformance(CheckpointStoreConformanceTests):
    """Conformance tests for S3CheckpointStore with mocked aioboto3."""

    @pytest.fixture
    async def store(
        self, mock_s3_client: MockS3Client, mock_aioboto3
    ) -> S3CheckpointStore:
        """Create an S3CheckpointStore with mocked aioboto3 client."""
        config = S3CheckpointStoreConfig(
            bucket="test-bucket",
            key_prefix="test/checkpoints",
            region_name="us-east-1",
        )
        store = S3CheckpointStore(config)
        await store.healthcheck()
        store._client = mock_s3_client
        return store


class TestS3CheckpointStoreSpecific:
    """Tests specific to S3CheckpointStore implementation."""

    @pytest.fixture
    async def store(
        self, mock_s3_client: MockS3Client, mock_aioboto3
    ) -> S3CheckpointStore:
        """Create an S3CheckpointStore with mocked aioboto3 client."""
        config = S3CheckpointStoreConfig(
            bucket="test-bucket",
            key_prefix="test/checkpoints",
            region_name="us-east-1",
        )
        store = S3CheckpointStore(config)
        await store.healthcheck()
        store._client = mock_s3_client
        return store

    @pytest.mark.asyncio
    async def test_key_prefix_applied(
        self, store: S3CheckpointStore, sample_envelope
    ) -> None:
        """Test that key prefix is correctly applied."""
        await store.save(sample_envelope)

        object_key = store._object_key(sample_envelope.run_id, sample_envelope.id)
        assert object_key.startswith("test/checkpoints/runs/")

    @pytest.mark.asyncio
    async def test_endpoint_url_configuration(self) -> None:
        """Test that custom endpoint URL can be configured."""
        config = S3CheckpointStoreConfig(
            bucket="test-bucket",
            endpoint_url="http://localhost:9000",
            region_name="us-east-1",
        )
        store = S3CheckpointStore(config)

        assert store.config.endpoint_url == "http://localhost:9000"

    @pytest.mark.asyncio
    async def test_bucket_configuration(
        self, mock_s3_client: MockS3Client, mock_aioboto3
    ) -> None:
        """Test that bucket name is configurable."""
        config = S3CheckpointStoreConfig(
            bucket="custom-bucket",
            key_prefix="ckpts",
        )
        store = S3CheckpointStore(config)
        await store.healthcheck()
        store._client = mock_s3_client

        assert store.config.bucket == "custom-bucket"

    @pytest.mark.asyncio
    async def test_concurrent_operations(
        self, store: S3CheckpointStore, sample_checkpoint
    ) -> None:
        """Test concurrent S3 operations."""
        import asyncio

        from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
        from pydantic_flow.hitl.checkpoints.interface import CheckpointId
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
    async def test_object_key_format(self, store: S3CheckpointStore) -> None:
        """Test that object keys follow expected format."""
        from pydantic_flow.hitl.checkpoints.interface import CheckpointId
        from pydantic_flow.hitl.checkpoints.interface import RunId

        run_id = RunId("test_run")
        checkpoint_id = CheckpointId("ckpt_123")

        key = store._object_key(run_id, checkpoint_id)

        assert key == "test/checkpoints/runs/test_run/ckpt_123.json"
        assert key.endswith(".json")

    @pytest.mark.asyncio
    async def test_connection_lazy_initialization(self) -> None:
        """Test that S3 client is lazily initialized."""
        config = S3CheckpointStoreConfig(bucket="test-bucket")
        store = S3CheckpointStore(config)

        assert store._client is None
