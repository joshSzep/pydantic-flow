"""S3 storage backend for checkpoint v2.

This module provides an S3-based implementation of the checkpoint storage
backend, optimized for cold storage, archival, and cross-region replication.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from pydantic_flow.checkpoints.types import ExecutionTrace
from pydantic_flow.checkpoints.types import NodeExecutionTrace
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateSnapshot


class S3CheckpointConfig(BaseModel):
    """Configuration for S3 checkpoint backend.

    Attributes:
        bucket: S3 bucket name.
        key_prefix: Prefix for all S3 keys.
        endpoint_url: Optional S3-compatible endpoint URL (for MinIO, etc).
        region_name: AWS region name.
        server_side_encryption: Enable server-side encryption (AES256).
        storage_class: S3 storage class (STANDARD, GLACIER, etc).
        compress_level: Compression level for state data (1-9).

    """

    bucket: str
    key_prefix: str = "checkpoints"
    endpoint_url: str | None = None
    region_name: str = "us-east-1"
    server_side_encryption: bool = True
    storage_class: str = "STANDARD"
    compress_level: int = Field(default=6, ge=1, le=9)


class S3CheckpointBackend:
    """S3 backend for checkpoint storage.

    Optimized for cold storage and archival with server-side encryption.
    Provides cross-region replication capabilities and lifecycle policies.

    Storage Layout:
        {key_prefix}/{run_id}/snapshots/{wave_number}.msgpack
        {key_prefix}/{run_id}/traces/{wave_number}.msgpack
        {key_prefix}/{run_id}/node_traces/{log_id}.msgpack
        {key_prefix}/{run_id}/metadata.json

    Production Features:
        - Server-side encryption (AES256)
        - S3 storage classes (STANDARD, GLACIER, etc)
        - Cross-region replication support
        - Lifecycle policies for cold storage
        - Content-type and metadata tagging

    Args:
        config: S3 backend configuration.

    """

    def __init__(self, config: S3CheckpointConfig):
        """Initialize S3 backend.

        Args:
            config: Backend configuration.

        """
        self.config = config
        self._client: Any | None = None

    async def initialize(self) -> None:
        """Initialize S3 client."""
        try:
            import aioboto3
        except ImportError as e:
            msg = (
                "aioboto3 is required for S3 backend. "
                "Install with: pip install aioboto3"
            )
            raise ImportError(msg) from e

        session = aioboto3.Session()
        self._client = await session.client(
            "s3",
            endpoint_url=self.config.endpoint_url,
            region_name=self.config.region_name,
        ).__aenter__()

    async def close(self) -> None:
        """Close S3 client."""
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None

    async def healthcheck(self) -> bool:
        """Check S3 bucket health.

        Returns:
            True if bucket is accessible.

        """
        if not self._client:
            return False

        try:
            await self._client.head_bucket(Bucket=self.config.bucket)
            return True
        except Exception:
            return False

    def _snapshot_key(self, run_id: RunId, wave_number: int) -> str:
        """Generate S3 key for snapshot."""
        return f"{self.config.key_prefix}/{run_id}/snapshots/{wave_number}.msgpack"

    def _trace_key(self, run_id: RunId, wave_number: int) -> str:
        """Generate S3 key for trace."""
        return f"{self.config.key_prefix}/{run_id}/traces/{wave_number}.msgpack"

    def _node_trace_key(self, run_id: RunId, log_id: str) -> str:
        """Generate S3 key for node trace."""
        return f"{self.config.key_prefix}/{run_id}/node_traces/{log_id}.msgpack"

    def _metadata_key(self, run_id: RunId) -> str:
        """Generate S3 key for run metadata."""
        return f"{self.config.key_prefix}/{run_id}/metadata.json"

    async def save_run_metadata(self, metadata: RunMetadata) -> None:
        """Save run metadata."""
        if not self._client:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        metadata_json = metadata.model_dump_json()

        put_kwargs: dict[str, Any] = {
            "Bucket": self.config.bucket,
            "Key": self._metadata_key(metadata.run_id),
            "Body": metadata_json.encode("utf-8"),
            "ContentType": "application/json",
            "StorageClass": self.config.storage_class,
        }

        if self.config.server_side_encryption:
            put_kwargs["ServerSideEncryption"] = "AES256"

        await self._client.put_object(**put_kwargs)

    async def get_run_metadata(self, run_id: RunId) -> RunMetadata | None:
        """Retrieve run metadata."""
        if not self._client:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        try:
            response = await self._client.get_object(
                Bucket=self.config.bucket, Key=self._metadata_key(run_id)
            )
            body = await response["Body"].read()
            return RunMetadata.model_validate_json(body.decode("utf-8"))
        except Exception as e:
            if "NoSuchKey" in str(e):
                return None
            raise

    async def save_state_snapshot(self, snapshot: StateSnapshot) -> None:
        """Save state snapshot."""
        if not self._client:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        data_compressed = snapshot.serialize()

        put_kwargs: dict[str, Any] = {
            "Bucket": self.config.bucket,
            "Key": self._snapshot_key(snapshot.run_id, snapshot.wave_number),
            "Body": data_compressed,
            "ContentType": "application/x-msgpack",
            "StorageClass": self.config.storage_class,
            "Metadata": {
                "snapshot_id": snapshot.snapshot_id,
                "wave_number": str(snapshot.wave_number),
                "state_hash": snapshot.state_hash,
            },
        }

        if self.config.server_side_encryption:
            put_kwargs["ServerSideEncryption"] = "AES256"

        await self._client.put_object(**put_kwargs)

    async def get_state_snapshot(
        self, run_id: RunId, wave_number: int
    ) -> StateSnapshot | None:
        """Retrieve state snapshot."""
        if not self._client:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        try:
            response = await self._client.get_object(
                Bucket=self.config.bucket,
                Key=self._snapshot_key(run_id, wave_number),
            )
            body = await response["Body"].read()
            return StateSnapshot.deserialize(body)
        except Exception as e:
            if "NoSuchKey" in str(e):
                return None
            raise

    async def update_state_snapshot(self, snapshot: StateSnapshot) -> None:
        """Update existing state snapshot."""
        await self.save_state_snapshot(snapshot)

    async def get_snapshots_range(
        self,
        run_id: RunId,
        start_wave: int,
        end_wave: int,
        order: Literal["ASC", "DESC"] = "ASC",
    ) -> list[StateSnapshot]:
        """Retrieve range of snapshots for state reconstruction."""
        if not self._client:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        if order not in ("ASC", "DESC"):
            msg = f"Invalid order: {order}"
            raise ValueError(msg)

        snapshots: list[StateSnapshot] = []

        for wave in range(start_wave, end_wave + 1):
            snapshot = await self.get_state_snapshot(run_id, wave)
            if snapshot:
                snapshots.append(snapshot)

        if order == "DESC":
            snapshots.reverse()

        return snapshots

    async def save_trace(self, trace: ExecutionTrace) -> None:
        """Save execution trace with checkpoint validation."""
        if not self._client:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        from pydantic_flow.checkpoints.serialization import TypedSerializer
        from pydantic_flow.checkpoints.serialization import compress

        snapshot_exists = await self.get_state_snapshot(trace.run_id, trace.wave_number)
        if not snapshot_exists:
            msg = f"Invalid checkpoint reference: {trace.checkpoint_snapshot_id}"
            raise ValueError(msg)

        data = TypedSerializer.serialize(trace)
        data_compressed = compress(data, level=self.config.compress_level)

        put_kwargs: dict[str, Any] = {
            "Bucket": self.config.bucket,
            "Key": self._trace_key(trace.run_id, trace.wave_number),
            "Body": data_compressed,
            "ContentType": "application/x-msgpack",
            "StorageClass": self.config.storage_class,
            "Metadata": {
                "trace_id": trace.trace_id,
                "wave_number": str(trace.wave_number),
            },
        }

        if self.config.server_side_encryption:
            put_kwargs["ServerSideEncryption"] = "AES256"

        await self._client.put_object(**put_kwargs)

        for node_trace in trace.node_traces:
            await self.save_node_trace(node_trace)

    async def get_trace(self, run_id: RunId, wave_number: int) -> ExecutionTrace | None:
        """Retrieve execution trace."""
        if not self._client:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        try:
            response = await self._client.get_object(
                Bucket=self.config.bucket,
                Key=self._trace_key(run_id, wave_number),
            )
            body = await response["Body"].read()

            from pydantic_flow.checkpoints.serialization import TypedSerializer
            from pydantic_flow.checkpoints.serialization import decompress

            decompressed = decompress(body)
            return TypedSerializer.deserialize(decompressed)
        except Exception as e:
            if "NoSuchKey" in str(e):
                return None
            raise

    async def delete_trace(self, run_id: RunId, wave_number: int) -> bool:
        """Delete execution trace."""
        if not self._client:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        try:
            await self._client.delete_object(
                Bucket=self.config.bucket,
                Key=self._trace_key(run_id, wave_number),
            )
            return True
        except Exception as e:
            if "NoSuchKey" in str(e):
                return False
            raise

    async def save_node_trace(self, node_trace: NodeExecutionTrace) -> None:
        """Save node execution trace."""
        if not self._client:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        from pydantic_flow.checkpoints.serialization import TypedSerializer
        from pydantic_flow.checkpoints.serialization import compress

        data = TypedSerializer.serialize(node_trace)
        data_compressed = compress(data, level=self.config.compress_level)

        temp_run_id = RunId("temp_run")

        put_kwargs: dict[str, Any] = {
            "Bucket": self.config.bucket,
            "Key": self._node_trace_key(temp_run_id, node_trace.log_id),
            "Body": data_compressed,
            "ContentType": "application/x-msgpack",
            "StorageClass": self.config.storage_class,
            "Metadata": {
                "log_id": node_trace.log_id,
                "node_id": node_trace.node_id,
            },
        }

        if self.config.server_side_encryption:
            put_kwargs["ServerSideEncryption"] = "AES256"

        await self._client.put_object(**put_kwargs)

    async def get_node_trace(self, log_id: str) -> NodeExecutionTrace | None:
        """Retrieve node execution trace."""
        if not self._client:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        prefix = f"{self.config.key_prefix}/"

        try:
            response = await self._client.list_objects_v2(
                Bucket=self.config.bucket, Prefix=prefix
            )

            if "Contents" not in response:
                return None

            for obj in response["Contents"]:
                if log_id in obj["Key"]:
                    obj_response = await self._client.get_object(
                        Bucket=self.config.bucket, Key=obj["Key"]
                    )
                    body = await obj_response["Body"].read()

                    from pydantic_flow.checkpoints.serialization import TypedSerializer
                    from pydantic_flow.checkpoints.serialization import decompress

                    decompressed = decompress(body)
                    return TypedSerializer.deserialize(decompressed)

            return None
        except Exception as e:
            if "NoSuchKey" in str(e):
                return None
            raise

    async def list_runs(
        self,
        *,
        before: datetime | None = None,
        after: datetime | None = None,
        limit: int | None = None,
    ) -> list[RunMetadata]:
        """List runs with optional filtering."""
        if not self._client:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        prefix = f"{self.config.key_prefix}/"

        response = await self._client.list_objects_v2(
            Bucket=self.config.bucket, Prefix=prefix
        )

        if "Contents" not in response:
            return []

        runs: list[RunMetadata] = []

        for obj in response["Contents"]:
            if obj["Key"].endswith("metadata.json"):
                try:
                    obj_response = await self._client.get_object(
                        Bucket=self.config.bucket, Key=obj["Key"]
                    )
                    body = await obj_response["Body"].read()
                    metadata = RunMetadata.model_validate_json(body.decode("utf-8"))

                    if before and metadata.started_at >= before:
                        continue
                    if after and metadata.started_at <= after:
                        continue

                    runs.append(metadata)
                except Exception:
                    continue

        runs.sort(key=lambda x: x.started_at, reverse=True)

        if limit:
            runs = runs[:limit]

        return runs

    async def delete_run(
        self, run_id: RunId, *, keep_checkpoints: bool = False
    ) -> None:
        """Delete all data for a run."""
        if not self._client:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        prefix = f"{self.config.key_prefix}/{run_id}/"

        response = await self._client.list_objects_v2(
            Bucket=self.config.bucket, Prefix=prefix
        )

        if "Contents" not in response:
            return

        for obj in response["Contents"]:
            key = obj["Key"]

            if keep_checkpoints and "/snapshots/" in key:
                continue

            await self._client.delete_object(Bucket=self.config.bucket, Key=key)

    async def append_events_batch(
        self,
        log_id: str,
        events: list[Any],
        start_sequence: int,
    ) -> None:
        """Append batch of events to event log.

        Note: This is a placeholder implementation for Phase 3.
        Full event storage will be implemented when needed.

        Args:
            log_id: Event log identifier.
            events: List of progress items to append.
            start_sequence: Starting sequence number for this batch.

        """
        pass

    async def stream_events(
        self,
        log_id: str,
        start_offset: int = 0,
        end_offset: int | None = None,
    ) -> list[Any]:
        """Stream events from event log.

        Note: This is a placeholder implementation for Phase 3.
        Full event retrieval will be implemented when needed.

        Args:
            log_id: Event log identifier.
            start_offset: Starting offset for event stream.
            end_offset: Optional ending offset.

        Returns:
            Empty list (placeholder).

        """
        return []
