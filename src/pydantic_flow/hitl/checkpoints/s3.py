"""S3-compatible checkpoint store implementation.

Stores checkpoints as JSON objects in S3-compatible object storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from pydantic_flow.hitl.checkpoints.base import BaseCheckpointStore
from pydantic_flow.hitl.checkpoints.interface import CheckpointBackendError
from pydantic_flow.hitl.checkpoints.interface import CheckpointConflict
from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
from pydantic_flow.hitl.checkpoints.interface import CheckpointId
from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
from pydantic_flow.hitl.checkpoints.interface import RunId
from pydantic_flow.hitl.checkpoints.interface import SortOrder
from pydantic_flow.hitl.checkpoints.serde import deserialize_checkpoint
from pydantic_flow.hitl.checkpoints.serde import serialize_checkpoint

if TYPE_CHECKING:
    pass


class S3CheckpointStoreConfig(BaseModel):
    """Configuration for S3-compatible checkpoint store.

    Attributes:
        bucket: S3 bucket name.
        key_prefix: Prefix for all S3 keys.
        endpoint_url: Optional S3-compatible endpoint URL.
        region_name: AWS region name.

    """

    bucket: str
    key_prefix: str = "checkpoints"
    endpoint_url: str | None = None
    region_name: str = "us-east-1"


class S3CheckpointStore(BaseCheckpointStore):
    """S3-compatible checkpoint store.

    Note: This is a simplified implementation. Full production implementation
    would include pointer objects for latest, content encoding, and retry logic.
    """

    def __init__(self, config: S3CheckpointStoreConfig) -> None:
        """Initialize the S3 store.

        Args:
            config: Store configuration.

        """
        self.config = config
        self._client = None

    async def _get_client(self):
        """Get or create S3 client."""
        if self._client is None:
            try:
                import aioboto3  # noqa: PLC0415
            except ImportError as e:
                msg = "aioboto3 is required for S3CheckpointStore"
                raise CheckpointBackendError(msg, cause=e) from e

            session = aioboto3.Session()
            self._client = await session.client(
                "s3",
                endpoint_url=self.config.endpoint_url,
                region_name=self.config.region_name,
            ).__aenter__()
        return self._client

    def _object_key(self, run_id: RunId, checkpoint_id: CheckpointId) -> str:
        """Generate S3 key for checkpoint object."""
        return f"{self.config.key_prefix}/runs/{run_id}/{checkpoint_id}.json"

    async def _do_save(
        self, envelope: CheckpointEnvelope, overwrite: bool
    ) -> CheckpointEnvelope:
        """Save checkpoint to S3.

        Args:
            envelope: The prepared checkpoint envelope with computed hash.
            overwrite: If False, raise CheckpointConflict if key exists.

        Returns:
            The saved envelope.

        Raises:
            CheckpointConflict: If checkpoint exists and overwrite=False.

        """
        client = await self._get_client()

        json_str = serialize_checkpoint(envelope)
        object_key = self._object_key(envelope.run_id, envelope.id)

        if not overwrite:
            try:
                await client.head_object(Bucket=self.config.bucket, Key=object_key)
                msg = (
                    f"Checkpoint {envelope.id} already exists for run {envelope.run_id}"
                )
                raise CheckpointConflict(msg)
            except client.exceptions.NoSuchKey:
                pass

        await client.put_object(
            Bucket=self.config.bucket,
            Key=object_key,
            Body=json_str.encode("utf-8"),
            ContentType="application/json",
        )

        return envelope

    async def _do_latest(
        self, run_id: RunId, node_id: str | None = None
    ) -> CheckpointEnvelope | None:
        """Get the most recent checkpoint from S3.

        Args:
            run_id: The run to query.
            node_id: Optional node filter.

        Returns:
            The latest checkpoint envelope, or None if not found.

        """
        client = await self._get_client()
        prefix = f"{self.config.key_prefix}/runs/{run_id}/"

        response = await client.list_objects_v2(
            Bucket=self.config.bucket, Prefix=prefix
        )

        if "Contents" not in response or not response["Contents"]:
            return None

        objects = response["Contents"]

        if node_id is not None:
            checkpoints: list[tuple[dict, CheckpointEnvelope]] = []
            for obj in objects:
                obj_response = await client.get_object(
                    Bucket=self.config.bucket, Key=obj["Key"]
                )
                body = await obj_response["Body"].read()
                envelope = deserialize_checkpoint(body.decode("utf-8"))
                if envelope.node_id == node_id:
                    checkpoints.append((obj, envelope))

            if not checkpoints:
                return None

            _, latest_envelope = max(
                checkpoints, key=lambda pair: pair[0]["LastModified"]
            )
            return latest_envelope

        latest_key = max(objects, key=lambda obj: obj["LastModified"])
        obj_response = await client.get_object(
            Bucket=self.config.bucket, Key=latest_key["Key"]
        )
        body = await obj_response["Body"].read()
        return deserialize_checkpoint(body.decode("utf-8"))

    async def _do_get(
        self, run_id: RunId, checkpoint_id: CheckpointId
    ) -> CheckpointEnvelope | None:
        """Get a specific checkpoint by ID from S3.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            The checkpoint envelope, or None if not found.

        """
        client = await self._get_client()
        object_key = self._object_key(run_id, checkpoint_id)

        try:
            response = await client.get_object(
                Bucket=self.config.bucket, Key=object_key
            )
            body = await response["Body"].read()
            return deserialize_checkpoint(body.decode("utf-8"))
        except Exception as e:
            if "NoSuchKey" in str(e):
                return None
            raise

    async def _do_list(
        self, query: CheckpointQuery
    ) -> tuple[list[CheckpointEnvelope], str | None]:
        """List checkpoints matching query criteria from S3.

        Args:
            query: Query parameters for filtering and pagination.

        Returns:
            Tuple of (list of checkpoint envelopes, next cursor or None).

        """
        if query.run_id is None:
            return [], None

        client = await self._get_client()
        prefix = f"{self.config.key_prefix}/runs/{query.run_id}/"

        list_params: dict[str, object] = {
            "Bucket": self.config.bucket,
            "Prefix": prefix,
            "MaxKeys": query.limit + 1,
        }

        if query.cursor:
            list_params["ContinuationToken"] = query.cursor

        response = await client.list_objects_v2(**list_params)

        if "Contents" not in response:
            return [], None

        all_envelopes: list[CheckpointEnvelope] = []
        for obj in response["Contents"]:
            obj_response = await client.get_object(
                Bucket=self.config.bucket, Key=obj["Key"]
            )
            body = await obj_response["Body"].read()
            envelope = deserialize_checkpoint(body.decode("utf-8"))

            if (
                (query.node_id is None or envelope.node_id == query.node_id)
                and (query.since is None or envelope.created_at >= query.since)
                and (query.until is None or envelope.created_at <= query.until)
            ):
                all_envelopes.append(envelope)

        reverse = query.sort_order == SortOrder.DESC
        all_envelopes.sort(key=lambda e: e.created_at, reverse=reverse)

        envelopes = all_envelopes[: query.limit]
        next_cursor = None

        if "NextContinuationToken" in response:
            next_cursor = response["NextContinuationToken"]
        elif len(all_envelopes) > query.limit:
            next_cursor = "has_more"

        return envelopes, next_cursor

    async def _do_delete(self, run_id: RunId, checkpoint_id: CheckpointId) -> bool:
        """Delete a specific checkpoint from S3.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            True if checkpoint was deleted, False if it didn't exist.

        """
        client = await self._get_client()
        object_key = self._object_key(run_id, checkpoint_id)

        try:
            await client.head_object(Bucket=self.config.bucket, Key=object_key)
        except client.exceptions.NoSuchKey:
            return False

        await client.delete_object(Bucket=self.config.bucket, Key=object_key)
        return True

    async def _do_purge(self, run_id: RunId) -> int:
        """Delete all checkpoints for a run from S3.

        Args:
            run_id: The run identifier.

        Returns:
            Number of checkpoints deleted.

        """
        client = await self._get_client()
        prefix = f"{self.config.key_prefix}/runs/{run_id}/"

        response = await client.list_objects_v2(
            Bucket=self.config.bucket, Prefix=prefix
        )

        if "Contents" not in response:
            return 0

        deleted_count = 0
        for obj in response["Contents"]:
            await client.delete_object(Bucket=self.config.bucket, Key=obj["Key"])
            deleted_count += 1

        return deleted_count

    async def _do_healthcheck(self) -> bool:
        """Verify S3 connectivity and permissions.

        Returns:
            True if S3 is healthy.

        """
        client = await self._get_client()
        await client.head_bucket(Bucket=self.config.bucket)
        return True

    def __repr__(self) -> str:
        """Return a string representation of the store."""
        return (
            f"S3CheckpointStore(bucket={self.config.bucket}, "
            f"key_prefix={self.config.key_prefix})"
        )
