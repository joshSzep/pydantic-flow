"""S3-compatible checkpoint store implementation.

Stores checkpoints as JSON objects in S3-compatible object storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from pydantic_flow.hitl.checkpoints.interface import CheckpointBackendError
from pydantic_flow.hitl.checkpoints.interface import CheckpointConflict
from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
from pydantic_flow.hitl.checkpoints.interface import CheckpointId
from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
from pydantic_flow.hitl.checkpoints.interface import RunId
from pydantic_flow.hitl.checkpoints.interface import SortOrder
from pydantic_flow.hitl.checkpoints.serde import compute_content_hash
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


class S3CheckpointStore:
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

    async def save(
        self, envelope: CheckpointEnvelope, *, overwrite: bool = False
    ) -> CheckpointEnvelope:
        """Save a checkpoint to S3.

        Args:
            envelope: The checkpoint envelope to save.
            overwrite: If False, raise CheckpointConflict if key exists.

        Returns:
            The saved envelope with computed content hash.

        Raises:
            CheckpointConflict: If checkpoint exists and overwrite=False.
            CheckpointBackendError: If S3 operation fails.

        """
        try:
            client = await self._get_client()

            envelope_copy = envelope.model_copy(deep=True)
            if envelope_copy.content_hash is None:
                envelope_copy.content_hash = compute_content_hash(envelope_copy)

            json_str = serialize_checkpoint(envelope_copy)
            object_key = self._object_key(envelope_copy.run_id, envelope_copy.id)

            if not overwrite:
                try:
                    await client.head_object(Bucket=self.config.bucket, Key=object_key)
                    msg = (
                        f"Checkpoint {envelope_copy.id} already exists "
                        f"for run {envelope_copy.run_id}"
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

            return envelope_copy

        except CheckpointConflict:
            raise
        except Exception as e:
            msg = f"Failed to save checkpoint: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def latest(
        self, run_id: RunId, node_id: str | None = None
    ) -> CheckpointEnvelope | None:
        """Get the most recent checkpoint for a run.

        Args:
            run_id: The run to query.
            node_id: Optional node filter.

        Returns:
            The latest checkpoint envelope, or None if not found.

        Raises:
            CheckpointBackendError: If S3 operation fails.

        """
        try:
            client = await self._get_client()
            prefix = f"{self.config.key_prefix}/runs/{run_id}/"

            response = await client.list_objects_v2(
                Bucket=self.config.bucket, Prefix=prefix
            )

            if "Contents" not in response or not response["Contents"]:
                return None

            objects = response["Contents"]

            # If node_id filter is specified, fetch and filter all checkpoints
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

                # Find the one with latest LastModified time
                _, latest_envelope = max(
                    checkpoints, key=lambda pair: pair[0]["LastModified"]
                )
                return latest_envelope

            # No filter - just get the most recently modified object
            latest_key = max(objects, key=lambda obj: obj["LastModified"])
            obj_response = await client.get_object(
                Bucket=self.config.bucket, Key=latest_key["Key"]
            )
            body = await obj_response["Body"].read()
            return deserialize_checkpoint(body.decode("utf-8"))

        except Exception as e:
            msg = f"Failed to get latest checkpoint: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def get(
        self, run_id: RunId, checkpoint_id: CheckpointId
    ) -> CheckpointEnvelope | None:
        """Get a specific checkpoint by ID.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            The checkpoint envelope, or None if not found.

        Raises:
            CheckpointBackendError: If S3 operation fails.

        """
        try:
            client = await self._get_client()
            object_key = self._object_key(run_id, checkpoint_id)

            response = await client.get_object(
                Bucket=self.config.bucket, Key=object_key
            )

            body = await response["Body"].read()
            return deserialize_checkpoint(body.decode("utf-8"))

        except Exception as e:
            if "NoSuchKey" in str(e):
                return None
            msg = f"Failed to get checkpoint: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def list(
        self, query: CheckpointQuery
    ) -> tuple[list[CheckpointEnvelope], str | None]:
        """List checkpoints matching query criteria.

        Args:
            query: Query parameters for filtering and pagination.

        Returns:
            Tuple of (list of checkpoint envelopes, next cursor or None).

        Raises:
            CheckpointBackendError: If S3 operation fails.

        """
        try:
            if query.run_id is None:
                return [], None

            client = await self._get_client()
            prefix = f"{self.config.key_prefix}/runs/{query.run_id}/"

            # Build list_objects_v2 parameters
            list_params: dict[str, object] = {
                "Bucket": self.config.bucket,
                "Prefix": prefix,
                "MaxKeys": query.limit + 1,  # Request one extra to check for more
            }

            # Use cursor for pagination
            if query.cursor:
                list_params["ContinuationToken"] = query.cursor

            response = await client.list_objects_v2(**list_params)

            if "Contents" not in response:
                return [], None

            # Fetch all objects and deserialize
            all_envelopes: list[CheckpointEnvelope] = []
            for obj in response["Contents"]:
                obj_response = await client.get_object(
                    Bucket=self.config.bucket, Key=obj["Key"]
                )
                body = await obj_response["Body"].read()
                envelope = deserialize_checkpoint(body.decode("utf-8"))

                # Apply filters
                if (
                    (query.node_id is None or envelope.node_id == query.node_id)
                    and (query.since is None or envelope.created_at >= query.since)
                    and (query.until is None or envelope.created_at <= query.until)
                ):
                    all_envelopes.append(envelope)

            # Sort by creation time
            reverse = query.sort_order == SortOrder.DESC
            all_envelopes.sort(key=lambda e: e.created_at, reverse=reverse)

            # Apply limit and determine next cursor
            envelopes = all_envelopes[: query.limit]
            next_cursor = None

            # Use S3's NextContinuationToken if available
            if "NextContinuationToken" in response:
                next_cursor = response["NextContinuationToken"]
            elif len(all_envelopes) > query.limit:
                # We have more filtered results than the limit
                next_cursor = "has_more"

            return envelopes, next_cursor

        except Exception as e:
            msg = f"Failed to list checkpoints: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def delete(self, run_id: RunId, checkpoint_id: CheckpointId) -> bool:
        """Delete a specific checkpoint.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            True if checkpoint was deleted, False if it didn't exist.

        Raises:
            CheckpointBackendError: If S3 operation fails.

        """
        try:
            client = await self._get_client()
            object_key = self._object_key(run_id, checkpoint_id)

            try:
                await client.head_object(Bucket=self.config.bucket, Key=object_key)
            except client.exceptions.NoSuchKey:
                return False

            await client.delete_object(Bucket=self.config.bucket, Key=object_key)
            return True

        except Exception as e:
            msg = f"Failed to delete checkpoint: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def purge(self, run_id: RunId) -> int:
        """Delete all checkpoints for a run.

        Args:
            run_id: The run identifier.

        Returns:
            Number of checkpoints deleted.

        Raises:
            CheckpointBackendError: If S3 operation fails.

        """
        try:
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

        except Exception as e:
            msg = f"Failed to purge checkpoints: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def healthcheck(self) -> bool:
        """Verify S3 connectivity and permissions.

        Raises:
            CheckpointBackendError: If S3 is unhealthy.

        """
        try:
            client = await self._get_client()
            await client.head_bucket(Bucket=self.config.bucket)
            return True

        except Exception as e:
            msg = f"Healthcheck failed: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    def __repr__(self) -> str:
        """Return a string representation of the store."""
        return (
            f"S3CheckpointStore(bucket={self.config.bucket}, "
            f"key_prefix={self.config.key_prefix})"
        )
