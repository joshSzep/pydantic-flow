"""Abstract base class for checkpoint stores with common functionality.

Provides shared implementation for validation, hashing, serialization,
and error handling. Backend-specific implementations extend this base.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from pydantic_flow.hitl.checkpoints.interface import CheckpointBackendError
from pydantic_flow.hitl.checkpoints.interface import CheckpointConflict
from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
from pydantic_flow.hitl.checkpoints.interface import CheckpointId
from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
from pydantic_flow.hitl.checkpoints.interface import RunId
from pydantic_flow.hitl.checkpoints.serde import compute_content_hash
from pydantic_flow.hitl.checkpoints.serde import verify_content_hash


class BaseCheckpointStore(ABC):
    """Abstract base class for checkpoint stores.

    Provides common functionality for all checkpoint store implementations:
    - Envelope validation and deep copying
    - Content hash computation and verification
    - Consistent error handling patterns
    - Template methods for backend-specific operations

    Subclasses implement backend-specific storage operations via abstract methods.
    """

    async def save(
        self, envelope: CheckpointEnvelope, *, overwrite: bool = False
    ) -> CheckpointEnvelope:
        """Save a checkpoint with automatic validation and hashing.

        Handles common save logic:
        1. Validates and copies the envelope
        2. Computes content hash if missing
        3. Delegates to backend-specific save
        4. Verifies saved data integrity

        Args:
            envelope: The checkpoint envelope to save.
            overwrite: If False, raise CheckpointConflict if ID exists.

        Returns:
            The saved envelope with computed content hash.

        Raises:
            CheckpointConflict: If checkpoint ID exists and overwrite=False.
            CheckpointBackendError: If storage operation fails.

        """
        validated_envelope = self._validate_and_prepare_envelope(envelope)

        try:
            saved_envelope = await self._do_save(
                validated_envelope, overwrite=overwrite
            )
            self._verify_saved_envelope(saved_envelope)
            return saved_envelope
        except CheckpointConflict:
            raise
        except CheckpointBackendError:
            raise
        except Exception as e:
            msg = (
                f"Failed to save checkpoint {envelope.id} "
                f"for run {envelope.run_id}: {e}"
            )
            raise CheckpointBackendError(msg, cause=e) from e

    async def latest(
        self, run_id: RunId, node_id: str | None = None
    ) -> CheckpointEnvelope | None:
        """Get the most recent checkpoint for a run with hash verification.

        Args:
            run_id: The run to query.
            node_id: Optional node filter.

        Returns:
            The latest checkpoint envelope, or None if not found.

        Raises:
            CheckpointBackendError: If storage operation fails or hash
                verification fails.

        """
        try:
            envelope = await self._do_latest(run_id, node_id)
            if envelope is not None:
                self._verify_saved_envelope(envelope)
            return envelope
        except CheckpointBackendError:
            raise
        except Exception as e:
            msg = f"Failed to get latest checkpoint for run {run_id}: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def get(
        self, run_id: RunId, checkpoint_id: CheckpointId
    ) -> CheckpointEnvelope | None:
        """Get a specific checkpoint by ID with hash verification.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            The checkpoint envelope, or None if not found.

        Raises:
            CheckpointBackendError: If storage operation fails or hash
                verification fails.

        """
        try:
            envelope = await self._do_get(run_id, checkpoint_id)
            if envelope is not None:
                self._verify_saved_envelope(envelope)
            return envelope
        except CheckpointBackendError:
            raise
        except Exception as e:
            msg = f"Failed to get checkpoint {checkpoint_id} for run {run_id}: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def list(
        self, query: CheckpointQuery
    ) -> tuple[list[CheckpointEnvelope], str | None]:
        """List checkpoints with hash verification for all results.

        Args:
            query: Query parameters for filtering and pagination.

        Returns:
            Tuple of (list of checkpoint envelopes, next cursor or None).

        Raises:
            CheckpointBackendError: If storage operation fails.

        """
        try:
            envelopes, cursor = await self._do_list(query)
            for envelope in envelopes:
                self._verify_saved_envelope(envelope)
            return envelopes, cursor
        except CheckpointBackendError:
            raise
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
            CheckpointBackendError: If storage operation fails.

        """
        try:
            return await self._do_delete(run_id, checkpoint_id)
        except CheckpointBackendError:
            raise
        except Exception as e:
            msg = f"Failed to delete checkpoint {checkpoint_id} for run {run_id}: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def purge(self, run_id: RunId) -> int:
        """Delete all checkpoints for a run.

        Args:
            run_id: The run identifier.

        Returns:
            Number of checkpoints deleted.

        Raises:
            CheckpointBackendError: If storage operation fails.

        """
        try:
            return await self._do_purge(run_id)
        except CheckpointBackendError:
            raise
        except Exception as e:
            msg = f"Failed to purge checkpoints for run {run_id}: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def healthcheck(self) -> bool:
        """Verify store connectivity and permissions.

        Returns:
            True if store is healthy and operational.

        Raises:
            CheckpointBackendError: If store is unhealthy.

        """
        try:
            return await self._do_healthcheck()
        except CheckpointBackendError:
            raise
        except Exception as e:
            msg = f"Healthcheck failed: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def count_checkpoints(self, run_id: RunId) -> int:
        """Count checkpoints for a specific run.

        Args:
            run_id: The run identifier.

        Returns:
            Number of checkpoints for the run.

        Raises:
            CheckpointBackendError: If storage operation fails.

        """
        try:
            return await self._do_count_checkpoints(run_id)
        except CheckpointBackendError:
            raise
        except Exception as e:
            msg = f"Failed to count checkpoints for run {run_id}: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def get_checkpoint_history(
        self, run_id: RunId, limit: int = 10
    ) -> list[CheckpointEnvelope]:
        """Get checkpoint history for a specific run, newest first.

        Args:
            run_id: The run identifier.
            limit: Maximum number of checkpoints to return.

        Returns:
            List of checkpoint envelopes, sorted by creation time (newest first).

        Raises:
            CheckpointBackendError: If storage operation fails.

        """
        try:
            envelopes = await self._do_get_checkpoint_history(run_id, limit)
            for envelope in envelopes:
                self._verify_saved_envelope(envelope)
            return envelopes
        except CheckpointBackendError:
            raise
        except Exception as e:
            msg = f"Failed to get checkpoint history for run {run_id}: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    def _validate_and_prepare_envelope(
        self, envelope: CheckpointEnvelope
    ) -> CheckpointEnvelope:
        """Validate and prepare envelope for storage.

        Creates a deep copy and computes content hash if needed.

        Args:
            envelope: The checkpoint envelope to prepare.

        Returns:
            Validated and prepared envelope copy.

        """
        envelope_copy = envelope.model_copy(deep=True)

        if envelope_copy.content_hash is None:
            envelope_copy.content_hash = compute_content_hash(envelope_copy)

        return envelope_copy

    def _verify_saved_envelope(self, envelope: CheckpointEnvelope) -> None:
        """Verify envelope integrity via content hash.

        Args:
            envelope: The envelope to verify.

        Raises:
            CheckpointBackendError: If hash verification fails.

        """
        if not verify_content_hash(envelope):
            msg = (
                f"Content hash mismatch for checkpoint {envelope.id} "
                f"in run {envelope.run_id}"
            )
            raise CheckpointBackendError(msg)

    @abstractmethod
    async def _do_save(
        self, envelope: CheckpointEnvelope, overwrite: bool
    ) -> CheckpointEnvelope:
        """Backend-specific checkpoint save implementation.

        Args:
            envelope: The prepared envelope with computed hash.
            overwrite: If False, raise CheckpointConflict if ID exists.

        Returns:
            The saved envelope.

        Raises:
            CheckpointConflict: If checkpoint exists and overwrite=False.
            Exception: Any backend-specific errors (will be wrapped).

        """
        ...

    @abstractmethod
    async def _do_latest(
        self, run_id: RunId, node_id: str | None = None
    ) -> CheckpointEnvelope | None:
        """Backend-specific implementation for getting latest checkpoint.

        Args:
            run_id: The run to query.
            node_id: Optional node filter.

        Returns:
            The latest checkpoint envelope, or None if not found.

        Raises:
            Exception: Any backend-specific errors (will be wrapped).

        """
        ...

    @abstractmethod
    async def _do_get(
        self, run_id: RunId, checkpoint_id: CheckpointId
    ) -> CheckpointEnvelope | None:
        """Backend-specific implementation for getting specific checkpoint.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            The checkpoint envelope, or None if not found.

        Raises:
            Exception: Any backend-specific errors (will be wrapped).

        """
        ...

    @abstractmethod
    async def _do_list(
        self, query: CheckpointQuery
    ) -> tuple[list[CheckpointEnvelope], str | None]:
        """Backend-specific implementation for listing checkpoints.

        Args:
            query: Query parameters for filtering and pagination.

        Returns:
            Tuple of (list of checkpoint envelopes, next cursor or None).

        Raises:
            Exception: Any backend-specific errors (will be wrapped).

        """
        ...

    @abstractmethod
    async def _do_delete(self, run_id: RunId, checkpoint_id: CheckpointId) -> bool:
        """Backend-specific implementation for deleting checkpoint.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            True if deleted, False if didn't exist.

        Raises:
            Exception: Any backend-specific errors (will be wrapped).

        """
        ...

    @abstractmethod
    async def _do_purge(self, run_id: RunId) -> int:
        """Backend-specific implementation for purging all checkpoints.

        Args:
            run_id: The run identifier.

        Returns:
            Number of checkpoints deleted.

        Raises:
            Exception: Any backend-specific errors (will be wrapped).

        """
        ...

    @abstractmethod
    async def _do_healthcheck(self) -> bool:
        """Backend-specific healthcheck implementation.

        Returns:
            True if healthy.

        Raises:
            Exception: Any backend-specific errors (will be wrapped).

        """
        ...

    @abstractmethod
    async def _do_count_checkpoints(self, run_id: RunId) -> int:
        """Backend-specific implementation for counting checkpoints.

        Args:
            run_id: The run identifier.

        Returns:
            Number of checkpoints for the run.

        Raises:
            Exception: Any backend-specific errors (will be wrapped).

        """
        ...

    @abstractmethod
    async def _do_get_checkpoint_history(
        self, run_id: RunId, limit: int
    ) -> list[CheckpointEnvelope]:
        """Backend-specific implementation for getting checkpoint history.

        Args:
            run_id: The run identifier.
            limit: Maximum number of checkpoints to return.

        Returns:
            List of checkpoint envelopes, sorted by creation time (newest first).

        Raises:
            Exception: Any backend-specific errors (will be wrapped).

        """
        ...
