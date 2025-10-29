"""Core interface and types for checkpoint storage.

This module defines the Protocol and types for checkpoint persistence,
enabling flows to interrupt and resume across processes or machines.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from enum import Enum
import secrets
import time
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Protocol

from pydantic import BaseModel
from pydantic import Field
from pydantic_core import core_schema

from pydantic_flow.core.errors import FlowCheckpoint


class CheckpointId(str):
    """Unique identifier for a checkpoint."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Tell Pydantic to treat CheckpointId as a string."""
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema()
        )


class RunId(str):
    """Unique identifier for a flow execution run."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Tell Pydantic to treat RunId as a string."""
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema()
        )


class SortOrder(Enum):
    """Sort order for checkpoint queries."""

    ASC = "asc"
    DESC = "desc"


class CheckpointEnvelope(BaseModel):
    """Wrapper for persisted checkpoint with metadata.

    Attributes:
        id: Unique identifier for this checkpoint.
        run_id: Identifier for the flow execution run.
        node_id: Optional node where interruption occurred.
        created_at: Timestamp when checkpoint was created.
        schema_version: Version for backward compatibility.
        checkpoint: The actual checkpoint data from the flow.
        metadata: Additional context about the checkpoint.
        content_hash: SHA-256 hash of checkpoint and metadata for verification.

    """

    id: CheckpointId
    run_id: RunId
    node_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: Literal[1] = 1
    checkpoint: FlowCheckpoint
    metadata: dict[str, Any] | None = None
    content_hash: str | None = None


class CheckpointQuery(BaseModel):
    """Query parameters for listing checkpoints.

    Attributes:
        run_id: Filter by run identifier.
        node_id: Filter by node identifier.
        limit: Maximum number of results to return.
        cursor: Pagination cursor from previous query.
        since: Return only checkpoints created after this time.
        until: Return only checkpoints created before this time.
        sort_order: Sort by created_at in ascending or descending order.

    """

    run_id: RunId | None = None
    node_id: str | None = None
    limit: Annotated[int, Field(ge=1, le=1000)] = 100
    cursor: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    sort_order: SortOrder = SortOrder.DESC


class CheckpointStore(Protocol):
    """Protocol for checkpoint storage backends.

    All methods are async to support various storage backends.
    Implementations must be safe for concurrent access.
    """

    async def save(
        self, envelope: CheckpointEnvelope, *, overwrite: bool = False
    ) -> CheckpointEnvelope:
        """Save a checkpoint to storage.

        Args:
            envelope: The checkpoint envelope to save.
            overwrite: If False, raise CheckpointConflict if ID exists.

        Returns:
            The saved envelope, potentially with updated timestamps or hashes.

        Raises:
            CheckpointConflict: If checkpoint ID exists and overwrite=False.
            CheckpointBackendError: If storage operation fails.

        """
        ...

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
            CheckpointBackendError: If storage operation fails.

        """
        ...

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
            CheckpointBackendError: If storage operation fails.

        """
        ...

    async def list(
        self, query: CheckpointQuery
    ) -> tuple[list[CheckpointEnvelope], str | None]:
        """List checkpoints matching query criteria.

        Args:
            query: Query parameters for filtering and pagination.

        Returns:
            Tuple of (list of checkpoint envelopes, next cursor or None).

        Raises:
            CheckpointBackendError: If storage operation fails.

        """
        ...

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
        ...

    async def purge(self, run_id: RunId) -> int:
        """Delete all checkpoints for a run.

        Args:
            run_id: The run identifier.

        Returns:
            Number of checkpoints deleted.

        Raises:
            CheckpointBackendError: If storage operation fails.

        """
        ...

    async def healthcheck(self) -> bool:
        """Verify store connectivity and permissions.

        Returns:
            True if store is healthy and operational.

        Raises:
            CheckpointBackendError: If store is unhealthy.

        """
        ...


class CheckpointStoreError(Exception):
    """Base exception for checkpoint store errors."""


class CheckpointConflict(CheckpointStoreError):
    """Raised when attempting to save a checkpoint that already exists."""


class CheckpointNotFound(CheckpointStoreError):
    """Raised when a requested checkpoint does not exist."""


class CheckpointBackendError(CheckpointStoreError):
    """Raised when a storage backend operation fails."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize backend error.

        Args:
            message: Error description.
            cause: The underlying exception that caused this error.

        """
        super().__init__(message)
        self.cause = cause


def generate_checkpoint_id() -> CheckpointId:
    """Generate a unique checkpoint ID using timestamp-based UUID.

    IDs are roughly sortable by creation time and globally unique.
    Format: timestamp_microseconds + random component for collision resistance.

    Returns:
        A new checkpoint identifier.

    """
    timestamp_us = int(time.time() * 1_000_000)
    random_suffix = secrets.token_hex(8)
    return CheckpointId(f"{timestamp_us:016x}{random_suffix}")
