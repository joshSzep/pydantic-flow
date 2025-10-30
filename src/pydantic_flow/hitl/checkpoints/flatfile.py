"""Flat-file JSON checkpoint store implementation.

Stores checkpoints as individual JSON files with atomic writes
and maintains an index for efficient listing.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum
from pathlib import Path

import anyio
from pydantic import BaseModel

from pydantic_flow.hitl.checkpoints.base import BaseCheckpointStore
from pydantic_flow.hitl.checkpoints.interface import CheckpointConflict
from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
from pydantic_flow.hitl.checkpoints.interface import CheckpointId
from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
from pydantic_flow.hitl.checkpoints.interface import RunId
from pydantic_flow.hitl.checkpoints.interface import SortOrder
from pydantic_flow.hitl.checkpoints.serde import deserialize_checkpoint
from pydantic_flow.hitl.checkpoints.serde import serialize_checkpoint


class PartitioningStrategy(Enum):
    """Strategy for organizing checkpoint files."""

    NONE = "none"
    BY_RUN = "by_run"
    BY_DATE = "by_date"


class FlatFileCheckpointStoreConfig(BaseModel):
    """Configuration for flat-file checkpoint store.

    Attributes:
        base_path: Root directory for checkpoint storage.
        partitioning: How to organize checkpoint files.

    """

    base_path: Path
    partitioning: PartitioningStrategy = PartitioningStrategy.BY_RUN


class IndexEntry(BaseModel):
    """Single entry in the index file."""

    checkpoint_id: CheckpointId
    run_id: RunId
    node_id: str | None
    created_at: datetime
    file_path: str


class FlatFileCheckpointStore(BaseCheckpointStore):
    """Flat-file JSON checkpoint store.

    Stores each checkpoint as an individual JSON file with atomic writes.
    Maintains an append-only index per run for efficient listing.
    """

    def __init__(self, config: FlatFileCheckpointStoreConfig) -> None:
        """Initialize the flat-file store.

        Args:
            config: Store configuration.

        """
        self.config = config
        self._lock = asyncio.Lock()

    def _get_checkpoint_path(self, envelope: CheckpointEnvelope) -> Path:
        """Determine file path for a checkpoint.

        Args:
            envelope: The checkpoint envelope.

        Returns:
            Path where checkpoint should be stored.

        """
        base = self.config.base_path

        if self.config.partitioning == PartitioningStrategy.BY_RUN:
            return base / "runs" / envelope.run_id / f"{envelope.id}.json"
        elif self.config.partitioning == PartitioningStrategy.BY_DATE:
            date_str = envelope.created_at.strftime("%Y-%m-%d")
            return base / "dates" / date_str / envelope.run_id / f"{envelope.id}.json"
        else:
            return base / f"{envelope.run_id}_{envelope.id}.json"

    def _get_index_path(self, run_id: RunId) -> Path:
        """Get path to index file for a run.

        Args:
            run_id: The run identifier.

        Returns:
            Path to index file.

        """
        base = self.config.base_path
        if self.config.partitioning == PartitioningStrategy.BY_RUN:
            return base / "runs" / run_id / "index.jsonl"
        else:
            return base / f"{run_id}_index.jsonl"

    async def _append_to_index(self, envelope: CheckpointEnvelope) -> None:
        """Append checkpoint entry to index file.

        Args:
            envelope: The checkpoint envelope to index.

        """
        index_path = self._get_index_path(envelope.run_id)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        entry = IndexEntry(
            checkpoint_id=envelope.id,
            run_id=envelope.run_id,
            node_id=envelope.node_id,
            created_at=envelope.created_at,
            file_path=str(
                self._get_checkpoint_path(envelope).relative_to(self.config.base_path)
            ),
        )

        async with await anyio.open_file(index_path, "a") as f:
            await f.write(entry.model_dump_json() + "\n")

    async def _read_index(self, run_id: RunId) -> list[IndexEntry]:
        """Read all entries from an index file.

        Args:
            run_id: The run identifier.

        Returns:
            List of index entries.

        """
        index_path = self._get_index_path(run_id)
        if not index_path.exists():
            return []

        entries: list[IndexEntry] = []
        async with await anyio.open_file(index_path, "r") as f:
            async for raw_line in f:
                line = raw_line.strip()
                if line:
                    entries.append(IndexEntry.model_validate_json(line))
        return entries

    async def _do_save(
        self, envelope: CheckpointEnvelope, overwrite: bool
    ) -> CheckpointEnvelope:
        """Save checkpoint to flat file.

        Args:
            envelope: The prepared checkpoint envelope with computed hash.
            overwrite: If False, raise CheckpointConflict if file exists.

        Returns:
            The saved envelope.

        Raises:
            CheckpointConflict: If checkpoint exists and overwrite=False.

        """
        async with self._lock:
            checkpoint_path = self._get_checkpoint_path(envelope)

            if not overwrite and checkpoint_path.exists():
                msg = (
                    f"Checkpoint {envelope.id} already exists for run {envelope.run_id}"
                )
                raise CheckpointConflict(msg)

            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            temp_path = checkpoint_path.with_suffix(".tmp")
            json_str = serialize_checkpoint(envelope)

            async with await anyio.open_file(temp_path, "w") as f:
                await f.write(json_str)

            await anyio.Path(temp_path).rename(checkpoint_path)

            if not overwrite:
                await self._append_to_index(envelope)

            return envelope

    async def _do_latest(
        self, run_id: RunId, node_id: str | None = None
    ) -> CheckpointEnvelope | None:
        """Get the most recent checkpoint from flat file.

        Args:
            run_id: The run to query.
            node_id: Optional node filter.

        Returns:
            The latest checkpoint envelope, or None if not found.

        """
        try:
            entries = await self._read_index(run_id)
            if node_id is not None:
                entries = [e for e in entries if e.node_id == node_id]

            if not entries:
                return None

            latest_entry = max(entries, key=lambda e: e.created_at)
            file_path = self.config.base_path / latest_entry.file_path

            async with await anyio.open_file(file_path, "r") as f:
                content: str = await f.read()  # type: ignore[assignment]

            return deserialize_checkpoint(content)

        except FileNotFoundError:
            return None

    async def _do_get(
        self, run_id: RunId, checkpoint_id: CheckpointId
    ) -> CheckpointEnvelope | None:
        """Get a specific checkpoint by ID from flat file.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            The checkpoint envelope, or None if not found.

        """
        try:
            entries = await self._read_index(run_id)
            matching = [e for e in entries if e.checkpoint_id == checkpoint_id]

            if not matching:
                return None

            file_path = self.config.base_path / matching[0].file_path

            async with await anyio.open_file(file_path, "r") as f:
                content: str = await f.read()  # type: ignore[assignment]

            return deserialize_checkpoint(content)

        except FileNotFoundError:
            return None

    async def _do_list(
        self, query: CheckpointQuery
    ) -> tuple[list[CheckpointEnvelope], str | None]:
        """List checkpoints matching query criteria from flat file.

        Args:
            query: Query parameters for filtering and pagination.

        Returns:
            Tuple of (list of checkpoint envelopes, next cursor or None).

        """
        if query.run_id is None:
            return [], None

        entries = await self._read_index(query.run_id)

        if query.node_id is not None:
            entries = [e for e in entries if e.node_id == query.node_id]

        if query.since is not None:
            entries = [e for e in entries if e.created_at >= query.since]

        if query.until is not None:
            entries = [e for e in entries if e.created_at <= query.until]

        reverse = query.sort_order == SortOrder.DESC
        entries.sort(key=lambda e: e.created_at, reverse=reverse)

        cursor_idx = 0
        if query.cursor is not None:
            try:
                cursor_idx = int(query.cursor)
            except ValueError:
                cursor_idx = 0

        page_entries = entries[cursor_idx : cursor_idx + query.limit]

        envelopes: list[CheckpointEnvelope] = []
        for entry in page_entries:
            file_path = self.config.base_path / entry.file_path
            async with await anyio.open_file(file_path, "r") as f:
                content: str = await f.read()  # type: ignore[assignment]
            envelopes.append(deserialize_checkpoint(content))

        next_cursor = None
        if cursor_idx + query.limit < len(entries):
            next_cursor = str(cursor_idx + query.limit)

        return envelopes, next_cursor

    async def _do_delete(self, run_id: RunId, checkpoint_id: CheckpointId) -> bool:
        """Delete a specific checkpoint from flat file.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            True if checkpoint was deleted, False if it didn't exist.

        """
        entries = await self._read_index(run_id)
        matching = [e for e in entries if e.checkpoint_id == checkpoint_id]

        if not matching:
            return False

        file_path = self.config.base_path / matching[0].file_path
        if file_path.exists():
            await anyio.Path(file_path).unlink()
            return True
        return False

    async def _do_purge(self, run_id: RunId) -> int:
        """Delete all checkpoints for a run from flat file.

        Args:
            run_id: The run identifier.

        Returns:
            Number of checkpoints deleted.

        """
        entries = await self._read_index(run_id)
        deleted_count = 0

        for entry in entries:
            file_path = self.config.base_path / entry.file_path
            if file_path.exists():
                await anyio.Path(file_path).unlink()
                deleted_count += 1

        index_path = self._get_index_path(run_id)
        if index_path.exists():
            await anyio.Path(index_path).unlink()

        return deleted_count

    async def _do_healthcheck(self) -> bool:
        """Verify store access and permissions.

        Returns:
            True if store is accessible.

        """
        self.config.base_path.mkdir(parents=True, exist_ok=True)

        test_file = self.config.base_path / ".healthcheck"
        async with await anyio.open_file(test_file, "w") as f:
            await f.write("ok")
        await anyio.Path(test_file).unlink()
        return True

    def __repr__(self) -> str:
        """Return a string representation of the store."""
        return (
            f"FlatFileCheckpointStore(base_path={self.config.base_path}, "
            f"partitioning={self.config.partitioning.value})"
        )
