"""Filesystem storage backend for checkpoint v2.

This module provides a filesystem-based implementation of the checkpoint storage
backend, optimized for portability, sharing, and human inspection.

Directory structure:
    {root}/
        .index.json                 # log_id -> run_id mapping for O(1) lookups
        {run_id}/
            manifest.json           # Run metadata
            checkpoints/
                wave_0000.msgpack.gz
                wave_0010.msgpack.gz
            traces/
                wave_0000.msgpack.gz
            events/
                {log_id}/
                    events_0000-0099.msgpack.gz
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
from pathlib import Path
import tarfile
from typing import Any
from typing import Literal

import anyio
from pydantic import BaseModel
from pydantic import Field

from pydantic_flow.checkpoints.serialization import TypedSerializer
from pydantic_flow.checkpoints.serialization import compress
from pydantic_flow.checkpoints.serialization import decompress
from pydantic_flow.checkpoints.types import ExecutionTrace
from pydantic_flow.checkpoints.types import NodeExecutionTrace
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateSnapshot


class FilesystemCheckpointConfig(BaseModel):
    """Configuration for filesystem checkpoint backend.

    Attributes:
        root_dir: Root directory for checkpoint storage.
        create_dirs: Whether to create directories on initialization.
        compress_level: Compression level for state data (1-9).
        pretty_json: Whether to pretty-print JSON files.

    """

    root_dir: Path
    create_dirs: bool = True
    compress_level: int = Field(default=6, ge=1, le=9)
    pretty_json: bool = True


class FilesystemCheckpointBackend:
    """Filesystem backend for checkpoint storage.

    Optimized for portability, sharing, and human inspection. Uses standard
    directory structure with msgpack-compressed binary files for snapshots/traces
    and JSON for metadata.

    Benefits:
        - Portable: Share as .tar.gz archives
        - Human-readable: Standard directory structure
        - Version control: Can commit checkpoint directories
        - Zero infrastructure: Works everywhere
        - Inspectable: Use tar, zcat, jq to inspect
        - O(1) lookups: Index file for fast queries

    Args:
        config: Filesystem backend configuration.

    """

    def __init__(self, config: FilesystemCheckpointConfig):
        """Initialize filesystem backend.

        Args:
            config: Backend configuration.

        """
        self.config = config
        self.root_dir = config.root_dir
        self.index_path = self.root_dir / ".index.json"
        self._log_to_run_cache: dict[str, str] = {}
        self._index_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize backend by creating directories and loading index."""
        if self._initialized:
            return

        if self.config.create_dirs:
            self.root_dir.mkdir(parents=True, exist_ok=True)

        # Load index from disk if it exists
        if self.index_path.exists():
            async with await anyio.open_file(self.index_path, "r") as f:
                content = await f.read()
                self._log_to_run_cache = json.loads(content) if content else {}

        self._initialized = True

    async def close(self) -> None:
        """Close backend resources (no-op for filesystem)."""
        pass

    async def healthcheck(self) -> bool:
        """Check backend health.

        Returns:
            True if directory is accessible and writable.

        """
        try:
            if not self.root_dir.exists():
                return False

            # Try to write a test file
            test_file = self.root_dir / ".healthcheck"
            test_file.write_text("ok")
            test_file.unlink()
            return True
        except Exception:
            return False

    def _run_dir(self, run_id: RunId) -> Path:
        """Get directory for a specific run."""
        return self.root_dir / run_id

    def _manifest_path(self, run_id: RunId) -> Path:
        """Get path to run manifest file."""
        return self._run_dir(run_id) / "manifest.json"

    def _checkpoint_path(self, run_id: RunId, wave_number: int) -> Path:
        """Get path to checkpoint file."""
        filename = f"wave_{wave_number:04d}.msgpack.gz"
        return self._run_dir(run_id) / "checkpoints" / filename

    def _trace_path(self, run_id: RunId, wave_number: int) -> Path:
        """Get path to trace file."""
        return self._run_dir(run_id) / "traces" / f"wave_{wave_number:04d}.msgpack.gz"

    def _node_trace_path(self, run_id: RunId, log_id: str) -> Path:
        """Get path to node trace file."""
        return self._run_dir(run_id) / "traces" / "nodes" / f"{log_id}.msgpack.gz"

    def _events_dir(self, run_id: RunId, log_id: str) -> Path:
        """Get directory for event log."""
        return self._run_dir(run_id) / "events" / log_id

    async def _update_index(self, log_id: str, run_id: RunId) -> None:
        """Update index with log_id -> run_id mapping."""
        async with self._index_lock:
            self._log_to_run_cache[log_id] = run_id

            # Write index to disk atomically
            temp_path = self.index_path.with_suffix(".tmp")
            indent = 2 if self.config.pretty_json else None
            async with await anyio.open_file(temp_path, "w") as f:
                await f.write(json.dumps(self._log_to_run_cache, indent=indent))

            await anyio.Path(temp_path).rename(self.index_path)

    async def _get_run_id_for_log(self, log_id: str) -> RunId | None:
        """Get run_id for a log_id from index."""
        result = self._log_to_run_cache.get(log_id)
        return RunId(result) if result else None

    # Metadata operations

    async def save_run_metadata(self, metadata: RunMetadata) -> None:
        """Save run metadata to manifest.json."""
        manifest_path = self._manifest_path(metadata.run_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        indent = 2 if self.config.pretty_json else None
        temp_path = manifest_path.with_suffix(".tmp")

        async with await anyio.open_file(temp_path, "w") as f:
            await f.write(metadata.model_dump_json(indent=indent))

        await anyio.Path(temp_path).rename(manifest_path)

    async def get_run_metadata(self, run_id: RunId) -> RunMetadata | None:
        """Retrieve run metadata from manifest.json."""
        manifest_path = self._manifest_path(run_id)
        if not manifest_path.exists():
            return None

        async with await anyio.open_file(manifest_path, "r") as f:
            content = await f.read()
            return RunMetadata.model_validate_json(content)

    # State snapshot operations

    async def save_state_snapshot(self, snapshot: StateSnapshot) -> None:
        """Save state snapshot as msgpack-compressed binary file."""
        checkpoint_path = self._checkpoint_path(snapshot.run_id, snapshot.wave_number)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize and compress
        data = TypedSerializer.serialize(snapshot)
        compressed = compress(data, level=self.config.compress_level)

        # Atomic write
        temp_path = checkpoint_path.with_suffix(".tmp")
        async with await anyio.open_file(temp_path, "wb") as f:
            await f.write(compressed)

        await anyio.Path(temp_path).rename(checkpoint_path)

    async def get_state_snapshot(
        self, run_id: RunId, wave_number: int
    ) -> StateSnapshot | None:
        """Retrieve state snapshot from binary file."""
        checkpoint_path = self._checkpoint_path(run_id, wave_number)
        if not checkpoint_path.exists():
            return None

        async with await anyio.open_file(checkpoint_path, "rb") as f:
            compressed = await f.read()

        decompressed = decompress(compressed)
        return TypedSerializer.deserialize(decompressed)

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
        snapshots: list[StateSnapshot] = []

        for wave in range(start_wave, end_wave + 1):
            snapshot = await self.get_state_snapshot(run_id, wave)
            if snapshot:
                snapshots.append(snapshot)

        if order == "DESC":
            snapshots.reverse()

        return snapshots

    # Trace operations

    async def save_trace(self, trace: ExecutionTrace) -> None:
        """Save execution trace as msgpack-compressed binary file."""
        trace_path = self._trace_path(trace.run_id, trace.wave_number)
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize and compress
        data = TypedSerializer.serialize(trace)
        compressed = compress(data, level=self.config.compress_level)

        # Atomic write
        temp_path = trace_path.with_suffix(".tmp")
        async with await anyio.open_file(temp_path, "wb") as f:
            await f.write(compressed)

        await anyio.Path(temp_path).rename(trace_path)

        # Update index for node traces
        for node_trace in trace.node_traces:
            await self._update_index(node_trace.log_id, trace.run_id)

    async def get_trace(self, run_id: RunId, wave_number: int) -> ExecutionTrace | None:
        """Retrieve execution trace from binary file."""
        trace_path = self._trace_path(run_id, wave_number)
        if not trace_path.exists():
            return None

        async with await anyio.open_file(trace_path, "rb") as f:
            compressed = await f.read()

        decompressed = decompress(compressed)
        return TypedSerializer.deserialize(decompressed)

    async def delete_trace(self, run_id: RunId, wave_number: int) -> bool:
        """Delete execution trace."""
        trace_path = self._trace_path(run_id, wave_number)
        if not trace_path.exists():
            return False

        trace_path.unlink()
        return True

    async def save_node_trace(self, node_trace: NodeExecutionTrace) -> None:
        """Save node execution trace as msgpack-compressed binary file."""
        # Need to find run_id for this log_id
        run_id = await self._get_run_id_for_log(node_trace.log_id)
        if not run_id:
            msg = (
                f"Cannot find run_id for log_id {node_trace.log_id}. "
                "Trace must be saved first."
            )
            raise ValueError(msg)

        node_trace_path = self._node_trace_path(run_id, node_trace.log_id)
        node_trace_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize and compress
        data = TypedSerializer.serialize(node_trace)
        compressed = compress(data, level=self.config.compress_level)

        # Atomic write
        temp_path = node_trace_path.with_suffix(".tmp")
        async with await anyio.open_file(temp_path, "wb") as f:
            await f.write(compressed)

        await anyio.Path(temp_path).rename(node_trace_path)

    async def get_node_trace(self, log_id: str) -> NodeExecutionTrace | None:
        """Retrieve node execution trace."""
        run_id = await self._get_run_id_for_log(log_id)
        if not run_id:
            return None

        node_trace_path = self._node_trace_path(run_id, log_id)
        if not node_trace_path.exists():
            return None

        async with await anyio.open_file(node_trace_path, "rb") as f:
            compressed = await f.read()

        decompressed = decompress(compressed)
        return TypedSerializer.deserialize(decompressed)

    # Event stream operations (placeholder)

    async def append_events_batch(
        self,
        log_id: str,
        events: list[Any],
        start_sequence: int,
    ) -> None:
        """Append batch of events to event log.

        Note: Placeholder implementation. Full event storage will be
        implemented when event streaming is needed.

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

        Note: Placeholder implementation. Full event retrieval will be
        implemented when event streaming is needed.

        Args:
            log_id: Event log identifier.
            start_offset: Starting offset for event stream.
            end_offset: Optional ending offset.

        Returns:
            Empty list (placeholder).

        """
        return []

    # Vacuum operations

    async def list_runs(
        self,
        *,
        before: datetime | None = None,
        after: datetime | None = None,
        limit: int | None = None,
    ) -> list[RunMetadata]:
        """List runs with optional filtering."""
        runs: list[RunMetadata] = []

        # Iterate over all run directories
        if not self.root_dir.exists():
            return runs

        for run_dir in self.root_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue

            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            async with await anyio.open_file(manifest_path, "r") as f:
                content = await f.read()
                metadata = RunMetadata.model_validate_json(content)

            # Apply filters
            if before and metadata.started_at >= before:
                continue
            if after and metadata.started_at <= after:
                continue

            runs.append(metadata)

        # Sort by started_at descending (newest first)
        runs.sort(key=lambda r: r.started_at, reverse=True)

        # Apply limit
        if limit:
            runs = runs[:limit]

        return runs

    async def delete_run(
        self, run_id: RunId, *, keep_checkpoints: bool = False
    ) -> None:
        """Delete all data for a run."""
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            return

        if keep_checkpoints:
            # Delete traces and events but keep checkpoints
            traces_dir = run_dir / "traces"
            if traces_dir.exists():
                import shutil

                shutil.rmtree(traces_dir)

            events_dir = run_dir / "events"
            if events_dir.exists():
                import shutil

                shutil.rmtree(events_dir)

            manifest_path = self._manifest_path(run_id)
            if manifest_path.exists():
                manifest_path.unlink()
        else:
            # Delete entire run directory
            import shutil

            shutil.rmtree(run_dir)

        # Clean up index entries
        async with self._index_lock:
            # Remove all log_ids for this run
            self._log_to_run_cache = {
                log_id: rid
                for log_id, rid in self._log_to_run_cache.items()
                if rid != run_id
            }

            # Write updated index
            indent = 2 if self.config.pretty_json else None
            temp_path = self.index_path.with_suffix(".tmp")
            async with await anyio.open_file(temp_path, "w") as f:
                await f.write(json.dumps(self._log_to_run_cache, indent=indent))

            await anyio.Path(temp_path).rename(self.index_path)

    # Export for sharing

    async def export_for_sharing(
        self,
        run_id: RunId,
        output_path: Path,
        *,
        include_traces: bool = True,
        include_events: bool = False,
    ) -> None:
        """Export run as shareable .tar.gz archive.

        Args:
            run_id: Run identifier to export.
            output_path: Path for output .tar.gz file.
            include_traces: Whether to include execution traces.
            include_events: Whether to include event logs.

        Raises:
            ValueError: If run does not exist.

        """
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            msg = f"Run {run_id} does not exist"
            raise ValueError(msg)

        # Create tar.gz archive
        with tarfile.open(output_path, "w:gz") as tar:
            # Always include manifest
            manifest_path = self._manifest_path(run_id)
            if manifest_path.exists():
                tar.add(manifest_path, arcname="manifest.json")

            # Always include checkpoints
            checkpoints_dir = run_dir / "checkpoints"
            if checkpoints_dir.exists():
                tar.add(checkpoints_dir, arcname="checkpoints")

            # Optionally include traces
            if include_traces:
                traces_dir = run_dir / "traces"
                if traces_dir.exists():
                    tar.add(traces_dir, arcname="traces")

            # Optionally include events
            if include_events:
                events_dir = run_dir / "events"
                if events_dir.exists():
                    tar.add(events_dir, arcname="events")

        # Generate README
        readme_content = await self._generate_readme(
            run_id, output_path, include_traces, include_events
        )
        readme_path = output_path.parent / f"{output_path.stem}_README.txt"
        async with await anyio.open_file(readme_path, "w") as f:
            await f.write(readme_content)

    async def _generate_readme(
        self,
        run_id: RunId,
        archive_path: Path,
        include_traces: bool,
        include_events: bool,
    ) -> str:
        """Generate README for exported archive."""
        metadata = await self.get_run_metadata(run_id)

        # Count checkpoints
        checkpoints_dir = self._run_dir(run_id) / "checkpoints"
        if checkpoints_dir.exists():
            checkpoint_count = len(list(checkpoints_dir.glob("*.msgpack.gz")))
        else:
            checkpoint_count = 0

        readme = f"""# Checkpoint Archive: {run_id}

## Archive Information
- **Archive File**: {archive_path.name}
- **Created**: {datetime.now().isoformat()}
- **Run ID**: {run_id}

## Run Information
"""

        if metadata:
            readme += f"""- **Flow ID**: {metadata.flow_id}
- **Started**: {metadata.started_at.isoformat()}
- **Completed**: {metadata.completed_at.isoformat() if metadata.completed_at else "N/A"}
- **Status**: {metadata.status.value}
"""

        readme += f"""
## Contents
- **Checkpoints**: {checkpoint_count} wave snapshots
- **Traces**: {"Included" if include_traces else "Not included"}
- **Events**: {"Included" if include_events else "Not included"}

## How to Use

### Extract Archive
```bash
tar -xzf {archive_path.name}
```

### Replay Execution
```bash
pydantic-flow debug replay {run_id} --checkpoint-dir ./
```

### List Checkpoints
```bash
ls -lh checkpoints/
```

### Inspect Checkpoint
```bash
# Decompress and view with msgpack tools
zcat checkpoints/wave_0000.msgpack.gz | msgpack2json | jq
```

## Archive Structure
```
manifest.json           # Run metadata
checkpoints/            # State snapshots (msgpack.gz)
  wave_0000.msgpack.gz
  wave_0001.msgpack.gz
  ...
"""

        if include_traces:
            readme += """traces/                 # Execution traces (msgpack.gz)
  wave_0000.msgpack.gz
  ...
"""

        if include_events:
            readme += """events/                 # Event logs (msgpack.gz)
  <log_id>/
    events_0000-0099.msgpack.gz
    ...
"""

        readme += """```

## Notes
- Checkpoint files are msgpack-compressed for efficient storage
- Use `zcat` and `msgpack2json` to inspect binary files
- Compatible with pydantic-flow checkpoint v2 system
"""

        return readme
