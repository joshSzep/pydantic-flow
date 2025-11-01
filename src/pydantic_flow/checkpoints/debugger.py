"""Checkpoint debugging coordinator - high-level workflows.

This module provides convenient workflows that compose the inspection
and rendering layers for common debugging tasks.
"""

from pydantic import BaseModel
from rich.console import Console

from pydantic_flow.checkpoints.inspection import CheckpointInspector
from pydantic_flow.checkpoints.interface import CheckpointStorageBackend
from pydantic_flow.checkpoints.rendering import CheckpointRenderer
from pydantic_flow.checkpoints.types import RunId


class CheckpointDebugger:
    """High-level checkpoint debugging workflows.

    Composes CheckpointInspector (data layer) and CheckpointRenderer
    (presentation layer) to provide convenient debugging operations.

    Example:
        ```python
        from pydantic_flow.checkpoints import SQLiteCheckpointBackend
        from pydantic_flow.checkpoints.debugger import CheckpointDebugger

        backend = SQLiteCheckpointBackend(path="checkpoints.db")
        debugger = CheckpointDebugger(backend)

        # Show all runs
        await debugger.show_runs()

        # Show timeline for specific run
        await debugger.show_timeline("run_123")

        # Get state at specific wave
        state = await debugger.get_state("run_123", wave=5)
        ```

    """

    def __init__(
        self,
        backend: CheckpointStorageBackend,
        console: Console | None = None,
    ) -> None:
        """Initialize debugger.

        Args:
            backend: Storage backend for checkpoint data
            console: Optional Rich console (creates new if not provided)

        """
        self.inspector = CheckpointInspector(backend)
        self.renderer = CheckpointRenderer(console)
        self.backend = backend

    async def show_runs(self, limit: int = 50) -> None:
        """Show list of all runs in a formatted table.

        Args:
            limit: Maximum number of runs to display

        """
        runs = await self.inspector.list_runs(limit=limit)
        self.renderer.render_runs_table(runs)

    async def show_timeline(
        self, run_id: RunId, start_wave: int = 0, end_wave: int | None = None
    ) -> None:
        """Show wave timeline for a run in a formatted table.

        Args:
            run_id: Run identifier
            start_wave: Starting wave number (default: 0)
            end_wave: Ending wave number (default: all waves)

        """
        snapshots = await self.inspector.get_wave_timeline(run_id, start_wave, end_wave)
        if not snapshots:
            self.renderer.console.print(
                f"[yellow]No waves found for run {run_id}[/yellow]"
            )
            return

        self.renderer.render_wave_timeline(
            snapshots, title=f"Timeline for {run_id[:12]}..."
        )

    async def show_run_details(self, run_id: RunId) -> None:
        """Show detailed information about a specific run.

        Args:
            run_id: Run identifier

        """
        metadata = await self.inspector.get_run(run_id)
        if not metadata:
            self.renderer.console.print(f"[red]Run {run_id} not found[/red]")
            return

        self.renderer.render_run_summary(metadata)

    async def get_state(self, run_id: RunId, wave: int) -> dict:
        """Get reconstructed state at a specific wave.

        This is a data-only operation (no rendering).

        Args:
            run_id: Run identifier
            wave: Wave number

        Returns:
            Complete state dictionary at the specified wave

        Raises:
            ValueError: If wave not found

        """
        return await self.inspector.reconstruct_state(run_id, wave)

    async def get_latest_state(self, run_id: RunId) -> dict | None:
        """Get reconstructed state at the latest wave.

        Args:
            run_id: Run identifier

        Returns:
            Complete state dictionary at latest wave, or None if no waves

        Raises:
            ValueError: If state reconstruction fails

        """
        latest_wave = await self.inspector.get_latest_wave(run_id)
        if latest_wave is None:
            return None

        return await self.inspector.reconstruct_state(run_id, latest_wave)

    async def replay_from_checkpoint(
        self, run_id: RunId, wave: int, show_events: bool = True
    ) -> dict:
        """Replay execution from a specific checkpoint by showing recorded events.

        This retrieves the execution trace and displays the events that occurred
        during that wave's execution, allowing you to see exactly what happened
        without re-running the flow.

        Args:
            run_id: Run identifier
            wave: Wave number to replay
            show_events: Whether to print events to console (default: True)

        Returns:
            Dictionary containing replay information:
            - snapshot: State snapshot at this wave
            - trace: Execution trace with node execution details
            - node_count: Number of nodes executed in this wave

        Raises:
            ValueError: If wave or trace not found

        """
        # Get the snapshot for this wave
        snapshot = await self.inspector.get_wave_snapshot(run_id, wave)
        if not snapshot:
            msg = f"No snapshot found for run {run_id} at wave {wave}"
            raise ValueError(msg)

        # Get the execution trace
        trace = await self.inspector.get_wave_trace(run_id, wave)
        if not trace:
            msg = f"No trace found for run {run_id} at wave {wave}"
            raise ValueError(msg)

        # Display replay information
        if show_events:
            self.renderer.console.print(
                f"\n[bold]Replaying Run {run_id[:12]}... Wave {wave}[/bold]"
            )
            duration = (trace.completed_at - trace.started_at).total_seconds()
            self.renderer.console.print(f"Duration: {duration:.2f}s")
            self.renderer.console.print(f"Nodes executed: {len(trace.node_traces)}")

            # Show details for each node in this wave
            for i, node_trace in enumerate(trace.node_traces, 1):
                node_info = f"\n  [bold]{i}. Node: {node_trace.node_id}[/bold]"
                self.renderer.console.print(node_info)
                node_duration = (
                    node_trace.completed_at - node_trace.started_at
                ).total_seconds()
                self.renderer.console.print(f"     Duration: {node_duration:.2f}s")
                self.renderer.console.print(
                    f"     Events: {node_trace.total_events} "
                    f"(tokens: {node_trace.event_summary.token_count}, "
                    f"tools: {node_trace.event_summary.tool_call_count})"
                )
                if node_trace.cache_hit:
                    self.renderer.console.print("     [green]Cache hit![/green]")
                if node_trace.error:
                    self.renderer.console.print(
                        f"     [red]Error: {node_trace.error.error_message}[/red]"
                    )

        return {
            "snapshot": snapshot,
            "trace": trace,
            "node_count": len(trace.node_traces),
        }

    async def rewind_to_wave(self, run_id: RunId, target_wave: int) -> dict:
        """Rewind execution to a previous wave for time-travel debugging.

        This reconstructs the state at the target wave, allowing you to inspect
        what the flow state looked like at that earlier point in time.

        Args:
            run_id: Run identifier
            target_wave: Wave number to rewind to

        Returns:
            Dictionary containing:
            - wave: Target wave number
            - state: Reconstructed state dictionary
            - snapshot: State snapshot metadata
            - next_waves: List of waves that followed (for context)

        Raises:
            ValueError: If target wave not found

        """
        # Reconstruct state at target wave
        state = await self.inspector.reconstruct_state(run_id, target_wave)

        # Get the snapshot metadata
        snapshot = await self.inspector.get_wave_snapshot(run_id, target_wave)
        if not snapshot:
            msg = f"No snapshot found for run {run_id} at wave {target_wave}"
            raise ValueError(msg)

        # Get subsequent waves for context
        all_snapshots = await self.inspector.get_wave_timeline(
            run_id, start_wave=target_wave + 1
        )
        next_waves = [s.wave_number for s in all_snapshots]

        self.renderer.console.print(f"\n[bold]Rewound to Wave {target_wave}[/bold]")
        self.renderer.console.print(f"Run: {run_id[:12]}...")
        self.renderer.console.print(f"State keys: {list(state.keys())}")
        if next_waves:
            MAX_WAVES_PREVIEW = 5
            self.renderer.console.print(
                f"Subsequent waves available: {next_waves[:MAX_WAVES_PREVIEW]}"
                + (" ..." if len(next_waves) > MAX_WAVES_PREVIEW else "")
            )

        return {
            "wave": target_wave,
            "state": state,
            "snapshot": snapshot,
            "next_waves": next_waves,
        }

    async def fork_from_wave(
        self,
        run_id: RunId,
        source_wave: int,
        state_modifications: dict[str, BaseModel] | None = None,
    ) -> dict:
        """Fork execution from a checkpoint wave with optional state modifications.

        This creates a new execution branch from a previous checkpoint, allowing
        you to modify the state and explore alternative execution paths.

        Args:
            run_id: Source run identifier
            source_wave: Wave number to fork from
            state_modifications: Optional dict of node_id -> new state to modify

        Returns:
            Dictionary containing:
            - source_run_id: Original run identifier
            - source_wave: Wave forked from
            - forked_state: Modified state ready for re-execution
            - modifications: Dict of which nodes were modified

        Raises:
            ValueError: If source wave not found or modifications invalid

        """
        # Reconstruct state at source wave
        original_state = await self.inspector.reconstruct_state(run_id, source_wave)

        # Get snapshot metadata for validation
        snapshot = await self.inspector.get_wave_snapshot(run_id, source_wave)
        if not snapshot:
            msg = f"No snapshot found for run {run_id} at wave {source_wave}"
            raise ValueError(msg)

        # Apply modifications if provided
        forked_state = original_state.copy()
        modifications = {}

        if state_modifications:
            for node_id, new_state in state_modifications.items():
                if node_id not in forked_state:
                    msg = f"Node '{node_id}' not found in state at wave {source_wave}"
                    raise ValueError(msg)

                old_state = forked_state[node_id]
                forked_state[node_id] = new_state
                modifications[node_id] = {
                    "old": old_state,
                    "new": new_state,
                }

        # Display fork information
        self.renderer.console.print(
            f"\n[bold green]Forked from Wave {source_wave}[/bold green]"
        )
        self.renderer.console.print(f"Source Run: {run_id[:12]}...")
        self.renderer.console.print(f"State nodes: {list(forked_state.keys())}")

        if modifications:
            self.renderer.console.print(
                f"\n[yellow]Modified {len(modifications)} node(s):[/yellow]"
            )
            for node_id in modifications:
                self.renderer.console.print(f"  • {node_id}")
        else:
            self.renderer.console.print("\n[dim]No modifications applied[/dim]")

        return {
            "source_run_id": run_id,
            "source_wave": source_wave,
            "forked_state": forked_state,
            "modifications": modifications,
        }

    async def export_to_archive(self, run_id: RunId, output_path: str) -> dict:
        """Export a run's checkpoints to a portable tar.gz archive.

        Creates a compressed archive containing all snapshots, traces, and
        metadata for a run. This archive can be shared and imported elsewhere.

        Args:
            run_id: Run identifier to export
            output_path: Path to output tar.gz file

        Returns:
            Dictionary containing:
            - run_id: Exported run identifier
            - archive_path: Path to created archive
            - snapshot_count: Number of snapshots exported
            - total_size_bytes: Archive size in bytes

        Raises:
            ValueError: If run not found
            OSError: If archive creation fails

        """
        from pathlib import Path
        import tarfile
        import tempfile

        # Get run metadata
        metadata = await self.inspector.get_run(run_id)
        if not metadata:
            msg = f"Run {run_id} not found"
            raise ValueError(msg)

        # Get all snapshots and traces
        snapshots = await self.inspector.get_wave_timeline(run_id)

        # Create temporary directory for JSON files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Write metadata
            metadata_file = tmp_path / "metadata.json"
            metadata_file.write_text(metadata.model_dump_json(indent=2))

            # Write snapshots (binary serialized for proper type preservation)
            snapshots_dir = tmp_path / "snapshots"
            snapshots_dir.mkdir()
            for snapshot in snapshots:
                snapshot_file = snapshots_dir / f"wave_{snapshot.wave_number}.msgpack"
                snapshot_file.write_bytes(snapshot.serialize())

            # Write traces
            traces_dir = tmp_path / "traces"
            traces_dir.mkdir()
            for snapshot in snapshots:
                trace = await self.inspector.get_wave_trace(
                    run_id, snapshot.wave_number
                )
                if trace:
                    trace_file = traces_dir / f"wave_{snapshot.wave_number}.json"
                    trace_file.write_text(trace.model_dump_json(indent=2))

            # Create tar.gz archive
            output = Path(output_path)
            with tarfile.open(output, "w:gz") as tar:
                tar.add(tmp_path, arcname=run_id[:12])

        # Get archive size
        archive_size = output.stat().st_size

        self.renderer.console.print(
            "\n[bold green]Exported run to archive[/bold green]"
        )
        self.renderer.console.print(f"Run: {run_id[:12]}...")
        self.renderer.console.print(f"Archive: {output_path}")
        self.renderer.console.print(f"Snapshots: {len(snapshots)}")
        self.renderer.console.print(f"Size: {archive_size / 1024:.1f} KB")

        return {
            "run_id": run_id,
            "archive_path": str(output),
            "snapshot_count": len(snapshots),
            "total_size_bytes": archive_size,
        }

    async def load_from_archive(self, archive_path: str) -> dict:
        """Load checkpoints from a portable tar.gz archive.

        Imports a run's checkpoints from an archive created by export_to_archive.
        This allows sharing and analyzing runs from other environments.

        Args:
            archive_path: Path to tar.gz archive file

        Returns:
            Dictionary containing:
            - run_id: Imported run identifier
            - snapshot_count: Number of snapshots imported
            - metadata: Run metadata

        Raises:
            ValueError: If archive is invalid
            OSError: If file operations fail

        """
        from pathlib import Path
        import tarfile
        import tempfile

        archive = Path(archive_path)
        if not archive.exists():
            msg = f"Archive not found: {archive_path}"
            raise ValueError(msg)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Extract archive
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(tmp_path)

            # Find the run directory (should be only one)
            run_dirs = list(tmp_path.iterdir())
            if len(run_dirs) != 1:
                msg = (
                    f"Invalid archive: expected 1 run directory, found {len(run_dirs)}"
                )
                raise ValueError(msg)

            run_dir = run_dirs[0]

            # Load metadata
            metadata_file = run_dir / "metadata.json"
            if not metadata_file.exists():
                msg = "Invalid archive: metadata.json not found"
                raise ValueError(msg)

            from pydantic_flow.checkpoints.types import RunMetadata

            metadata = RunMetadata.model_validate_json(metadata_file.read_text())

            # Save to backend
            await self.backend.save_run_metadata(metadata)

            # Load snapshots (binary deserialized for proper type preservation)
            snapshots_dir = run_dir / "snapshots"
            snapshot_count = 0
            if snapshots_dir.exists():
                from pydantic_flow.checkpoints.types import StateSnapshot

                for snapshot_file in sorted(snapshots_dir.glob("wave_*.msgpack")):
                    snapshot = StateSnapshot.deserialize(snapshot_file.read_bytes())
                    await self.backend.save_state_snapshot(snapshot)
                    snapshot_count += 1

            # Load traces
            traces_dir = run_dir / "traces"
            if traces_dir.exists():
                from pydantic_flow.checkpoints.types import ExecutionTrace

                for trace_file in sorted(traces_dir.glob("wave_*.json")):
                    trace = ExecutionTrace.model_validate_json(trace_file.read_text())
                    await self.backend.save_trace(trace)

        self.renderer.console.print(
            "\n[bold green]Imported run from archive[/bold green]"
        )
        self.renderer.console.print(f"Run: {metadata.run_id[:12]}...")
        self.renderer.console.print(f"Flow: {metadata.flow_id}")
        self.renderer.console.print(f"Snapshots: {snapshot_count}")

        return {
            "run_id": metadata.run_id,
            "snapshot_count": snapshot_count,
            "metadata": metadata,
        }
