"""FlowDebugger for time-travel debugging and checkpoint inspection."""

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.table import Table

from pydantic_flow.checkpoints.interface import CheckpointStorageBackend
from pydantic_flow.checkpoints.reconstructor import StateReconstructor
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.streaming import ProgressItem


class FlowDebugger:
    """Time-travel debugger for checkpoint v2 flows.

    Provides methods to inspect, replay, and manipulate flow execution history.
    Enables time-travel debugging by rewinding to previous waves, forking execution
    paths, and comparing recorded vs re-executed behavior.

    Example:
        ```python
        from pydantic_flow.checkpoints import SQLiteBackend, FlowDebugger

        backend = SQLiteBackend(path="checkpoints.db")
        debugger = FlowDebugger(backend)

        # List all runs
        runs = await debugger.list_runs()
        for run in runs:
            print(f"{run.run_id}: {run.status} ({run.wave_count} waves)")

        # Replay specific run
        async for event in debugger.replay_from_checkpoint("run_123", wave=5):
            print(event)

        # Rewind and fork
        state = await debugger.rewind_to_wave("run_123", target_wave=3)
        # Continue execution from wave 3 with modified state
        ```

    """

    def __init__(self, backend: CheckpointStorageBackend) -> None:
        """Initialize debugger with checkpoint backend.

        Args:
            backend: Storage backend containing checkpoints and traces

        """
        self.backend = backend
        self.console = Console()

    async def list_runs(
        self,
        status: RunMetadata.Status | None = None,
        limit: int = 50,
    ) -> list[RunMetadata]:
        """List execution runs with optional filtering.

        Args:
            status: Filter by run status (RUNNING, COMPLETED, FAILED, INTERRUPTED)
            limit: Maximum number of runs to return

        Returns:
            List of run metadata sorted by start time (most recent first)

        """
        return await self.backend.list_runs(limit=limit)

    async def get_wave_timeline(self, run_id: str) -> list[StateSnapshot]:
        """Get chronological timeline of all waves in a run.

        Args:
            run_id: Run identifier

        Returns:
            List of state snapshots ordered by wave number

        Raises:
            ValueError: If run_id not found

        """
        snapshots = await self.backend.get_snapshots_range(
            run_id=run_id,
            start_wave=0,
            end_wave=999999,  # Get all waves
        )
        if not snapshots:
            msg = f"No snapshots found for run {run_id}"
            raise ValueError(msg)
        return snapshots

    async def replay_node_stream(
        self,
        run_id: str,
        wave: int,
        delay: float = 0.0,
    ) -> AsyncIterator[ProgressItem]:
        """Replay recorded event stream from a wave execution.

        Args:
            run_id: Run identifier
            wave: Wave number
            delay: Optional delay between events (seconds) for slow-motion replay

        Yields:
            ProgressItem events in chronological order

        Raises:
            ValueError: If trace not found

        """
        trace = await self.backend.get_trace(
            run_id=run_id,
            wave_number=wave,
        )
        if trace is None:
            msg = f"No trace found for {run_id}/wave={wave}"
            raise ValueError(msg)

        # Note: Individual events are stored separately in the backend
        # Would need to query event log via event_log_id from node_traces
        # For now, yield NodeExecutionTrace summaries
        for _node_trace in trace.node_traces:
            # Could fetch actual ProgressItem events here via backend
            # For MVP, just note the trace exists
            pass

        # TODO: Implement actual event replay by fetching from event log
        msg = "Event replay not yet implemented - requires event log fetch"
        raise NotImplementedError(msg)

    async def replay_from_checkpoint(
        self,
        run_id: str,
        wave: int,
    ) -> dict[str, Any]:
        """Replay execution state from a checkpoint.

        Reconstructs state by applying deltas forward from last full snapshot.

        Args:
            run_id: Run identifier
            wave: Target wave number

        Returns:
            Reconstructed state dictionary

        Raises:
            ValueError: If checkpoint not found

        """
        # Get snapshot for the wave
        snapshots = await self.backend.get_snapshots_range(
            run_id=run_id,
            start_wave=wave,
            end_wave=wave,
        )
        if not snapshots:
            msg = f"No checkpoint found for {run_id}/wave={wave}"
            raise ValueError(msg)

        snapshot = snapshots[0]

        # If it's a full snapshot, return directly
        if snapshot.full_state is not None:
            return snapshot.full_state

        # Otherwise, reconstruct from deltas
        from pydantic_flow.checkpoints.types import RunId

        reconstructor = StateReconstructor(self.backend)
        return await reconstructor.reconstruct_state_at(RunId(run_id), wave)

    async def compare_replay_vs_reexecution(
        self,
        run_id: str,
        wave: int,
    ) -> dict[str, Any]:
        """Compare recorded checkpoint state vs re-executing from previous wave.

        Useful for detecting non-deterministic behavior or state corruption.

        Args:
            run_id: Run identifier
            wave: Wave to compare

        Returns:
            Dictionary with keys:
                - recorded_state: State from checkpoint
                - differences: List of state differences found

        Raises:
            ValueError: If checkpoints not found
            NotImplementedError: Re-execution requires flow definition

        """
        # Get recorded state (verify checkpoint exists)
        _ = await self.replay_from_checkpoint(run_id, wave)

        # Re-execution requires flow definition which we don't have in debugger
        # This would need to be implemented at a higher level with access to Flow
        msg = "Re-execution requires flow definition - not available in debugger"
        raise NotImplementedError(msg)

    async def rewind_to_wave(self, run_id: str, target_wave: int) -> dict[str, Any]:
        """Rewind execution state to a previous wave (time-travel).

        Args:
            run_id: Run identifier
            target_wave: Wave number to rewind to

        Returns:
            State dictionary at target wave

        Raises:
            ValueError: If checkpoint not found

        """
        return await self.replay_from_checkpoint(run_id, target_wave)

    async def fork_from_wave(
        self,
        run_id: str,
        from_wave: int,
        new_run_id: str,
    ) -> str:
        """Fork execution from a checkpoint to create a new run branch.

        Creates a new run starting from an existing checkpoint, allowing
        exploration of alternative execution paths.

        Args:
            run_id: Source run identifier
            from_wave: Wave to fork from
            new_run_id: Identifier for forked run

        Returns:
            New run ID

        Raises:
            ValueError: If source checkpoint not found
            NotImplementedError: Forking requires flow definition

        """
        # Verify source checkpoint exists
        _ = await self.rewind_to_wave(run_id, from_wave)

        # Forking requires creating a new run and re-executing with modified state
        # This needs flow definition and execution engine
        msg = "Forking requires flow definition - not available in debugger"
        raise NotImplementedError(msg)

    async def load_from_archive(self, archive_path: str) -> list[str]:
        """Load checkpoints from exported archive.

        Args:
            archive_path: Path to .tar.gz archive

        Returns:
            List of imported run IDs

        Raises:
            NotImplementedError: Archive import not yet implemented

        """
        msg = "Archive import not yet implemented"
        raise NotImplementedError(msg)

    def render_wave_table(self, snapshots: list[StateSnapshot]) -> None:
        """Render wave timeline as Rich table.

        Args:
            snapshots: List of state snapshots to display

        """
        table = Table(title="Wave Timeline")
        table.add_column("Wave", style="cyan", no_wrap=True)
        table.add_column("Snapshot ID", style="magenta")
        table.add_column("Type", style="green")
        table.add_column("State Hash", style="yellow")
        table.add_column("Node Count", style="blue")

        for snapshot in snapshots:
            snapshot_type = "Full" if snapshot.full_state is not None else "Delta"
            node_count = len(snapshot.full_state or snapshot.forward_delta or {})

            table.add_row(
                str(snapshot.wave_number),
                snapshot.snapshot_id[:12],
                snapshot_type,
                snapshot.state_hash[:12],
                str(node_count),
            )

        self.console.print(table)

    async def render_run_list(self, runs: list[RunMetadata]) -> None:
        """Render list of runs as Rich table.

        Args:
            runs: List of run metadata to display

        """
        table = Table(title="Flow Runs")
        table.add_column("Run ID", style="cyan", no_wrap=True)
        table.add_column("Status", style="magenta")
        table.add_column("Waves", style="green")
        table.add_column("Started", style="yellow")
        table.add_column("Duration", style="blue")

        for run in runs:
            # Format duration
            duration = ""
            if run.completed_at and run.started_at:
                delta = run.completed_at - run.started_at
                duration = f"{delta.total_seconds():.2f}s"
            elif run.started_at:
                delta = datetime.now().astimezone() - run.started_at
                duration = f"{delta.total_seconds():.2f}s (running)"

            # Format start time
            start_str = (
                run.started_at.strftime("%Y-%m-%d %H:%M:%S")
                if run.started_at
                else "N/A"
            )

            table.add_row(
                run.run_id[:12],
                run.status.value,
                str(run.total_waves),
                start_str,
                duration,
            )

        self.console.print(table)
