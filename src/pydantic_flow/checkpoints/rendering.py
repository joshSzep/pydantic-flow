"""Checkpoint rendering - terminal UI and visualization.

This module provides the presentation layer for checkpoint debugging.
Renders checkpoint data using Rich for beautiful terminal output.
"""

from datetime import datetime

from rich.console import Console
from rich.table import Table

from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateSnapshot


class CheckpointRenderer:
    """Terminal UI renderer for checkpoint data.

    Uses Rich library to create beautiful, informative terminal output
    for checkpoint inspection and debugging workflows.
    """

    def __init__(self, console: Console | None = None) -> None:
        """Initialize renderer.

        Args:
            console: Optional Rich console instance (creates new if not provided)

        """
        self.console = console or Console()

    def render_runs_table(
        self, runs: list[RunMetadata], title: str = "Flow Runs"
    ) -> None:
        """Render list of runs as a formatted table.

        Args:
            runs: List of run metadata to display
            title: Table title

        """
        table = Table(title=title)
        table.add_column("Run ID", style="cyan", no_wrap=True)
        table.add_column("Status", style="magenta")
        table.add_column("Waves", style="green", justify="right")
        table.add_column("Started", style="yellow")
        table.add_column("Duration", style="blue", justify="right")

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
                run.run_id[:12] + "...",
                run.status.value,
                str(run.total_waves),
                start_str,
                duration,
            )

        self.console.print(table)

    def render_wave_timeline(
        self, snapshots: list[StateSnapshot], title: str = "Wave Timeline"
    ) -> None:
        """Render wave timeline as a formatted table.

        Args:
            snapshots: List of state snapshots to display
            title: Table title

        """
        table = Table(title=title)
        table.add_column("Wave", style="cyan", no_wrap=True, justify="right")
        table.add_column("Snapshot ID", style="magenta")
        table.add_column("Type", style="green")
        table.add_column("State Hash", style="yellow")
        table.add_column("Nodes", style="blue", justify="right")

        for snapshot in snapshots:
            snapshot_type = "Full" if snapshot.full_state is not None else "Delta"
            node_count = len(snapshot.full_state or snapshot.forward_delta or {})

            table.add_row(
                str(snapshot.wave_number),
                snapshot.snapshot_id[:12] + "...",
                snapshot_type,
                snapshot.state_hash[:12] + "...",
                str(node_count),
            )

        self.console.print(table)

    def render_run_summary(self, metadata: RunMetadata) -> None:
        """Render detailed summary of a single run.

        Args:
            metadata: Run metadata to display

        """
        self.console.print(f"\n[bold cyan]Run: {metadata.run_id}[/bold cyan]")
        self.console.print(f"Flow ID: {metadata.flow_id}")
        self.console.print(
            f"Status: [{self._status_color(metadata.status)}]{metadata.status.value}[/]"
        )
        self.console.print(f"Total Waves: {metadata.total_waves}")

        if metadata.started_at:
            self.console.print(
                f"Started: {metadata.started_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        if metadata.completed_at:
            self.console.print(
                f"Completed: {metadata.completed_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            delta = metadata.completed_at - metadata.started_at
            self.console.print(f"Duration: {delta.total_seconds():.2f}s")

        if metadata.error:
            self.console.print("\n[bold red]Error:[/bold red]")
            self.console.print(f"  Type: {metadata.error.error_type}")
            self.console.print(f"  Message: {metadata.error.error_message}")

    def _status_color(self, status: RunMetadata.Status) -> str:
        """Get Rich color for status.

        Args:
            status: Run status

        Returns:
            Rich color name

        """
        if status == RunMetadata.Status.COMPLETED:
            return "green"
        if status == RunMetadata.Status.FAILED:
            return "red"
        if status == RunMetadata.Status.RUNNING:
            return "yellow"
        return "white"
