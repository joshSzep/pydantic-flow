"""CLI commands for checkpoint v2 debugging."""

import asyncio
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Annotated
from typing import cast

from rich.console import Console
import typer

from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointBackend
from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.debugger import CheckpointDebugger
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.checkpoints.vacuum import VacuumManager
from pydantic_flow.checkpoints.vacuum import VacuumPolicy

app = typer.Typer(
    name="debug",
    help="Checkpoint v2 debugging and time-travel commands",
    no_args_is_help=True,
)
console = Console()


def get_debugger(db_path: Path) -> CheckpointDebugger:
    """Create a debugger instance from a database path."""
    config = SQLiteCheckpointConfig(db_path=db_path)
    backend = SQLiteCheckpointBackend(config=config)
    return CheckpointDebugger(backend=backend)


@app.command()
def list_runs(
    db_path: Annotated[
        Path,
        typer.Argument(
            help="Path to checkpoint database",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("checkpoints.db"),
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Maximum number of runs to show")
    ] = 20,
):
    """List all checkpoint runs in the database."""

    async def _list_runs():
        debugger = get_debugger(db_path)
        try:
            await debugger.backend.initialize()
            await debugger.show_runs(limit=limit)
        finally:
            await debugger.backend.close()

    asyncio.run(_list_runs())


@app.command()
def replay(
    run_id: Annotated[str, typer.Argument(help="Run ID to replay")],
    db_path: Annotated[
        Path,
        typer.Option(
            "--db",
            help="Path to checkpoint database",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("checkpoints.db"),
    wave: Annotated[
        int, typer.Option("--wave", "-w", help="Wave number to replay")
    ] = 1,
):
    """Replay execution from a specific checkpoint."""

    async def _replay():
        debugger = get_debugger(db_path)
        try:
            await debugger.backend.initialize()
            await debugger.replay_from_checkpoint(run_id=cast(RunId, run_id), wave=wave)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1) from None
        finally:
            await debugger.backend.close()

    asyncio.run(_replay())


@app.command()
def timeline(
    run_id: Annotated[str, typer.Argument(help="Run ID to show timeline for")],
    db_path: Annotated[
        Path,
        typer.Option(
            "--db",
            help="Path to checkpoint database",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("checkpoints.db"),
):
    """Show execution timeline for a run."""

    async def _timeline():
        debugger = get_debugger(db_path)
        try:
            await debugger.backend.initialize()
            await debugger.show_timeline(run_id=cast(RunId, run_id))
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1) from None
        finally:
            await debugger.backend.close()

    asyncio.run(_timeline())


@app.command()
def details(
    run_id: Annotated[str, typer.Argument(help="Run ID to show details for")],
    db_path: Annotated[
        Path,
        typer.Option(
            "--db",
            help="Path to checkpoint database",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("checkpoints.db"),
):
    """Show detailed information about a specific run."""

    async def _details():
        debugger = get_debugger(db_path)
        try:
            await debugger.backend.initialize()
            await debugger.show_run_details(run_id=cast(RunId, run_id))
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1) from None
        finally:
            await debugger.backend.close()

    asyncio.run(_details())


@app.command()
def rewind(
    run_id: Annotated[str, typer.Argument(help="Run ID to rewind")],
    to_wave: Annotated[int, typer.Option("--to-wave", "-w", help="Target wave")],
    db_path: Annotated[
        Path,
        typer.Option(
            "--db",
            help="Path to checkpoint database",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("checkpoints.db"),
):
    """Rewind execution to a previous wave (time-travel backward)."""

    async def _rewind():
        debugger = get_debugger(db_path)
        try:
            await debugger.backend.initialize()
            result = await debugger.rewind_to_wave(
                run_id=cast(RunId, run_id), target_wave=to_wave
            )
            console.print("\n[bold green]State reconstructed successfully[/bold green]")
            console.print(f"State keys: {list(result['state'].keys())}")
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1) from None
        finally:
            await debugger.backend.close()

    asyncio.run(_rewind())


@app.command()
def fork(
    run_id: Annotated[str, typer.Argument(help="Run ID to fork from")],
    from_wave: Annotated[
        int, typer.Option("--from-wave", "-w", help="Wave number to fork from")
    ],
    db_path: Annotated[
        Path,
        typer.Option(
            "--db",
            help="Path to checkpoint database",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("checkpoints.db"),
):
    """Fork execution from a checkpoint (create branching execution)."""

    async def _fork():
        debugger = get_debugger(db_path)
        try:
            await debugger.backend.initialize()
            # For CLI, fork without modifications
            # (users can modify state programmatically)
            result = await debugger.fork_from_wave(
                run_id=cast(RunId, run_id),
                source_wave=from_wave,
                state_modifications=None,
            )
            console.print("\n[bold green]Fork created successfully[/bold green]")
            console.print(f"Forked state keys: {list(result.keys())}")
            console.print(
                "\n[dim]Use this state to initialize a new flow execution[/dim]"
            )
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1) from None
        finally:
            await debugger.backend.close()

    asyncio.run(_fork())


@app.command()
def export(
    run_id: Annotated[str, typer.Argument(help="Run ID to export")],
    output: Annotated[
        Path,
        typer.Argument(help="Output path for archive (e.g., checkpoint.tar.gz)"),
    ],
    db_path: Annotated[
        Path,
        typer.Option(
            "--db",
            help="Path to checkpoint database",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("checkpoints.db"),
):
    """Export a run to a portable archive."""

    async def _export():
        debugger = get_debugger(db_path)
        try:
            await debugger.backend.initialize()
            result = await debugger.export_to_archive(
                run_id=cast(RunId, run_id), output_path=str(output)
            )
            console.print(
                f"\n[bold green]Exported run {result['run_id'][:12]}...[/bold green]"
            )
            console.print(f"Archive: {result['archive_path']}")
            console.print(f"Snapshots: {result['snapshot_count']}")
            console.print(f"Size: {result['size'] / 1024:.1f} KB")
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1) from None
        finally:
            await debugger.backend.close()

    asyncio.run(_export())


@app.command(name="import")
def import_archive(
    archive: Annotated[
        Path,
        typer.Argument(
            help="Archive path to import",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    db_path: Annotated[
        Path,
        typer.Option(
            "--db",
            help="Path to checkpoint database",
            dir_okay=False,
        ),
    ] = Path("checkpoints.db"),
):
    """Import a run from a portable archive."""

    async def _import():
        debugger = get_debugger(db_path)
        try:
            await debugger.backend.initialize()
            result = await debugger.load_from_archive(str(archive))
            console.print(
                f"\n[bold green]Imported run {result['run_id'][:12]}...[/bold green]"
            )
            console.print(f"Flow: {result['flow_name']}")
            console.print(f"Snapshots: {result['snapshot_count']}")
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1) from None
        finally:
            await debugger.backend.close()

    asyncio.run(_import())


@app.command()
def vacuum_traces(
    db_path: Annotated[
        Path,
        typer.Argument(
            help="Path to checkpoint database",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("checkpoints.db"),
    days: Annotated[
        int, typer.Option("--days", "-d", help="Delete traces older than N days")
    ] = 30,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be deleted without deleting"),
    ] = False,
):
    """Delete execution traces older than specified number of days."""

    async def _vacuum_traces():
        config = SQLiteCheckpointConfig(db_path=db_path)
        backend = SQLiteCheckpointBackend(config=config)
        vacuum_manager = VacuumManager(backend=backend)

        try:
            await backend.initialize()
            cutoff = datetime.now() - timedelta(days=days)

            if dry_run:
                console.print(
                    "\n[bold yellow]DRY RUN:[/bold yellow] "
                    "Showing what would be deleted"
                )

            console.print(f"Deleting traces older than {cutoff.date()}...")
            report = await vacuum_manager.vacuum_traces_before(cutoff, dry_run=dry_run)

            console.print(
                f"\n[bold green]Traces deleted:[/bold green] {report.traces_deleted}"
            )
            if report.dry_run:
                console.print(
                    "[yellow]Note: This was a dry run, "
                    "no data was actually deleted[/yellow]"
                )
        finally:
            await backend.close()

    asyncio.run(_vacuum_traces())


@app.command()
def vacuum_run(
    db_path: Annotated[
        Path,
        typer.Argument(
            help="Path to checkpoint database",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("checkpoints.db"),
    run_id: Annotated[str, typer.Argument(help="Run ID to delete")] = "",
    keep_checkpoints: Annotated[
        bool,
        typer.Option(
            "--keep-checkpoints",
            help="Keep state snapshots, delete only traces",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be deleted without deleting"),
    ] = False,
):
    """Delete all data for a specific run."""

    async def _vacuum_run():
        if not run_id:
            console.print("[red]Error:[/red] run_id is required")
            raise typer.Exit(code=1)

        config = SQLiteCheckpointConfig(db_path=db_path)
        backend = SQLiteCheckpointBackend(config=config)
        vacuum_manager = VacuumManager(backend=backend)

        try:
            await backend.initialize()

            if dry_run:
                console.print(
                    "\n[bold yellow]DRY RUN:[/bold yellow] "
                    "Showing what would be deleted"
                )

            console.print(f"Deleting run {run_id[:12]}...")
            report = await vacuum_manager.vacuum_run(
                RunId(run_id), keep_checkpoints=keep_checkpoints, dry_run=dry_run
            )

            console.print(
                f"\n[bold green]Traces deleted:[/bold green] {report.traces_deleted}"
            )
            console.print(
                f"[bold green]Runs deleted:[/bold green] {report.runs_deleted}"
            )
            if report.dry_run:
                console.print(
                    "[yellow]Note: This was a dry run, "
                    "no data was actually deleted[/yellow]"
                )
        finally:
            await backend.close()

    asyncio.run(_vacuum_run())


@app.command()
def vacuum_policy(  # noqa: PLR0913
    db_path: Annotated[
        Path,
        typer.Argument(
            help="Path to checkpoint database",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("checkpoints.db"),
    trace_retention_days: Annotated[
        int | None,
        typer.Option("--trace-days", help="Keep traces for N days"),
    ] = None,
    completed_retention_days: Annotated[
        int | None,
        typer.Option("--completed-days", help="Keep completed runs for N days"),
    ] = None,
    failed_retention_days: Annotated[
        int | None,
        typer.Option("--failed-days", help="Keep failed runs for N days"),
    ] = None,
    keep_checkpoints: Annotated[
        bool,
        typer.Option(
            "--keep-checkpoints",
            help="Keep state snapshots when deleting runs",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be deleted without deleting"),
    ] = False,
):
    """Apply retention policy to delete old checkpoint data."""

    async def _vacuum_policy():
        config = SQLiteCheckpointConfig(db_path=db_path)
        backend = SQLiteCheckpointBackend(config=config)
        vacuum_manager = VacuumManager(backend=backend)

        policy = VacuumPolicy(
            trace_retention_days=trace_retention_days,
            completed_run_retention_days=completed_retention_days,
            failed_run_retention_days=failed_retention_days,
            keep_checkpoints=keep_checkpoints,
            dry_run=dry_run,
        )

        try:
            await backend.initialize()

            if dry_run:
                console.print(
                    "\n[bold yellow]DRY RUN:[/bold yellow] "
                    "Showing what would be deleted"
                )

            console.print("Applying vacuum policy...")
            report = await vacuum_manager.vacuum_by_policy(policy)

            console.print(
                f"\n[bold green]Traces deleted:[/bold green] {report.traces_deleted}"
            )
            console.print(
                f"[bold green]Runs deleted:[/bold green] {report.runs_deleted}"
            )
            if report.dry_run:
                console.print(
                    "[yellow]Note: This was a dry run, "
                    "no data was actually deleted[/yellow]"
                )
        finally:
            await backend.close()

    asyncio.run(_vacuum_policy())


if __name__ == "__main__":
    app()
