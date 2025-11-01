"""Main entry point for pydantic-flow when run as a module."""

import typer

from pydantic_flow.checkpoints.cli import app as debug_app
from pydantic_flow.project_info import get_project_info

app = typer.Typer(
    name="pydantic-flow",
    help="Type-safe AI agent framework built on pydantic-ai",
    no_args_is_help=True,
)

# Add checkpoint debugging commands
app.add_typer(debug_app, name="debug")


@app.command()
def version():
    """Show version information."""
    info = get_project_info()
    typer.echo(f"pydantic-flow v{info.version}")


@app.command()
def info():
    """Show project information."""
    project_info = get_project_info()
    typer.echo(f"pydantic-flow v{project_info.version}")
    typer.echo(project_info.description)


def main():
    """Provide entry point for tests and backwards compatibility."""
    info_data = get_project_info()
    print(f"pydantic-flow v{info_data.version}: {info_data.description}")


if __name__ == "__main__":
    import sys

    # If no arguments provided, show version info instead of help
    if len(sys.argv) == 1:
        main()
    else:
        app()
