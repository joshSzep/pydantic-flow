"""Main entry point for pydantic-flow when run as a module."""

from pydantic_flow.project_info import get_project_info


def main():
    """Print project description and version."""
    info = get_project_info()
    print(f"pydantic-flow v{info.version}: {info.description}")


if __name__ == "__main__":
    main()
