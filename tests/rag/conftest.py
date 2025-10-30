"""Test configuration for RAG tests."""


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--run-optional",
        action="store_true",
        default=False,
        help="Run tests requiring optional dependencies or services",
    )
