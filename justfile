# Development automation for pydantic-flow
# https://github.com/casey/just

# Load environment variables from .env file if it exists
set dotenv-load := true

# Use bash as the shell
set shell := ["bash", "-c"]

# List available commands
default:
    @just --list

# Install the project and development dependencies
install:
    uv sync --all-extras --dev

# Install pre-commit hooks
install-hooks:
    uv run pre-commit install

# Run the full development setup
setup: install install-hooks
    @echo "✅ Development environment setup complete!"

# Run all tests
test:
    uv run pytest

# Run tests with coverage report
test-cov:
    uv run pytest --cov-report=html
    @echo "📊 Coverage report generated in htmlcov/"

# Run tests in watch mode
test-watch:
    uv run pytest --watch

# Lint the code with ruff
lint:
    uv run ruff check .

# Fix linting issues automatically
lint-fix:
    uv run ruff check --fix .

# Format the code with ruff
format:
    uv run ruff format .

# Check formatting without making changes
format-check:
    uv run ruff format --check .

# Run type checking with ty
typecheck:
    uv run ty check .

# Run all quality checks
check: format-check lint typecheck test

# Run pre-commit on all files
pre-commit:
    uv run pre-commit run --all-files

# Clean up temporary files and caches
clean:
    rm -rf .pytest_cache/
    rm -rf htmlcov/
    rm -rf .coverage
    rm -rf dist/
    rm -rf build/
    rm -rf *.egg-info/
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Build the package
build:
    uv build

# Update dependencies
update:
    uv lock --upgrade

# Show project information
info:
    @echo "📦 Project: pydantic-flow"
    @echo "🐍 Python: $(python --version)"
    @echo "📋 Dependencies:"
    @uv tree

# Run the full CI pipeline locally
ci: clean install check build
    @echo "✅ CI pipeline completed successfully!"