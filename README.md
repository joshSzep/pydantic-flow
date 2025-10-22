# pflow

> A type-safe, batteries-included AI agent framework built on pydantic-ai

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**pflow** is designed to rival LangChain while maintaining the developer experience and type safety that the Pydantic ecosystem is known for. Built on [pydantic-ai](https://ai.pydantic.dev/), it provides a comprehensive toolkit for building complex AI agents from 0 to 100.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Plugin System](#plugin-system)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Features

### 🚀 Batteries Included
- **Tools**: HTTP, filesystem, vector operations out of the box
- **Memory**: Persistent conversation history and context management
- **Observability**: Built-in tracing with OpenTelemetry
- **Durability**: State persistence and recovery mechanisms
- **CLI**: Complete command-line interface for agent management

### 🔒 Type Safety First
- Full type annotations with comprehensive IDE support
- Modern Python 3.13+ syntax (`A | B`, `dict`/`list`/`tuple`)
- Pydantic models for all public interfaces
- Runtime validation with detailed error messages

### ⚡ Developer Experience
- Async-first design with sync wrappers where appropriate
- Auto-discovery of plugins and tools
- Fast-running comprehensive test suite
- Rich terminal output and debugging

### 🔌 Extensible
- Plugin architecture inspired by pytest ecosystem
- External libraries can create `pflow-*` extensions
- Type-safe protocol-based interfaces
- Dependency injection with automatic wiring

## Installation

```bash
pip install pflow
```

### Development Installation

```bash
git clone https://github.com/joshSzep/pflow.git
cd pflow
uv sync
```

## Quick Start

```python
# Coming soon!
```

## Architecture

pflow follows core principles that ensure scalability and maintainability:

- **Long-Lived Agents**: Persistent objects with integrated memory and tools
- **Immutable State**: Idempotent APIs over stateful operations
- **Fail Fast**: Rich exception context with custom error types
- **Dependency Injection**: Auto-discovery with explicit configuration

For detailed architectural decisions and patterns, see [AGENTS.md](./AGENTS.md).

## Development

### Requirements
- Python 3.13+
- uv for dependency management

### Setup
```bash
# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Run linting
uv run ruff check
uv run ruff format
```

### Project Structure
```
pflow/
├── src/pflow/           # Core framework
├── tests/               # Test suite
├── docs/                # Documentation
├── AGENTS.md           # Architecture guide
└── pyproject.toml      # Project configuration
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`uv run pytest && uv run ruff check`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by [LangChain](https://langchain.com/) - the AI agent framework to rival
- Built on [pydantic-ai](https://ai.pydantic.dev/) - the foundation for type-safe AI agents
- Inspired by the [Pydantic](https://pydantic.dev/) ecosystem's commitment to developer experience
- Plugin architecture inspired by [pytest](https://pytest.org/)'s extensible design

---

**Status**: Early development (v0.1.0) - APIs may change
