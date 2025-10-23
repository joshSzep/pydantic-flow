# Contributing to pydantic-flow

Thank you for your interest in contributing to pydantic-flow! We welcome contributions from the community and are grateful for any help you can provide.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Plugin Development](#plugin-development)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/yourusername/pydantic-flow.git
   cd pydantic-flow
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Install pre-commit hooks:**
   ```bash
   uv run pre-commit install
   ```

4. **Verify installation:**
   ```bash
   uv run pytest
   uv run ruff check
   ```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-anthropic-support`
- `fix/memory-leak-in-vector-tools`
- `docs/update-plugin-guide`
- `refactor/simplify-agent-lifecycle`

### Commit Messages

Follow conventional commit format:
```
type(scope): brief description

Longer description if needed

- Bullet points for details
- Reference issues with #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(agents): add persistent memory support`
- `fix(http): handle connection timeouts properly`
- `docs(readme): update quick start example`

## Code Style

pydantic-flow follows strict code style guidelines for consistency and type safety:

### Type Annotations

- **Full type annotations everywhere** - no exceptions
- **Modern syntax only:**
  - Use `dict`/`list`/`tuple` never `Dict`/`List`/`Tuple`
  - Use `A | B` never `Union[A, B]`
  - Use `A | None` never `Optional[A]`

```python
# ✅ Good
async def process_data(items: list[str]) -> dict[str, int] | None:
    pass

# ❌ Bad
async def process_data(items: List[str]) -> Optional[Dict[str, int]]:
    pass
```

### Data Structures

- **Use Pydantic models and dataclasses** for all public interfaces
- **No bare tuples/dicts** in function signatures or return values
- **Enums over strings** wherever possible

```python
# ✅ Good
from enum import Enum
from pydantic import BaseModel

class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"

class AgentResult(BaseModel):
    status: AgentStatus
    message: str
    data: dict[str, str]

async def run_agent() -> AgentResult:
    pass

# ❌ Bad
async def run_agent() -> tuple[str, str, dict]:
    pass
```

### Imports

- **Absolute imports only** - no relative imports
- **One symbol per line**
- **Sort imports** (handled by ruff)

```python
# ✅ Good
from pydantic_flow.agents.base import Agent
from pydantic_flow.tools.http import HttpTool
from pydantic_flow.types import AgentConfig

# NOT this
from pydantic_flow.agents.base import Agent, BaseAgent

# ❌ Bad - Don't do this
from pydantic_flow.agents.base import Agent, BaseAgent
from .tools import HttpTool
```

### Error Handling

- **Fail fast** with descriptive exceptions
- **Custom exceptions** with rich context
- **No silent failures**

```python
# ✅ Good
class AgentConfigurationError(Exception):
    def __init__(self, agent_id: str, missing_fields: list[str]):
        self.agent_id = agent_id
        self.missing_fields = missing_fields
        super().__init__(
            f"Agent {agent_id} missing required fields: {', '.join(missing_fields)}"
        )

# ❌ Bad
if not config:
    return None  # Silent failure
```

## Testing

### Test Requirements

- **Comprehensive coverage** - aim for >95%
- **Fast execution** - entire suite should run quickly
- **Type safety** - tests must pass type checking

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=pydantic_flow --cov-report=term-missing

# Run specific test
uv run pytest tests/test_agents.py::test_agent_creation

# Run tests in watch mode
uv run pytest --watch
```

### Writing Tests

- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Test both success and failure cases
- Mock external dependencies

```python
async def test_agent_processes_message_successfully():
    # Arrange
    agent = Agent(model="test-model")
    message = "Hello, agent!"
    
    # Act
    result = await agent.process(message)
    
    # Assert
    assert result.status == AgentStatus.SUCCESS
    assert "hello" in result.message.lower()
```

## Documentation

### Docstrings

Use Google-style docstrings for all public APIs:

```python
async def create_agent(
    model: str,
    tools: list[str] | None = None,
    memory: bool = True
) -> Agent:
    """Create a new AI agent with specified configuration.
    
    Args:
        model: The AI model to use (e.g., "gpt-4", "claude-3").
        tools: List of tool names to enable. Defaults to basic tools.
        memory: Whether to enable persistent memory. Defaults to True.
        
    Returns:
        Configured agent ready for use.
        
    Raises:
        AgentConfigurationError: If model is not supported.
        ToolNotFoundError: If specified tool is not available.
        
    Example:
        >>> agent = await create_agent("gpt-4", tools=["http", "fs"])
        >>> result = await agent.run("Analyze this website")
    """
```

### Documentation Updates

- Update relevant docs for API changes
- Add examples for new features
- Keep AGENTS.md current with architectural changes

## Submitting Changes

### Pull Request Process

1. **Ensure tests pass:**
   ```bash
   uv run pytest
   uv run ruff check
   uv run ruff format
   ```

2. **Update documentation** if needed

3. **Create pull request** with:
   - Clear title and description
   - Reference related issues
   - List breaking changes
   - Include examples if adding features

## Related Issues
Fixes #123
```

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

EXCEPTION: While MAJOR is version 0, any change can be breaking.

### Release Checklist

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Create release PR
5. Tag release after merge
6. Publish to PyPI

## Questions?

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Security**: Email security issues privately

Thank you for contributing to pydantic-flow! 🚀
