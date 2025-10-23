# Agents

## Vision

**pydantic-flow** is a type-safe, batteries-included AI agent framework built on [pydantic-ai](https://ai.pydantic.dev/) and designed to rival LangChain while maintaining the developer experience and type safety that the Pydantic ecosystem is known for.

The framework follows pydantic-ai's lead in being developer-experience focused, leveraging open source and open standards to create a comprehensive toolkit for building complex AI agents from 0 to 100.

## Architecture Philosophy

### Core Principles

- **Type Safety First**: Full type annotations everywhere with comprehensive IDE support
- **Async-First Design**: Primary APIs are async with sync wrappers where appropriate
- **Dependency Injection**: Auto-discovery of plugins and tools with explicit configuration
- **Fail Fast**: Rich exception context with custom exceptions for clear error handling
- **Immutable State**: Prefer idempotent APIs over stateful ones where possible
- **Long-Lived Agents**: Agents as persistent objects with integrated memory, tools, and durability

### Technical Standards

- **Python 3.14+**: Leveraging modern Python features
- **Modern Type Syntax**: `dict`/`list`/`tuple` over `Dict`/`List`/`Tuple`, `A | B` over `Union[A,B]`, `A | None` over `Optional[A]`
- **Structured Data**: Pydantic models and dataclasses for all public interfaces - no bare tuples/dicts
- **Enumerations**: Type-safe enums over hard-coded strings wherever possible
- **Absolute Imports**: One symbol per line, no relative imports
- **Pydantic Configuration**: All config backed by Pydantic models with environment variable support
- **Comprehensive Testing**: Fast-running test suite critical to developer experience
- **Comments**: Use sparingly and only to explain the _WHY_. If a section of code is complex enough to warrant a _HOW_ comment, it should be extracted. **NEVER** leave comments which explain differences between before/after edits.
- **Documentation**: Clear, concise, and minimal examples that get the point across without overwhelming the reader. NEVER generate change summary documents: instead update existing documentation (README.md, AGENTS.md) from a present-tense perspective.
- **Tests**: All new features must include tests that validate functionality and type safety. Only start class names with `Test` if they are test classes.


## Agent Architecture

### Agent Lifecycle

Agents are designed as long-lived objects that maintain state and context across conversations:

- **Persistent State**: Memory, tool configurations, and context preservation
- **Idempotent Operations**: State changes through explicit operations, not side effects
- **Tool Integration**: Auto-discovered plugins with type-safe interfaces
- **Memory Management**: Built-in support for conversation history and context

### Plugin System

The framework uses entry points for auto-discovery, designed for seamless external library integration:

- **External Plugin Ecosystem**: Architected like pytest's ecosystem (pytest-cov, pytest-asyncio) for easy third-party plugin creation
- **Tool Providers**: `pydantic_flow.plugins` entry point for automatic plugin discovery and integration
- **Type Safety**: Protocol-based tool interfaces with full type hints
- **Dependency Injection**: Automatic wiring of dependencies and configurations
- **Opinionated Monolith**: Core pydantic-flow includes comprehensive batteries while supporting external extensions

### Key Technologies

- **pydantic-ai**: Core AI agent functionality and type safety
- **Pydantic**: Data validation and settings management
- **OpenTelemetry**: Observability and tracing
- **Rich**: Enhanced terminal output and debugging
- **Typer**: CLI interface for agent management
- **SQLite**: Local persistence and state management
- **AnyIO**: Async abstraction layer

## Development Patterns

### API Design

- **Direct Instantiation**: `Agent(...)` over builder patterns
- **Dual Import Paths**: Support both `from pydantic_flow import Agent` and `from pydantic_flow.agents import Agent`
- **Functional + Class-Based**: Solid class foundation with functional helpers
- **Method Clarity**: Explicit operations over implicit state changes

### Error Handling

- **Custom Exceptions**: Rich context with framework-specific error types
- **Validation**: Pydantic model validation with detailed error messages
- **Fail Fast**: Early detection and clear error propagation

### Configuration

- **Pydantic Models**: All configuration through validated models
- **Environment Integration**: Environment variables with Pydantic backing
- **Type Validation**: Runtime validation with IDE support

### Flow Construction

The framework supports two distinct patterns for building workflows, each optimized for different use cases:

#### Implicit DAG Pattern

Node dependencies are expressed through constructor parameters, with execution order determined automatically via topological sort. Best for linear, acyclic workflows where dependencies flow naturally from data transformations.

**Characteristics:**
- Entry nodes inferred from nodes with no dependencies
- Execution order calculated from node dependency graph
- Uses efficient topological sort algorithm
- Minimal API surface - just add nodes, system handles the rest

**Trade-offs:**
- Cannot express cycles or loops
- No conditional routing
- Simpler mental model for straightforward pipelines

#### Explicit Edge Pattern

Edges and entry points are declared explicitly, with execution managed by the stepper engine. Required for flows with loops, conditional routing, or complex control flow.

**Characteristics:**
- Entry nodes must be explicitly declared via `set_entry_nodes()`
- Edges defined via `add_edge()` and `add_conditional_edges()`
- Stepper engine executes nodes in frontier-based waves
- Supports cycles and dynamic routing via `Route.END` sentinel

**Trade-offs:**
- More verbose API (entry points, explicit edges)
- Slightly higher execution overhead (stepper vs direct DAG execution)
- Enables complex control flow patterns impossible in DAG mode

#### Engine Selection

The framework automatically selects the appropriate execution engine:
- **Auto mode** (default): Detects cycles or conditional edges, uses stepper if needed
- **DAG mode**: Forces topological sort, errors if cycles/routing detected
- **Stepper mode**: Forces stepper engine for all flows

Users can override via `flow.compile(mode=ExecutionMode.DAG|STEPPER|AUTO)`.

#### Pattern Purity

**Mixing patterns within a single flow is discouraged** as it creates ambiguity about execution order and control flow. When both patterns are needed, use sub-flows (FlowNode) to compose them as separate, encapsulated units.

## Goals

### Batteries Included

Provide a comprehensive ecosystem that rivals LangChain:

- **Core Monolith**: Opinionated framework with extensive built-in capabilities for complete agent development
- **Tools**: HTTP, filesystem, vector operations, and extensible plugin system
- **Memory**: Persistent conversation history and context management
- **Observability**: Built-in tracing and monitoring with OpenTelemetry
- **Durability**: State persistence and recovery mechanisms
- **CLI**: Complete command-line interface for agent management
- **External Ecosystem**: Plugin architecture enabling third-party extensions while maintaining comprehensive core functionality

### Developer Experience

- **Type Safety**: Full IDE integration with auto-completion and error detection
- **Fast Feedback**: Comprehensive but lightning-fast test suite
- **Clear APIs**: Intuitive interfaces with minimal cognitive overhead
- **Rich Documentation**: Comprehensive guides with minimal but effective examples

### Open Standards

Built on open source foundations:

- **Pydantic Ecosystem**: Leveraging the mature validation and serialization ecosystem
- **Standard Protocols**: Using established patterns for plugin and tool interfaces
- **Community Integration**: Compatible with existing Python AI/ML tooling

## Framework Scope

This document defines the WHY and HOW of pydantic-flow agents, not the WHAT. Implementation details and specific code blueprints are intentionally omitted to allow for organic evolution while maintaining architectural coherence.

The framework aims to provide the missing batteries for pydantic-ai users who want to build production-ready AI agents without sacrificing type safety or developer experience.
