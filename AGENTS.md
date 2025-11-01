# Agents

## Vision

**pydantic-flow** is a type-safe, batteries-included AI agent framework built on [pydantic-ai](https://ai.pydantic.dev/) and designed to rival LangChain while maintaining the developer experience and type safety that the Pydantic ecosystem is known for.

The framework follows pydantic-ai's lead in being developer-experience focused, leveraging open source and open standards to create a comprehensive toolkit for building complex AI agents from 0 to 100.

## Architecture Philosophy

### Core Principles

- **Type Safety First**: Full type annotations everywhere with comprehensive IDE support
- **Streaming-Native**: Primary APIs expose async streams of progress; non-streaming via convenience wrappers
- **Async-First Design**: Primary APIs are async with sync wrappers where appropriate
- **Dependency Injection**: Auto-discovery of plugins and tools with explicit configuration
- **Fail Fast**: Rich exception context with custom exceptions for clear error handling
- **Immutable State**: Prefer idempotent APIs over stateful ones where possible
- **Long-Lived Agents**: Agents as persistent objects with integrated memory, tools, and durability
- **Small Progress Vocabulary**: Focused set of progress items (tokens, tools, retrievals, fields, errors)

### Technical Standards

- **Python 3.14+**: Leveraging modern Python features
- **Modern Type Syntax**: `dict`/`list`/`tuple` over `Dict`/`List`/`Tuple`, `A | B` over `Union[A,B]`, `A | None` over `Optional[A]`
- **Structured Data**: Pydantic models for all public interfaces - no bare tuples/dicts
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
- **Streaming Execution**: Progress visibility through async streams of events

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

- **Streaming First**: `astream()` as primary interface with `run()` convenience wrapper
- **Direct Instantiation**: `Agent(...)` over builder patterns
- **Dual Import Paths**: Support both `from pydantic_flow import Agent` and `from pydantic_flow.agents import Agent`
- **Functional + Class-Based**: Solid class foundation with functional helpers
- **Method Clarity**: Explicit operations over implicit state changes
- **Progress Visibility**: Small vocabulary of progress items for observability

### Error Handling

- **Custom Exceptions**: Rich context with framework-specific error types
- **Validation**: Pydantic model validation with detailed error messages
- **Fail Fast**: Early detection and clear error propagation

### Configuration

- **Pydantic Models**: All configuration through validated models
- **Environment Integration**: Environment variables with Pydantic backing
- **Type Validation**: Runtime validation with IDE support

### Flow Construction

The framework uses a **unified node-reference API** where users always work with node objects rather than string identifiers. The framework automatically selects the optimal execution engine based on flow structure analysis.

#### Unified API Design

All flow construction uses direct node references:
- `flow.add_edge(source_node, target_node)` - Node objects, not strings
- `flow.set_entry_nodes(node1, node2)` - Node objects as entry points  
- Router functions return `BaseNode | Route` for control flow decisions
- Automatic engine selection via graph analysis (cycle detection, conditional edge detection)

#### Dual Execution Engines

The framework maintains two execution engines and automatically selects the optimal one:

**DAG Engine:**
- Uses topological sort (O(V+E) complexity)
- Optimal for acyclic flows without conditional routing
- Entry nodes inferred from nodes with no incoming edges
- Direct execution without control flow overhead

**Stepper Engine:**
- Frontier-based wave execution
- Required for cycles, loops, and conditional routing
- Supports `Route.END` sentinel for termination
- Handles dynamic control flow patterns

#### Automatic Engine Selection

Graph analysis determines the appropriate engine:
- **Auto mode** (default): Detects cycles or conditional edges, uses stepper if needed
- **DAG mode**: Forces topological sort, errors if cycles/routing detected  
- **Stepper mode**: Forces stepper engine for all flows

Selection can be overridden via `flow.compile(mode=ExecutionMode.DAG|STEPPER|AUTO)`.

Users can inspect selection reasoning via `compiled_flow.explain()` which returns:
- Selected engine mode
- Detected features (cycles, conditional edges, explicit edges)
- Inferred entry nodes
- Human-readable selection reasons

#### Type Safety Benefits

The node-reference approach provides:
- Full IDE autocomplete and type checking
- Compile-time validation of node connections
- Refactoring safety - rename detection across codebase
- Clear error messages when nodes are not found

### Unified Checkpoint System

The framework implements a **unified checkpoint system** that serves both debugging and HITL use cases:

**StateSnapshot as Universal Model:**
- Single checkpoint type (`StateSnapshot`) for all scenarios: HITL interrupts, debugging, manual pauses, errors
- Reason-based categorization via `SnapshotReason` enum (HITL_INTERRUPT, AUTOMATIC, MANUAL_PAUSE, ERROR, COMPLETION)
- Full state preservation with optional delta compression for efficiency
- Metadata support for custom interrupt context

**Conversation Memory as Linked List:**
- Append-only message chain with snapshots referencing HEAD pointer
- Messages stored once, referenced by multiple snapshots
- Enables conversation reconstruction at any snapshot point
- Natural branching for forked execution paths

**Query and Inspection APIs:**
- `CheckpointInspector`: Read-only data access for interrupted runs, snapshots, and conversations
- `CheckpointDebugger`: High-level workflows with Rich rendering
- CLI commands: `list-interrupts`, `show-interrupt`, `resume-with-decision`

**Universal Resume:**
- Single `resume_from_snapshot()` method works for all snapshot types
- Reconstructs state from deltas when needed
- Restores conversation context automatically
- Type-safe with full StateSnapshot object or ID-based lookup

**Storage Backends:**
- SQLite (local development), Postgres (production), S3/Filesystem (archival)
- Conversation messages tracked separately with reference tables
- Protected message identification prevents unsafe pruning

This unified approach eliminates code duplication, enables powerful time-travel debugging of HITL sessions, and provides a consistent interface for all checkpoint operations.



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
