# Agents

## Vision

**pflow** is a type-safe, batteries-included AI agent framework built on [pydantic-ai](https://ai.pydantic.dev/) and designed to rival LangChain while maintaining the developer experience and type safety that the Pydantic ecosystem is known for.

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

- **Python 3.13+**: Leveraging modern Python features
- **Modern Type Syntax**: `dict`/`list`/`tuple` over `Dict`/`List`/`Tuple`, `A | B` over `Union[A,B]`, `A | None` over `Optional[A]`
- **Structured Data**: Pydantic models and dataclasses for all public interfaces - no bare tuples/dicts
- **Enumerations**: Type-safe enums over hard-coded strings wherever possible
- **Absolute Imports**: One symbol per line, no relative imports
- **Pydantic Configuration**: All config backed by Pydantic models with environment variable support
- **Comprehensive Testing**: Fast-running test suite critical to developer experience

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
- **Tool Providers**: `pflow.plugins` entry point for automatic plugin discovery and integration
- **Type Safety**: Protocol-based tool interfaces with full type hints
- **Dependency Injection**: Automatic wiring of dependencies and configurations
- **Opinionated Monolith**: Core pflow includes comprehensive batteries while supporting external extensions

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
- **Dual Import Paths**: Support both `from pflow import Agent` and `from pflow.agents import Agent`
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

This document defines the WHY and HOW of pflow agents, not the WHAT. Implementation details and specific code blueprints are intentionally omitted to allow for organic evolution while maintaining architectural coherence.

The framework aims to provide the missing batteries for pydantic-ai users who want to build production-ready AI agents without sacrificing type safety or developer experience.
