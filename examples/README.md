# pflow Examples

This directory contains example scripts demonstrating various features of the pflow framework.

## Available Examples

### 📝 `example.py`
**Complete Weather Workflow Demo**

A comprehensive demonstration of the pflow framework featuring:
- Type-safe node composition with generics
- Multiple workflow patterns (API-based and LLM-based)
- Automatic dependency resolution
- Pydantic model validation
- DAG execution ordering

**Run with:**
```bash
cd examples
uv run python example.py
```

**Features demonstrated:**
- `ToolNode` for API calls
- `PromptNode` for LLM interactions
- `ParserNode` for data transformation
- Flow orchestration with multiple nodes
- Strongly typed input/output models

### 🔍 `type_safety_demo.py`
**Type Safety Demonstration**

Shows the improved type safety features in the Flow class:
- Generic type parameters `Flow[InputT, OutputT]`
- BaseModel constraints for inputs and outputs
- IDE auto-completion support
- Compile-time type checking

**Run with:**
```bash
cd examples
uv run python type_safety_demo.py
```

**Features demonstrated:**
- Strongly typed flow creation
- BaseModel output validation
- Type annotation patterns
- Type safety benefits

## Getting Started

1. Navigate to the examples directory:
   ```bash
   cd examples
   ```

2. Run any example:
   ```bash
   uv run python example.py
   # or
   uv run python type_safety_demo.py
   ```

3. Examine the source code to understand how pflow components work together

## Requirements

All examples use the same dependencies as the main pflow project:
- Python 3.13+
- pydantic for data validation
- The pflow framework itself

The examples are designed to work out of the box with the project's existing environment setup.