# pflow: A Type-Safe Pydantic-AI Workflow Framework

A modern, type-safe, composable Python framework for building AI workflows using Pydantic models as inputs and outputs for each processing node.

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## 🌟 Features

- **Type-Safe by Design**: Full Python 3.14+ generics support with comprehensive IDE integration
- **Pydantic-Powered**: Built around `BaseModel` for schema definition and validation
- **Reference-Based Wiring**: Connect nodes using `.output` references, not magic strings
- **Automatic DAG Resolution**: Intelligent dependency tracking and execution ordering
- **Serializable & Inspectable**: Full observability and debugging support
- **Built-in Node Types**: Comprehensive set of workflow building blocks

## 🚀 Quick Start

### Installation

```bash
pip install pflow
```

### Basic Example

```python
import asyncio
from pydantic import BaseModel
from pflow import Flow, PromptNode, ParserNode, ToolNode

# Define your data models
class WeatherQuery(BaseModel):
    location: str
    temperature_unit: str = "celsius"

class WeatherInfo(BaseModel):
    temperature: float
    condition: str
    location: str

# Define transformation functions
def call_weather_api(query: WeatherQuery) -> WeatherInfo:
    # Your weather API integration
    return WeatherInfo(
        temperature=22.5,
        condition="sunny", 
        location=query.location
    )

def parse_weather_response(response: str) -> WeatherInfo:
    # Parse LLM response into structured data
    parts = response.split("|")
    return WeatherInfo(
        temperature=float(parts[0]),
        condition=parts[1].strip(),
        location=parts[2].strip()
    )

async def main():
    # Create workflow nodes
    api_node = ToolNode[WeatherQuery, WeatherInfo](
        tool_func=call_weather_api,
        name="weather_api"
    )
    
    llm_node = PromptNode[WeatherQuery, str](
        prompt="What's the weather like in {location}?",
        name="weather_llm"
    )
    
    parser_node = ParserNode[str, WeatherInfo](
        parser_func=parse_weather_response,
        input=llm_node.output,  # Type-safe wiring!
        name="weather_parser"
    )
    
    # Create and run workflow
    flow = Flow()
    flow.add_nodes(api_node)  # Simple API-based flow
    
    # Execute with type safety
    query = WeatherQuery(location="Paris")
    results = await flow.run(query)
    
    weather = results["weather_api"]  # Fully typed!
    print(f"Temperature: {weather.temperature}°C")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🧩 Core Concepts

### Nodes

Nodes are the building blocks of your workflow. Each node is typed with input and output models:

#### PromptNode[InputModel, OutputType]
Calls an LLM using a templated prompt:

```python
weather_prompt = PromptNode[WeatherQuery, str](
    prompt="Get weather for {location} in {temperature_unit}",
    model="openai:gpt-4",
    name="weather_prompt"
)
```

#### ParserNode[InputType, OutputModel]
Applies a function to transform data:

```python
weather_parser = ParserNode[str, WeatherInfo](
    parser_func=parse_weather_text,
    input=weather_prompt.output,
    name="weather_parser"
)
```

#### ToolNode[InputModel, OutputModel]
Calls external tools/APIs:

```python
weather_api = ToolNode[WeatherQuery, WeatherInfo](
    tool_func=call_weather_service,
    name="weather_api"
)
```

#### RetryNode[OutputModel]
Wraps nodes with retry logic:

```python
reliable_api = RetryNode(
    wrapped_node=weather_api,
    max_retries=3,
    name="reliable_weather"
)
```

#### IfNode[OutputModel]
Enables conditional branching:

```python
weather_branch = IfNode(
    predicate=lambda data: data.temperature > 25,
    if_true=hot_weather_node,
    if_false=normal_weather_node,
    input=weather_api.output,
    name="weather_condition"
)
```

### Flow Orchestration

The `Flow` class manages execution with automatic dependency resolution:

```python
flow = Flow()
flow.add_nodes(node1, node2, node3)

# Automatic topological sorting
execution_order = flow.get_execution_order()

# Type-safe execution
results = await flow.run(input_data)
```

### Sub-flow Composition with FlowNode

The `FlowNode` enables hierarchical composition by wrapping complete flows as nodes:

```python
# Create a reusable sub-flow
research_flow = Flow(input_type=UserQuery, output_type=ResearchResults)
research_node = ToolNode[UserQuery, ResearchData](
    tool_func=research_api,
    name="research"
)
research_flow.add_nodes(research_node)

# Wrap the sub-flow as a node
research_flow_node = FlowNode[UserQuery, ResearchResults](
    flow=research_flow,
    name="research_sub_flow"
)

# Use in a parent flow
parent_flow = Flow(input_type=UserQuery, output_type=FinalResults)
summary_node = ToolNode[ResearchResults, SummaryData](
    tool_func=generate_summary,
    input=research_flow_node.output,  # Chain sub-flows
    name="summary"
)

parent_flow.add_nodes(research_flow_node, summary_node)

# Execute hierarchical workflow
results = await parent_flow.run(query)
research_data = results.research_sub_flow.research
summary_data = results.summary
```

Key benefits of sub-flow composition:
- **Hierarchical Architecture**: Build complex workflows from simpler, reusable components
- **Encapsulation**: Sub-flows hide internal complexity behind clean interfaces
- **Reusability**: Same sub-flow can be used in multiple parent flows
- **Type Safety**: Full type checking across sub-flow boundaries
- **Testing**: Sub-flows can be tested in isolation

## 🎯 Design Philosophy

### Type Safety First
```python
# Full IDE support with autocomplete
weather_node: PromptNode[WeatherQuery, str] = ...
parsed_node: ParserNode[str, WeatherInfo] = ...

# Compile-time type checking
weather_info = results["weather_api"]  # Type: WeatherInfo
temperature = weather_info.temperature  # Type: float
```

### Reference-Based Wiring
```python
# ✅ Type-safe references
parser_node = ParserNode(
    parser_func=parse_data,
    input=prompt_node.output  # NodeOutput[str]
)

# ❌ No magic strings
parser_node = ParserNode(
    parser_func=parse_data,
    input="prompt_node.output"  # Fragile!
)
```

### Pydantic Integration
```python
class MyInput(BaseModel):
    query: str
    options: list[str] = []

class MyOutput(BaseModel):
    result: str
    confidence: float
    
# Automatic validation and serialization
node = ToolNode[MyInput, MyOutput](...)
```

## 🔧 Advanced Features

### Complex Workflows
```python
# Multi-path workflow with branching
flow = Flow()

# Initial processing
data_node = ToolNode[Input, Data](...)

# Conditional branching  
branch_node = IfNode(
    predicate=lambda x: x.value > threshold,
    if_true=high_value_processor,
    if_false=low_value_processor,
    input=data_node.output
)

# Merge results
final_node = ParserNode[ProcessedData, Output](
    input=branch_node.output,
    ...
)

flow.add_nodes(data_node, branch_node, final_node)
```

### Error Handling
```python
# Automatic retries with backoff
retry_node = RetryNode(
    wrapped_node=unreliable_api,
    max_retries=5
)

# Flow-level error handling
try:
    results = await flow.run(input_data)
except FlowError as e:
    print(f"Workflow failed: {e}")
```

### Observability
```python
# Inspect execution order
print(flow.get_execution_order())

# Access intermediate results
results = await flow.run(input_data)
intermediate = results["parser_node"]
final = results["summary_node"]
```

## 🏗️ Architecture

Built on modern Python foundations:

- **Python 3.14+**: Latest type system features
- **Pydantic**: Data validation and serialization  
- **Async-First**: All operations are async by default
- **Modern Generics**: `Node[Input, Output]` syntax
- **Entry Points**: Plugin system for extensibility

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Related Projects

- [pydantic-ai](https://ai.pydantic.dev/) - The foundational AI framework we build upon
- [Pydantic](https://pydantic.dev/) - Data validation using Python type annotations

---

**Made with ❤️ for the Python AI community**

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
- Modern Python 3.14+ syntax (`A | B`, `dict`/`list`/`tuple`)
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
- Python 3.14+
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
