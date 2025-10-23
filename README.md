# pydantic-flow: A Type-Safe Pydantic-AI Workflow Framework

A modern, type-safe, composable Python framework for building AI workflows using Pydantic models as inputs and outputs for each processing node.

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: ty](https://img.shields.io/badge/type%20checked-ty-blue.svg)](https://github.com/pydantic/ty)
[![Built with: uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

## 🌟 Features

- **Type-Safe by Design**: Full Python 3.14+ generics support with comprehensive IDE integration
- **Pydantic-Powered**: Built around `BaseModel` for schema definition and validation
- **Reference-Based Wiring**: Connect nodes using `.output` references, not magic strings
- **Automatic DAG Resolution**: Intelligent dependency tracking and execution ordering using topological sorting ([learn more](docs/dag_resolution.md))
- **Production-Ready Prompt Library**: Standalone templating system with multiple engines (f-string, Jinja2, Mustache), structured I/O, and observability
- **Serializable & Inspectable**: Full observability and debugging support with OpenTelemetry integration
- **Built-in Node Types**: Comprehensive set of workflow building blocks

## 🚀 Quick Start

### Installation

```bash
pip install pydantic-flow
```

### Basic Example

```python
import asyncio
from pydantic import BaseModel
from pydantic_flow import Flow, PromptNode, ParserNode, ToolNode

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

#### MergeToolNode & MergeParserNode
Enable fan-in patterns where multiple node outputs combine into one:

```python
# Multiple nodes producing different data
analysis_node = ToolNode[Input, Analysis](
    tool_func=analyze_data,
    name="analysis"
)

facts_node = ToolNode[Input, Facts](
    tool_func=gather_facts,
    name="facts"
)

metadata_node = ToolNode[Input, Metadata](
    tool_func=get_metadata,
    name="metadata"
)

# Merge multiple inputs into single output
def combine_all(analysis: Analysis, facts: Facts, metadata: Metadata) -> Report:
    return Report(
        analysis=analysis,
        facts=facts,
        metadata=metadata
    )

merge_node = MergeToolNode[Analysis, Facts, Metadata, Report](
    inputs=(analysis_node.output, facts_node.output, metadata_node.output),
    tool_func=combine_all,
    name="merge"
)

# MergePromptNode for combining inputs into LLM prompts
merge_prompt = MergePromptNode[Analysis, Facts, str](
    inputs=(analysis_node.output, facts_node.output),
    prompt="Analyze: {analysis}\nFacts: {facts}\nProvide summary:",
    model="openai:gpt-4",
    name="llm_merge"
)

# Continue processing merged result
final_node = ToolNode[Report, FinalOutput](
    tool_func=finalize_report,
    input=merge_node.output,
    name="final"
)

# Pattern: A -> B,C,D -> E -> F
flow.add_nodes(analysis_node, facts_node, metadata_node, merge_node, final_node)
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

### Loops and Conditional Routing

pydantic-flow supports cycles and conditional control flow using a stepper-based execution engine. This enables patterns like while-loops, retry logic, and dynamic routing based on state.

#### Self-Loop Counter Example

A node that loops back to itself until a condition is met:

```python
from pydantic import BaseModel
from pydantic_flow import Flow, Route, RunConfig
from pydantic_flow.core.routing import T_Route
from pydantic_flow.nodes import BaseNode

class CounterState(BaseModel):
    n: int

class OutputState(BaseModel):
    tick: CounterState

class TickNode(BaseNode[CounterState, CounterState]):
    async def run(self, input_data: CounterState) -> CounterState:
        return CounterState(n=input_data.n + 1)

# Create flow with loop
flow = Flow(input_type=CounterState, output_type=OutputState)
tick_node = TickNode(name="tick")
flow.add_nodes(tick_node)
flow.set_entry_nodes("tick")

# Add conditional routing
def router(state: BaseModel) -> T_Route:
    tick_state = getattr(state, "tick", None)
    if tick_state and tick_state.n >= 5:
        return Route.END  # Terminate
    return "tick"  # Loop back

flow.add_conditional_edges("tick", router)

# Compile and execute
compiled = flow.compile()
config = RunConfig(max_steps=50)
result = await compiled.invoke(CounterState(n=0), config)
# result.tick.n == 5
```

#### Two-Node While-Loop Example

Plan-execute pattern with conditional routing:

```python
from pydantic import BaseModel
from pydantic_flow import Flow, Route, RunConfig
from pydantic_flow.nodes import BaseNode

class WorkState(BaseModel):
    iterations: int
    total: int

class FullOutput(BaseModel):
    plan: WorkState
    execute: WorkState

class PlanNode(BaseNode[WorkState, WorkState]):
    async def run(self, input_data: WorkState) -> WorkState:
        return WorkState(
            iterations=input_data.iterations + 1,
            total=input_data.total
        )

class ExecuteNode(BaseNode[WorkState, WorkState]):
    async def run(self, input_data: WorkState) -> WorkState:
        new_total = input_data.total + (input_data.iterations * 10)
        return WorkState(iterations=input_data.iterations, total=new_total)

# Create flow with two-node loop
flow = Flow(input_type=WorkState, output_type=FullOutput)
plan_node = PlanNode(name="plan")
execute_node = ExecuteNode(name="execute")

flow.add_nodes(plan_node, execute_node)
flow.set_entry_nodes("plan")
flow.add_edge("plan", "execute")

# Router decides: loop back to plan or END
def router(state: BaseModel) -> T_Route:
    execute_state = getattr(state, "execute", None)
    if execute_state and execute_state.iterations >= 5:
        return Route.END
    return "plan"

flow.add_conditional_edges("execute", router)

# Execute with safety limits
compiled = flow.compile()
config = RunConfig(max_steps=50, trace_iterations=True)
result = await compiled.invoke(WorkState(iterations=0, total=0), config)
```

#### Routing with Mapping Dictionaries

Map router outcomes to target nodes:

```python
def boolean_router(state: BaseModel) -> str:
    # Return a key that maps to a target node
    should_proceed = getattr(state, "ready", False)
    return "proceed" if should_proceed else "retry"

flow.add_conditional_edges(
    "checker",
    boolean_router,
    mapping={"proceed": "next_step", "retry": "checker"}
)
```

#### Error Handling

Production flows should handle potential errors gracefully. Here's a comprehensive error handling pattern:

```python
from pydantic_flow import (
    Flow, Route, RunConfig, 
    RecursionLimitError, RoutingError, FlowTimeoutError, FlowError
)
import logging

logger = logging.getLogger(__name__)

async def run_with_error_handling():
    """Example of proper error handling for loop-capable flows."""
    flow = create_my_flow()
    compiled = flow.compile()
    
    config = RunConfig(
        max_steps=100,
        timeout_seconds=60,
        trace_iterations=True
    )
    
    try:
        result = await compiled.invoke(initial_state, config)
        logger.info(f"Flow completed successfully: {result}")
        return result
        
    except RecursionLimitError as e:
        # Flow exceeded max_steps - likely an infinite loop
        logger.error(f"Flow hit iteration limit: {e}")
        # Consider: save partial state, alert monitoring, etc.
        raise
        
    except FlowTimeoutError as e:
        # Flow took too long - may need more time or optimization
        logger.error(f"Flow execution timed out: {e}")
        # Consider: increase timeout, optimize nodes, add caching
        raise
        
    except RoutingError as e:
        # Invalid router configuration - this is a programming error
        logger.error(f"Router configuration error: {e}")
        # This should be caught in testing - fix your routers!
        raise
        
    except FlowError as e:
        # Other flow execution errors
        logger.error(f"Flow execution failed: {e}")
        raise
        
    except Exception as e:
        # Unexpected errors (should be rare)
        logger.exception(f"Unexpected error in flow execution: {e}")
        raise
```

**Common Error Scenarios:**

- **RecursionLimitError**: Usually indicates a router that never returns `Route.END`. Check your termination conditions.
- **FlowTimeoutError**: Either increase `timeout_seconds` or optimize slow nodes. Check node execution times.
- **RoutingError**: Programming error - router returned invalid node name. Validate router logic and use mapping dicts for type safety.

**Quick Examples:**

```python
# RecursionLimitError: Raised when max_steps is exceeded
config = RunConfig(max_steps=25)  # Default limit
try:
    result = await compiled.invoke(input_data, config)
except RecursionLimitError as e:
    print(f"Loop exceeded limit: {e}")
    # e includes recent iteration trace

# RoutingError: Raised when router returns an invalid target
def bad_router(state: BaseModel) -> str:
    return "nonexistent_node"  # RoutingError!

flow.add_conditional_edges("start", bad_router)

# FlowTimeoutError: Raised when execution exceeds time limit
config = RunConfig(timeout_seconds=30)
try:
    result = await compiled.invoke(input_data, config)
except FlowTimeoutError:
    print("Flow execution timed out")
```

#### Key Concepts

- **Route.END**: Special sentinel to terminate flow execution
- **Conditional Edges**: Dynamic routing based on current state
- **RunConfig**: Control `max_steps`, `timeout_seconds`, and `trace_iterations`
- **Stepper Engine**: Automatically selected for flows with cycles or conditional edges
- **Type Safety**: Router functions receive and return typed values

See `examples/loops/` for complete runnable examples.

### Flow Construction Patterns

pydantic-flow supports two patterns for building flows, each suited to different use cases:

#### Pattern 1: Implicit DAG (Node Dependencies)

Best for **acyclic workflows** where dependencies are clear and linear:

```python
from pydantic_flow.nodes import NodeWithInput

class ProcessNode(NodeWithInput[DataType, ResultType]):
    """Node with explicit input dependency."""
    
    def __init__(self, input_node: BaseNode[Any, DataType]):
        self.input = input_node.output
        super().__init__(name="process")
    
    async def run(self, input_data: DataType) -> ResultType:
        # Process the data from input_node
        return process(input_data)

# Build flow - execution order determined automatically
flow = Flow(input_type=InputType, output_type=OutputType)
fetch = FetchNode(name="fetch")
process = ProcessNode(input_node=fetch)  # Dependency declared in constructor
transform = TransformNode(input_node=process)

flow.add_nodes(fetch, process, transform)
# No need to call set_entry_nodes() - inferred from dependencies
# No explicit edges needed - determined via topological sort
result = await flow.run(input_data)
```

**When to use:**
- Linear or tree-like data pipelines
- No loops or conditional routing needed
- Dependencies naturally expressed through node constructors

#### Pattern 2: Explicit Edges (Stepper Engine)

Required for **loops, conditional routing, or complex control flow**:

```python
flow = Flow(input_type=StateType, output_type=StateType)

# Add all nodes first
plan = PlanNode(name="plan")
execute = ExecuteNode(name="execute")
flow.add_nodes(plan, execute)

# Explicitly define entry point (required for stepper engine)
flow.set_entry_nodes("plan")

# Add static edges
flow.add_edge("plan", "execute")

# Add conditional routing
def should_continue(state: BaseModel) -> T_Route:
    execute_state = getattr(state, "execute")
    if execute_state.iterations >= 5:
        return Route.END
    return "plan"  # Loop back to start

flow.add_conditional_edges("execute", should_continue)

# Compile (automatically uses stepper engine due to conditional edges)
compiled = flow.compile()
result = await compiled.invoke(initial_state, config)
```

**When to use:**
- Loops or cycles in the workflow
- Conditional routing based on state
- Complex control flow patterns
- Multiple entry points

#### Choosing the Right Mode

You can also explicitly control which engine to use:

```python
from pydantic_flow import ExecutionMode

# Force DAG mode (will error if cycles/conditional edges exist)
compiled = flow.compile(mode=ExecutionMode.DAG)

# Force stepper engine (works with any flow structure)
compiled = flow.compile(mode=ExecutionMode.STEPPER)

# Auto-detect (default) - uses stepper if needed, otherwise DAG
compiled = flow.compile(mode=ExecutionMode.AUTO)
```

#### ⚠️ Avoid Mixing Patterns

**Don't mix implicit and explicit construction in the same flow:**

```python
# ❌ BAD: Mixing patterns causes confusion
flow = Flow(...)
node_a = NodeWithInput(input_node=start)  # Implicit dependency
flow.add_nodes(start, node_a)
flow.add_edge("start", "node_b")  # Explicit edge
flow.set_entry_nodes("start")  # Explicit entry
# Result: Unclear execution order, potential conflicts
```

**Instead, choose one pattern consistently:**

```python
# ✅ GOOD: Pure explicit pattern
flow = Flow(...)
flow.add_nodes(start, node_a, node_b)
flow.set_entry_nodes("start")
flow.add_edge("start", "node_a")
flow.add_edge("node_a", "node_b")
```

If you need both patterns, use **sub-flows** (FlowNode) to compose them separately.

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

### Prompt Library

pydantic-flow includes a production-ready prompt templating system that can be used standalone or integrated with workflows:

```python
from pydantic import BaseModel
from pydantic_flow.prompt import (
    PromptTemplate,
    ChatPromptTemplate,
    TemplateFormat,
    ChatMessage,
    ChatRole,
    JsonModelParser
)

# Define typed models
class WeatherQuery(BaseModel):
    location: str
    unit: str = "celsius"

class WeatherResponse(BaseModel):
    temperature: float
    condition: str

# Create type-safe template with parser
template = PromptTemplate[WeatherQuery, WeatherResponse](
    template="What's the weather in {location} ({unit})?",
    format=TemplateFormat.F_STRING,
    input_model=WeatherQuery,
    output_parser=JsonModelParser(WeatherResponse)
)

# Render with validation
query = WeatherQuery(location="Paris")
prompt_text = template.render(query)

# Or render and parse in one step
result: WeatherResponse = await template.render_and_parse(query)

# Chat templates with multiple messages
chat_template = ChatPromptTemplate[WeatherQuery, str](
    messages=[
        ChatMessage(role=ChatRole.SYSTEM, content="You are a weather assistant."),
        ChatMessage(role=ChatRole.USER, content="Weather in {location}?")
    ],
    format=TemplateFormat.JINJA2,
    input_model=WeatherQuery
)

messages = chat_template.render_messages(query)
```

**Key Features:**
- Multiple template engines: f-string, Jinja2, Mustache
- Type-safe input/output with Pydantic models
- Built-in output parsers (JSON, delimited, custom)
- OpenTelemetry observability integration
- Chat message support with role-based formatting
- Variable validation and extraction

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
- External libraries can create `pydantic-flow-*` extensions
- Type-safe protocol-based interfaces
- Dependency injection with automatic wiring

## Installation

```bash
pip install pydantic-flow
```

### Development Installation

```bash
git clone https://github.com/joshSzep/pydantic-flow.git
cd pydantic-flow
uv sync
```

## Quick Start

```python
# Coming soon!
```

## Architecture

pydantic-flow follows core principles that ensure scalability and maintainability:

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
pydantic-flow/
├── src/pydantic_flow/   # Core framework
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
