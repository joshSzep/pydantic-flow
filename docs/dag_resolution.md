# DAG Resolution and Automatic Execution Ordering

This document explains how pydantic-flow automatically resolves dependencies and determines execution order for workflow nodes using directed acyclic graph (DAG) analysis and topological sorting.

## Overview

One of pydantic-flow's key features is that you never have to manually specify the execution order of nodes. Instead, you declaratively define relationships between nodes using typed references, and the framework automatically:

1. Discovers dependencies between nodes
2. Builds a dependency graph (DAG)
3. Validates that no cycles exist
4. Determines a valid execution order using topological sorting
5. Executes nodes in the correct sequence

## Reference-Based Wiring

### The Magic of `.output`

Instead of using magic strings or manual ordering, nodes are connected using **typed output references**:

```python
from pydantic import BaseModel
from pydantic_flow import Flow, PromptNode, ParserNode

class Query(BaseModel):
    location: str

class Info(BaseModel):
    data: str

# Create nodes
node1 = PromptNode[Query, str](
    prompt="Get info for {location}",
    name="prompt"
)

node2 = ParserNode[str, Info](
    parser_func=parse_data,
    input=node1.output,  # ← Type-safe reference!
    name="parser"
)

# Add in ANY order - framework figures out the dependencies
flow = Flow(input_type=Query, output_type=Results)
flow.add_nodes(node2, node1)  # Order doesn't matter!
```

### NodeOutput: The Connection Mechanism

When you access `node.output`, you get a `NodeOutput` object:

```python
@dataclass(frozen=True)
class NodeOutput[OutputT]:
    """Represents a typed output reference from a node."""
    
    node: BaseNode[Any, OutputT]
    
    @property
    def type_hint(self) -> type[OutputT]:
        """Get the output type hint for this node output."""
        return self.node._output_type
```

This object:
- **Carries type information**: The generic `OutputT` parameter provides IDE autocomplete
- **References the source node**: Points back to the node that produces the data
- **Enables dependency tracking**: The framework uses this to build the dependency graph

## How DAG Resolution Works

### Step 1: Dependency Discovery

Each node tracks its dependencies through the `dependencies` property:

```python
class NodeWithInput[InputT, OutputT](BaseNode[InputT, OutputT]):
    def __init__(self, input: NodeOutput[InputT] | None = None, ...):
        super().__init__(name)
        self.input = input
    
    @property
    def dependencies(self) -> list[BaseNode[Any, Any]]:
        """Get the list of nodes this node depends on."""
        if self.input is None:
            return []  # Takes input from flow
        return [self.input.node]  # Depends on this node
```

**Example:**
```python
node1 = PromptNode[Query, str](name="node1")
node2 = ParserNode[str, Info](input=node1.output, name="node2")

# node2.dependencies returns [node1]
# node1.dependencies returns [] (takes flow input)
```

### Step 2: Building the Dependency Graph

When you call `add_nodes()`, the framework builds a directed graph:

```python
def _calculate_execution_order(self) -> None:
    """Calculate execution order using topological sorting."""
    
    # Track how many dependencies each node has
    in_degree = dict.fromkeys(self.nodes, 0)
    
    # Track which nodes depend on each node
    adjacency = {node: [] for node in self.nodes}
    
    # Build the graph
    for node in self.nodes:
        for dep in getattr(node, "dependencies", []):
            if dep in self.nodes:
                adjacency[dep].append(node)  # dep → node edge
                in_degree[node] += 1  # node has one more incoming edge
```

**Example Graph:**

For this workflow:
```python
node1 = PromptNode[Query, str](name="node1")
node2 = ParserNode[str, Info](input=node1.output, name="node2")
node3 = ParserNode[str, Info](input=node1.output, name="node3")
node4 = ParserNode[Info, Result](input=node2.output, name="node4")
```

The graph looks like:
```
flow_input → node1 → node2 → node4
              ↓
           node3
```

Data structures:
```python
in_degree = {
    node1: 0,  # No dependencies
    node2: 1,  # Depends on node1
    node3: 1,  # Depends on node1
    node4: 1   # Depends on node2
}

adjacency = {
    node1: [node2, node3],  # node1 feeds into node2 and node3
    node2: [node4],         # node2 feeds into node4
    node3: [],              # node3 is a leaf
    node4: []               # node4 is a leaf
}
```

### Step 3: Topological Sorting (Kahn's Algorithm)

The framework uses **Kahn's algorithm** to determine a valid execution order:

```python
# Start with nodes that have no dependencies
queue = deque([node for node in self.nodes if in_degree[node] == 0])
execution_order = []

while queue:
    # Pick a node with all dependencies satisfied
    current = queue.popleft()
    execution_order.append(current)
    
    # Update dependency counts for dependent nodes
    for neighbor in adjacency[current]:
        in_degree[neighbor] -= 1
        if in_degree[neighbor] == 0:
            # All dependencies satisfied, ready to execute
            queue.append(neighbor)
```

**Step-by-step execution:**

1. **Initial state**: `queue = [node1]` (in_degree = 0)
2. **Process node1**: Add to order, update dependencies
   - `execution_order = [node1]`
   - `node2.in_degree = 0`, `node3.in_degree = 0`
   - `queue = [node2, node3]`
3. **Process node2**: Add to order, update dependencies
   - `execution_order = [node1, node2]`
   - `node4.in_degree = 0`
   - `queue = [node3, node4]`
4. **Process node3**: Add to order
   - `execution_order = [node1, node2, node3]`
   - `queue = [node4]`
5. **Process node4**: Add to order
   - `execution_order = [node1, node2, node3, node4]`
   - `queue = []` (done!)

**Result**: `[node1, node2, node3, node4]` or `[node1, node3, node2, node4]` (both valid!)

### Step 4: Cycle Detection

If there's a circular dependency, topological sorting cannot complete:

```python
if len(execution_order) != len(self.nodes):
    raise CyclicDependencyError("Cyclic dependency detected in the flow")
```

**Example of a cycle** (this would be caught):
```python
# This creates: A → B → C → A (invalid!)
nodeA.input = nodeC.output
nodeB.input = nodeA.output
nodeC.input = nodeB.output
```

When the algorithm runs:
- All nodes start with `in_degree >= 1`
- No nodes can be added to the initial queue
- `execution_order` remains empty
- Exception is raised

## Execution Phase

Once the execution order is determined, the `flow.run()` method executes nodes sequentially:

```python
async def run(self, inputs: InputT) -> OutputT:
    """Execute the flow with the given inputs."""
    self._results = {}
    
    for node in self._execution_order:
        # Determine where to get input
        node_input = getattr(node, "input", None)
        
        if node_input is not None:
            # Get result from dependency node
            input_node = node_input.node
            input_data = self._results[input_node.name]
        else:
            # Use flow's initial input
            input_data = inputs
        
        # Execute node with appropriate input
        result = await node.run(input_data)
        
        # Store result for dependent nodes
        self._results[node.name] = result
    
    # Construct output model from all results
    return self._output_type(**self._results)
```

**Key aspects:**

1. **Results Cache**: Each node's output is stored in `_results` by name
2. **Input Resolution**: Nodes either get data from dependencies or from flow input
3. **Sequential Execution**: Nodes run one at a time in the computed order
4. **Type Safety**: Input/output types are validated at runtime

## Benefits of This Approach

### 1. Type Safety

The compiler and IDE know the types at every step:

```python
node1: PromptNode[Query, str]  # Outputs str
node2: ParserNode[str, Info]   # Requires str input

node2 = ParserNode[str, Info](
    input=node1.output  # ✅ Type matches!
)

node2 = ParserNode[int, Info](
    input=node1.output  # ❌ Type error: NodeOutput[str] != int
)
```

### 2. Declarative Workflows

You describe **what** you want, not **how** to execute it:

```python
# Add nodes in any order
flow.add_nodes(final_node, middle_node, initial_node)

# Framework figures out: initial → middle → final
```

### 3. No Manual Ordering

No need to track execution order manually:

```python
# ❌ Manual (fragile, error-prone)
results = []
results.append(await node1.run(input))
results.append(await node2.run(results[0]))
results.append(await node3.run(results[0]))
results.append(await node4.run(results[1]))

# ✅ Automatic (safe, maintainable)
results = await flow.run(input)
```

### 4. Parallelization Ready

The DAG structure makes it easy to identify independent nodes:

```python
# In this graph:
#   node1 → node2
#   node1 → node3
#
# node2 and node3 could run in parallel!
```

Future versions could automatically parallelize independent branches.

### 5. Visualization and Debugging

The dependency graph can be inspected and visualized:

```python
# Get execution order for debugging
order = flow.get_execution_order()
print(f"Execution order: {order}")

# Validate flow structure
flow.validate()  # Raises CyclicDependencyError if invalid
```

## Example: Complex Workflow

Here's a complete example showing how DAG resolution handles a complex workflow:

```python
from pydantic import BaseModel
from pydantic_flow import Flow, PromptNode, ParserNode, ToolNode, IfNode

class Query(BaseModel):
    topic: str

class Research(BaseModel):
    facts: list[str]

class Analysis(BaseModel):
    insights: list[str]

class Summary(BaseModel):
    content: str

class Report(BaseModel):
    research: Research
    analysis: Analysis
    summary: Summary

# Create nodes
research_node = ToolNode[Query, Research](
    tool_func=gather_research,
    name="research"
)

analysis_node = ParserNode[Research, Analysis](
    parser_func=analyze_facts,
    input=research_node.output,
    name="analysis"
)

summary_node = PromptNode[Research, Summary](
    prompt="Summarize: {facts}",
    input=research_node.output,
    name="summary"
)

# Create flow and add nodes in random order!
flow = Flow(input_type=Query, output_type=Report)
flow.add_nodes(summary_node, analysis_node, research_node)

# Framework automatically determines:
# 1. research_node must run first (no dependencies)
# 2. analysis_node and summary_node can run after research_node
# 3. Execution order: [research, analysis, summary] or [research, summary, analysis]

# Execute
query = Query(topic="AI workflows")
report = await flow.run(query)

# Access results with full type safety
print(report.research.facts)
print(report.analysis.insights)
print(report.summary.content)
```

## Comparison with Other Frameworks

### LangChain
```python
# LangChain: Manual chaining with | operator
chain = prompt | llm | parser

# Or: Manual execution order
result1 = await node1.run(input)
result2 = await node2.run(result1)
```

### pydantic-flow
```python
# Declarative: Just define relationships
node2 = ParserNode(input=node1.output)
flow.add_nodes(node1, node2)

# Automatic execution order
result = await flow.run(input)
```

## Implementation Details

### Time Complexity

- **Graph Construction**: O(N + E) where N = nodes, E = edges
- **Topological Sort**: O(N + E) using Kahn's algorithm
- **Total**: O(N + E) for each call to `add_nodes()`

### Space Complexity

- **in_degree dictionary**: O(N)
- **adjacency dictionary**: O(N + E)
- **execution_order list**: O(N)
- **Total**: O(N + E)

### Edge Cases

1. **Empty Flow**: Valid, returns empty results
2. **Single Node**: Trivial DAG, executes immediately
3. **Linear Chain**: A → B → C → D (optimal ordering)
4. **Diamond Pattern**: A → B,C → D (multiple valid orderings)
5. **Multiple Roots**: A → C, B → C (both A and B can start)
6. **Fan-In Pattern**: Multiple nodes merge into one using `MergeNode`

## Multi-Input Nodes (Fan-In Pattern)

### Overview

In addition to single-input nodes, pydantic-flow supports **multi-input nodes** that enable fan-in patterns where multiple node outputs are combined into a single processing step.

### MergeNode Architecture

The `MergeNode` base class accepts a tuple of `NodeOutput` references:

```python
class MergeNode[*InputTs, OutputT](BaseNode[tuple[*InputTs], OutputT]):
    """Base class for nodes that merge multiple inputs."""
    
    def __init__(
        self,
        inputs: tuple[NodeOutput[Any], ...],
        name: str | None = None,
    ):
        super().__init__(name)
        self.inputs = inputs  # Tuple of NodeOutput references
    
    @property
    def dependencies(self) -> list[BaseNode[Any, Any]]:
        """Get all dependency nodes from multiple inputs."""
        return [node_output.node for node_output in self.inputs]
```

### Concrete Implementations

#### MergeToolNode

Combines multiple inputs and passes them to a tool function:

```python
from pydantic_flow import MergeToolNode

# Three nodes producing different data
node_a = ToolNode[Input, DataA](tool_func=get_a, name="A")
node_b = ToolNode[Input, DataB](tool_func=get_b, name="B")
node_c = ToolNode[Input, DataC](tool_func=get_c, name="C")

# Merge function accepts all input types
def combine(a: DataA, b: DataB, c: DataC) -> Result:
    return Result(merged=f"{a}+{b}+{c}")

# Create merge node with tuple of inputs
merge_node = MergeToolNode[DataA, DataB, DataC, Result](
    inputs=(node_a.output, node_b.output, node_c.output),
    tool_func=combine,
    name="merge"
)
```

#### MergeParserNode

Similar to `MergeToolNode` but for parser functions:

```python
from pydantic_flow import MergeParserNode

def parse_both(text_a: str, text_b: str) -> Parsed:
    return Parsed(combined=f"{text_a} and {text_b}")

parser = MergeParserNode[str, str, Parsed](
    inputs=(node_a.output, node_b.output),
    parser_func=parse_both,
    name="parser"
)
```

### Fan-In Dependency Resolution

When a `MergeNode` is added to a flow, the DAG resolution algorithm:

1. **Detects multiple inputs**: Checks for `inputs` attribute containing tuple
2. **Extracts all dependencies**: Adds all input nodes to dependency graph
3. **Ensures order**: All input nodes must execute before the merge node
4. **Gathers results**: Collects outputs from all dependencies as a tuple

### Example: Complex Fan-Out/Fan-In Pattern

```python
# Pattern: A → B,C → D and A → C, then D,C → E
class ProcessedA(BaseModel):
    data: str

class ProcessedB(BaseModel):
    data_b: str

class ProcessedC(BaseModel):
    data_c: str

class ProcessedD(BaseModel):
    data_d: str

class ProcessedE(BaseModel):
    final: str

# A receives initial input
node_a = ToolNode[Input, ProcessedA](tool_func=process_a, name="A")

# A fans out to B and C
node_b = ToolNode[ProcessedA, ProcessedB](
    tool_func=process_b,
    input=node_a.output,
    name="B"
)

node_c = ToolNode[ProcessedA, ProcessedC](
    tool_func=process_c,
    input=node_a.output,
    name="C"
)

# B goes to D
node_d = ToolNode[ProcessedB, ProcessedD](
    tool_func=process_d,
    input=node_b.output,
    name="D"
)

# D and C fan in to E (merge node)
def merge_d_and_c(d: ProcessedD, c: ProcessedC) -> ProcessedE:
    return ProcessedE(final=f"E({d.data_d},{c.data_c})")

node_e = MergeToolNode[ProcessedD, ProcessedC, ProcessedE](
    inputs=(node_d.output, node_c.output),
    tool_func=merge_d_and_c,
    name="E"
)

# Add to flow
flow.add_nodes(node_a, node_b, node_c, node_d, node_e)

# Execution order ensures: A → B,C → D → E
# Both D and C must complete before E runs
```

### Flow Execution with Multi-Input Nodes

The `Flow.run()` method detects and handles multi-input nodes:

```python
for node in self._execution_order:
    # Check if multi-input node
    node_inputs = getattr(node, "inputs", None)
    if node_inputs is not None and isinstance(node_inputs, tuple):
        # Gather all dependency results as tuple
        input_data = tuple(
            self._results[dep.node.name] for dep in node_inputs
        )
    else:
        # Single-input or no-input node (existing behavior)
        # ... handle as before
    
    # Execute with appropriate input
    result = await node.run(input_data)
    self._results[node.name] = result
```

### Type Safety with Multi-Input Nodes

The `MergeNode` uses `TypeVarTuple` (PEP 646) for full type safety:

```python
# IDE knows each input type
merge_node = MergeToolNode[DataA, DataB, DataC, Result](
    inputs=(node_a.output, node_b.output, node_c.output),
    #       ↑ NodeOutput[DataA]
    #                    ↑ NodeOutput[DataB]
    #                                 ↑ NodeOutput[DataC]
    tool_func=combine
)

# Function signature must match
def combine(a: DataA, b: DataB, c: DataC) -> Result:
    #           ↑ Matches input tuple types
    ...
```

### Dependency Graph with Fan-In

For the complex example above, the dependency graph is:

```
Input → A → B → D ↘
         ↓         E
         C ------↗

in_degree:
  A: 0 (no dependencies)
  B: 1 (depends on A)
  C: 1 (depends on A)
  D: 1 (depends on B)
  E: 2 (depends on D and C)  ← Multi-input!

adjacency:
  A: [B, C]
  B: [D]
  C: [E]
  D: [E]
  E: []
```

Topological sort produces: `[A, B, C, D, E]` or `[A, C, B, D, E]`

Note that `E` only executes after **both** `D` and `C` complete, since `in_degree[E] = 2`.

### Benefits of Multi-Input Nodes

1. **Complex Workflows**: Model real-world patterns where multiple data sources combine
2. **Type-Safe Merging**: Each input maintains its own type with full IDE support
3. **Declarative**: Express fan-in patterns naturally without manual coordination
4. **Reusable**: Merge logic encapsulated in pure functions
5. **Testable**: Merge nodes tested independently with mocked inputs

## Future Enhancements

### Automatic Parallelization

The DAG structure enables automatic parallel execution:

```python
# Future API
flow = Flow(parallel=True)
flow.add_nodes(...)

# Automatically runs independent nodes in parallel
results = await flow.run(input)
```

### Conditional DAG Modification

Dynamic graph modification based on runtime conditions:

```python
# Future: Conditional node execution
flow.add_conditional_branch(
    predicate=lambda x: x.confidence > 0.8,
    nodes=[high_confidence_path],
    else_nodes=[verification_path]
)
```

### Visualization

Generate visual representations of the execution graph:

```python
# Future: Export to visualization formats
flow.to_mermaid()  # Mermaid diagram
flow.to_graphviz()  # Graphviz DOT format
flow.visualize()  # Interactive visualization
```

## Conclusion

pydantic-flow's automatic DAG resolution eliminates an entire class of bugs and cognitive overhead by:

1. Using type-safe references instead of magic strings
2. Automatically determining execution order via topological sorting
3. Detecting circular dependencies at graph construction time
4. Enabling declarative workflow definition
5. Providing a foundation for future optimizations like parallelization

This approach combines the best of dataflow programming with Python's type system to create workflows that are both safe and maintainable.
