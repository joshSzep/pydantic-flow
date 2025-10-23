# Loop Examples

This directory contains examples demonstrating loop support in pydantic-flow.

## Examples

### Self-Loop Counter (`self_loop_counter.py`)

A simple example showing a single node that loops back to itself until a condition is met.

```bash
python examples/loops/self_loop_counter.py
```

The counter increments on each iteration and terminates when reaching 5 using `Route.END`.

### Two-Node While-Loop (`two_node_while_loop.py`)

Demonstrates a plan-execute pattern where two nodes form a loop with conditional routing.

```bash
python examples/loops/two_node_while_loop.py
```

The flow alternates between planning and executing work until a maximum iteration count is reached.

## Key Concepts

- **Route.END**: Special sentinel value to terminate flow execution
- **Conditional Edges**: Dynamic routing based on state
- **RunConfig**: Control recursion limits and observability
- **Stepper Engine**: Automatically used for flows with loops or conditional edges
