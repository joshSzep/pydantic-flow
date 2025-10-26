# Human-in-the-Loop (HITL) Examples

This directory contains examples demonstrating pydantic-flow's comprehensive Human-in-the-Loop functionality.

## Examples

### `hitl_simple.py` - Basic Interrupt Handlers

Demonstrates:
- Registering interrupt handlers at node and flow level
- Conditional interruption based on progress events
- Interrupt priority system
- Checkpoint creation and inspection

**Run:**
```bash
python examples/hitl_simple.py
```

**Key Concepts:**
- `InterruptDecision.proceed()` - Continue execution
- `InterruptDecision.interrupt(reason, metadata)` - Request interruption
- `InterruptionRequested` exception with checkpoint
- Flow-level vs node-level handlers

### `hitl_complex.py` - Multi-Stage Approval Workflow

Demonstrates:
- Multiple interrupt handlers with different priorities
- Handler execution order (CRITICAL → HIGH → NORMAL → LOW)
- Security checks, risk analysis, and compliance validation
- Audit logging with lowest priority
- Complex metadata tracking

**Run:**
```bash
python examples/hitl_complex.py
```

**Key Concepts:**
- `HandlerPriority` enum (CRITICAL=0, HIGH=26, NORMAL=51, LOW=76)
- Priority-based handler orchestration
- Checkpoint metadata for workflow state
- Handler execution order verification

## Core Concepts

### Interrupt Handlers

Handlers are async functions that receive `ProgressItem` events and return `InterruptDecision`:

```python
from pydantic_flow.streaming.events import ProgressItem, InterruptDecision

async def my_handler(item: ProgressItem) -> InterruptDecision:
    # Check conditions
    if should_interrupt(item):
        return InterruptDecision.interrupt("Reason", metadata={"key": "value"})
    return InterruptDecision.proceed()
```

### Registration

**Node-level:**
```python
node.register_interrupt_handler(handler, priority=50)
```

**Flow-level:**
```python
flow.register_interrupt_handler(handler, priority=50)
```

### Priority System

Handlers execute in priority order (lowest number first):

- **CRITICAL (0)**: Security checks, must always run
- **HIGH (26)**: Important business logic checks
- **NORMAL (51)**: Standard validation
- **LOW (76)**: Logging, auditing, nice-to-have checks

### Checkpoints

When interruption occurs, a `FlowCheckpoint` is created containing:

- `flow_id`: Flow identifier
- `run_id`: Execution identifier
- `interrupted_node_id`: Where interruption occurred
- `node_states`: Completed node outputs
- `edge_history`: Edges traversed
- `metadata`: Custom data from interrupt decision

## Progress Events

Interrupt handlers can check different progress event types:

- `StreamStart`: Execution begins
- `TokenChunk`: LLM token generated
- `ToolCall`: Tool invocation
- `ToolResult`: Tool completed
- `PartialFields`: Incremental structured output
- `StreamEnd`: Execution completes
- `NonFatalError`: Recoverable error
- `Heartbeat`: Liveness signal

Example:
```python
from pydantic_flow.streaming.events import StreamEnd, TokenChunk

async def check_completion(item: ProgressItem) -> InterruptDecision:
    if isinstance(item, StreamEnd):
        return InterruptDecision.interrupt("Final review required")
    
    if isinstance(item, TokenChunk) and "sensitive" in item.text:
        return InterruptDecision.interrupt("Sensitive content detected")
    
    return InterruptDecision.proceed()
```

## Best Practices

### 1. Use Appropriate Priorities

```python
# Critical security - always runs first
flow.register_interrupt_handler(security_check, priority=HandlerPriority.CRITICAL)

# Business logic
flow.register_interrupt_handler(validation, priority=HandlerPriority.NORMAL)

# Audit logging - runs last
flow.register_interrupt_handler(audit_log, priority=HandlerPriority.LOW)
```

### 2. Include Meaningful Metadata

```python
return InterruptDecision.interrupt(
    "Compliance review required",
    metadata={
        "policy_id": "POL-2024-001",
        "risk_level": "medium",
        "reviewer_role": "compliance_officer"
    }
)
```

### 3. Make Handlers Idempotent

Handlers may execute multiple times during retries:

```python
async def idempotent_handler(item: ProgressItem) -> InterruptDecision:
    # Use deterministic logic, avoid side effects
    if item.node_id in reviewed_nodes:
        return InterruptDecision.proceed()
    return InterruptDecision.interrupt("Needs review")
```

### 4. Clear Handlers When Done

```python
# Temporary handler for specific execution
node.register_interrupt_handler(temp_handler, priority=50)
try:
    result = await flow.run(input_data)
finally:
    node.clear_interrupt_handlers()
```

## Complete Documentation

For comprehensive HITL documentation, see [docs/hitl.md](../docs/hitl.md).

## API Reference

- **`InterruptDecision`**: `.proceed()` and `.interrupt(reason, metadata)`
- **`FlowCheckpoint`**: Serializable checkpoint for resumption
- **`InterruptionRequested`**: Exception raised on interrupt
- **`HandlerPriority`**: Priority constants (CRITICAL, HIGH, NORMAL, LOW)
- **`ProgressItem`**: Base class for all progress events
- **`node.register_interrupt_handler(callback, priority, metadata)`**
- **`flow.register_interrupt_handler(callback, priority, metadata)`**
- **`node.clear_interrupt_handlers()`**
- **`flow.clear_interrupt_handlers()`**
