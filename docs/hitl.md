# Human-in-the-Loop (HITL)

Pydantic-flow provides comprehensive Human-in-the-Loop functionality, allowing workflows to pause execution, request human intervention, and resume with human-provided input. This enables review workflows, approval processes, and interactive decision-making.

## Unified Checkpoint System

HITL interruptions use the unified checkpoint system (`StateSnapshot`) providing:

- **Unified architecture**: HITL interrupts use the same checkpoint system as debugging/time-travel
- **Conversation preservation**: Full conversation history maintained at interrupt points
- **Query tools**: CLI and API methods to list and inspect interrupted workflows
- **Time-travel debugging**: Debug interrupted workflows using the same tools as regular execution
- **Persistent snapshots**: All interrupts automatically create `reason=HITL_INTERRUPT` snapshots

**CLI Tools:**
```bash
# List all runs awaiting human decisions
pydantic-flow debug list-interrupts checkpoints.db

# Show detailed context at interrupt point (state, conversation, metadata)
pydantic-flow debug show-interrupt <run_id> --db checkpoints.db

# Resume with human decision
pydantic-flow debug resume-with-decision <run_id> --decision '{"approved": true}'
```

---

## Core Concepts

### Three-Layer Interruption Model

HITL interruptions can occur at three levels:

1. **Event Level**: Check every progress item (tokens, tool calls, etc.) via `interrupt_callback` field
2. **Node Level**: Register handlers on specific nodes to check their outputs
3. **Flow Level**: Register handlers that apply across all nodes in a flow

Handlers execute in priority order (lowest number first). If any handler requests interruption, execution stops immediately and raises `InterruptionRequested` with a checkpoint.

### Interruption Decision

All interrupt handlers return an `InterruptDecision`:

```python
from pydantic_flow.streaming.events import InterruptDecision

# Allow execution to continue
InterruptDecision.proceed()

# Request interruption with a reason
InterruptDecision.interrupt("Approval required", metadata={"priority": "high"})
```

### Checkpoints

When interruption occurs, the framework creates a `FlowCheckpoint` containing:

- `flow_id`: Unique identifier for the flow
- `run_id`: Execution run identifier
- `interrupted_node_id`: Which node triggered interruption
- `interrupt_reason`: Why interruption occurred
- `node_states`: Dict of completed node outputs
- `edge_history`: List of edges traversed
- `interrupt_metadata`: Optional custom data

Checkpoints enable resuming execution from the interruption point.

## HumanNode

`HumanNode` is a specialized node that always interrupts execution to request human input:

```python
from pydantic_flow.nodes.human import HumanNode, HumanResponse
from pydantic import BaseModel

class ReviewInput(BaseModel):
    content: str

# Simple text input
human_node = HumanNode[ReviewInput, HumanResponse](
    prompt="Please review this content"
)

# Dynamic prompts from input
human_node = HumanNode[ReviewInput, HumanResponse](
    prompt=lambda input_data: f"Review: {input_data.content}"
)

# With options
human_node = HumanNode[ReviewInput, HumanResponse](
    prompt="Select an option",
    options=["Approve", "Reject", "Modify"]
)
```

### Response Parsing

By default, `HumanNode` returns `HumanResponse`. To transform the response:

```python
from pydantic_flow.nodes.human import HumanNode, HumanResponse

class ApprovalResult(BaseModel):
    approved: bool
    comments: str

def parse_approval(response: HumanResponse) -> ApprovalResult:
    return ApprovalResult(
        approved=response.approved,
        comments=response.value
    )

human_node = HumanNode[ReviewInput, ApprovalResult](
    prompt="Review this item",
    response_parser=parse_approval
)
```

## ApprovalNode

`ApprovalNode` is a specialized `HumanNode` for yes/no approval workflows:

```python
from pydantic_flow.nodes.human import ApprovalNode

approval = ApprovalNode[ReviewInput](
    prompt="Approve this change?"
)

# Dynamic prompts
approval = ApprovalNode[ReviewInput](
    prompt=lambda input_data: f"Approve changes to {input_data.content}?"
)
```

`ApprovalNode` automatically returns `HumanResponse` with `approved` field.

## Interrupt Handlers

### Node-Level Handlers

Register handlers on specific nodes:

```python
from pydantic_flow.streaming.events import HumanInputRequest, InterruptDecision
from pydantic_flow.nodes.prompt import PromptNode
from pydantic_flow.core.errors import HandlerPriority

prompt_node = PromptNode[Input, str](prompt="Generate response")

async def review_output(request: HumanInputRequest) -> InterruptDecision:
    # Check some condition
    if should_review(request):
        return InterruptDecision.interrupt("Manual review required")
    return InterruptDecision.proceed()

# Register with priority
prompt_node.register_interrupt_handler(
    review_output,
    priority=HandlerPriority.NORMAL,  # 0-100, lower executes first
    metadata={"handler_name": "output_review"}
)
```

### Flow-Level Handlers

Register handlers that apply to all nodes:

```python
from pydantic_flow.flow.flow import Flow

flow = Flow(input_type=Input, output_type=Output)

async def global_review(request: HumanInputRequest) -> InterruptDecision:
    # Apply global policy
    if request.metadata.get("sensitive"):
        return InterruptDecision.interrupt("Sensitive content review")
    return InterruptDecision.proceed()

flow.register_interrupt_handler(global_review, priority=50)
```

### Handler Priority

Priority determines execution order (lower number = earlier execution):

```python
from pydantic_flow.core.errors import HandlerPriority

# HandlerPriority enum values:
HandlerPriority.CRITICAL = 0   # Always executes, highest priority
HandlerPriority.HIGH = 25       # Important checks
HandlerPriority.NORMAL = 50     # Default priority
HandlerPriority.LOW = 75        # Optional checks
HandlerPriority.AUDIT = 100     # Logging/auditing
```

Handlers execute in priority order. If any handler requests interruption, execution stops immediately.

### Clearing Handlers

Remove all registered handlers:

```python
# Node-level
prompt_node.clear_interrupt_handlers()

# Flow-level
flow.clear_interrupt_handlers()
```

## Complete Workflow Example

### Basic Approval Flow

```python
from pydantic import BaseModel
from pydantic_flow.flow.flow import Flow
from pydantic_flow.nodes.prompt import PromptNode
from pydantic_flow.nodes.human import ApprovalNode, HumanResponse
from pydantic_flow.core.errors import InterruptionRequested

class ContentInput(BaseModel):
    text: str

class ProcessedOutput(BaseModel):
    result: str

# Create nodes
processor = PromptNode[ContentInput, str](
    prompt="Process this: {input.text}"
)

approval = ApprovalNode[str](
    prompt=lambda input_data: f"Approve output: {input_data}?"
)

# Build flow
flow = Flow(input_type=ContentInput, output_type=HumanResponse)
flow.add_node(processor)
flow.add_node(approval, dependencies=[processor])

# First execution - will interrupt at approval node
try:
    result = await extract_result_from_stream(flow.astream(ContentInput(text="test"))
except InterruptionRequested as exc:
    checkpoint = exc.checkpoint
    print(f"Flow interrupted: {checkpoint.interrupt_reason}")
    
    # Present to human, collect decision
    human_decision = get_human_approval()  # Your UI logic
    
    # Resume with response
    response = HumanResponse(
        value="approved" if human_decision else "rejected",
        approved=human_decision
    )
    
    # Resume execution
    final_result = await flow.resume(checkpoint, inputs=ContentInput(text="test"))
    print(f"Approved: {final_result.approved}")
```

### Conditional Interruption

Only interrupt under specific conditions:

```python
from pydantic_flow.streaming.events import InterruptDecision, HumanInputRequest

async def conditional_review(request: HumanInputRequest) -> InterruptDecision:
    """Only interrupt for high-value items."""
    if request.metadata.get("value", 0) > 1000:
        return InterruptDecision.interrupt(
            "High-value transaction requires approval",
            metadata={"escalation_level": "manager"}
        )
    return InterruptDecision.proceed()

prompt_node.register_interrupt_handler(conditional_review, priority=10)
```

### Multi-Stage Approval

Multiple approval points in a workflow:

```python
# Stage 1: Content review
content_review = HumanNode[ContentInput, HumanResponse](
    prompt="Review content quality"
)

# Stage 2: Compliance check
compliance_review = ApprovalNode[str](
    prompt="Approve for compliance?"
)

# Stage 3: Final approval
final_approval = ApprovalNode[str](
    prompt="Final approval?"
)

flow = Flow(input_type=ContentInput, output_type=HumanResponse)
flow.add_node(processor)
flow.add_node(content_review, dependencies=[processor])
flow.add_node(compliance_review, dependencies=[content_review])
flow.add_node(final_approval, dependencies=[compliance_review])

# Execute through multiple interruption points
current_input = ContentInput(text="content")
checkpoint = None

while True:
    try:
        if checkpoint:
            result = await flow.resume(checkpoint, inputs=current_input)
        else:
            result = await extract_result_from_stream(flow.astream(current_input)
        
        # Completed successfully
        print(f"All approvals complete: {result}")
        break
        
    except InterruptionRequested as exc:
        checkpoint = exc.checkpoint
        print(f"Approval needed: {checkpoint.interrupt_reason}")
        
        # Get human input
        response = get_human_response(checkpoint)
        
        if not response.approved:
            print("Workflow rejected")
            break
```

## Event-Level Interruption

Check specific progress items via the `interrupt_callback` field:

```python
from pydantic_flow.streaming.events import StreamStart, InterruptDecision

async def check_stream_start(item: StreamStart) -> InterruptDecision:
    # Custom logic for stream start events
    if item.metadata.get("requires_review"):
        return InterruptDecision.interrupt("Stream requires review")
    return InterruptDecision.proceed()

# Set on progress item
item = StreamStart(run_id="123", node_id="node1")
item.interrupt_callback = check_stream_start
```

## Best Practices

### 1. Use Appropriate Priority Levels

```python
# Critical security checks
security_handler.register(..., priority=HandlerPriority.CRITICAL)

# Normal business logic
business_handler.register(..., priority=HandlerPriority.NORMAL)

# Optional logging
audit_handler.register(..., priority=HandlerPriority.AUDIT)
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

### 3. Checkpoint Persistence

Save checkpoints for resilience:

```python
try:
    result = await extract_result_from_stream(flow.astream(input_data)
except InterruptionRequested as exc:
    # Persist checkpoint
    save_checkpoint(exc.checkpoint)
    
    # Later, reload and resume
    checkpoint = load_checkpoint(checkpoint_id)
    result = await flow.resume(checkpoint, inputs=input_data)
```

### 4. Clear Handlers When Done

```python
# Temporary handler for specific execution
node.register_interrupt_handler(temp_handler, priority=50)
try:
    result = await extract_result_from_stream(flow.astream(input_data)
finally:
    node.clear_interrupt_handlers()
```

### 5. Handler Idempotency

Handlers may execute multiple times during retries:

```python
async def idempotent_handler(request: HumanInputRequest) -> InterruptDecision:
    # Use deterministic logic, avoid side effects
    if request.node_id in reviewed_nodes:
        return InterruptDecision.proceed()
    return InterruptDecision.interrupt("Needs review")
```

## API Reference

### Core Classes

- **`InterruptDecision`**: Return value from interrupt handlers
  - `.proceed()`: Continue execution
  - `.interrupt(reason, metadata)`: Request interruption

- **`FlowCheckpoint`**: Serializable checkpoint state
  - `.flow_id`: Flow identifier
  - `.run_id`: Execution identifier
  - `.node_states`: Completed outputs
  - `.interrupt_reason`: Why stopped
  - `.interrupt_metadata`: Custom data

- **`InterruptionRequested`**: Exception raised on interrupt
  - `.checkpoint`: FlowCheckpoint for resumption
  - `.reason`: Interruption reason

- **`HumanInputRequest`**: Metadata passed to handlers
  - `.node_id`: Which node is interrupting
  - `.run_id`: Execution identifier
  - `.metadata`: Custom data

- **`HumanResponse`**: Standard human input format
  - `.value`: Text response
  - `.approved`: Boolean flag
  - `.metadata`: Optional custom data

### Nodes

- **`HumanNode[InputModel, OutputModel]`**
  - `prompt`: Static string or `Callable[[InputModel], str]`
  - `options`: Optional list of choices
  - `response_parser`: Optional `Callable[[HumanResponse], OutputModel]`

- **`ApprovalNode[InputModel]`**
  - Specialized `HumanNode` for yes/no decisions
  - Returns `HumanResponse` with `approved` field

### Handler Registration

- **`node.register_interrupt_handler(callback, priority, metadata)`**
- **`flow.register_interrupt_handler(callback, priority, metadata)`**
- **`node.clear_interrupt_handlers()`**
- **`flow.clear_interrupt_handlers()`**

### Flow Methods

- **`await extract_result_from_stream(flow.astream(inputs)`**: Execute flow, may raise `InterruptionRequested`
- **`await flow.resume(checkpoint, inputs)`**: Resume from checkpoint

## Type Safety

All HITL components are fully type-safe:

```python
# Type-checked node creation
human_node: HumanNode[InputModel, OutputModel] = HumanNode(
    prompt="Review",
    response_parser=parse_func
)

# Type-checked handler
async def handler(request: HumanInputRequest) -> InterruptDecision:
    return InterruptDecision.proceed()

# Type-checked checkpoint
checkpoint: FlowCheckpoint = exc.checkpoint
```

IDE autocomplete and type checkers provide full support for all HITL APIs.
