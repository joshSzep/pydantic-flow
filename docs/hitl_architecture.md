# Human-in-the-Loop (HITL) Architecture Design

**Author**: Architecture Design  
**Date**: October 26, 2025  
**Status**: Proposal for Review

---

## Executive Summary

This document proposes a comprehensive Human-in-the-Loop (HITL) architecture for pydantic-flow that integrates deeply with the existing streaming-native event system. The design maintains the framework's core principles while adding powerful interruption capabilities at every level: events, nodes, and flows.

### Core Design Principles

1. **Streaming-Native Integration**: HITL builds on top of the existing `ProgressItem` event stream
2. **Zero-Wrapper Philosophy**: Break backwards compatibility rather than add complexity through wrappers
3. **Interruption at Every Level**: Events, nodes, and flows all support interruption
4. **Type-Safe Callbacks**: Leverage Python 3.14+ type system for HITL handlers
5. **Async-First**: All interruption points are async-aware

---

## Architecture Overview

### Three-Layer Interruption Model

```
┌─────────────────────────────────────────────────────┐
│                    Flow Level                        │
│  FlowInterruptHandler - observes node transitions   │
│  Can interrupt before/after node execution           │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────┐
│                    Node Level                        │
│  NodeInterruptHandler - observes node lifecycle     │
│  Can interrupt at node start/end                     │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────┐
│                   Event Level                        │
│  EventInterruptCallback - on every ProgressItem     │
│  Can interrupt streaming operations mid-flight      │
└─────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. Event-Level Interruption

Every `ProgressItem` gains an optional interrupt callback that can halt streaming.

#### Modified ProgressItem Base Class

```python
# src/pydantic_flow/streaming/events.py

from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

# New type alias for interrupt callbacks
InterruptCallback = Callable[['ProgressItem'], Awaitable['InterruptDecision']]


class InterruptDecision(BaseModel):
    """Decision from an interrupt handler.
    
    Attributes:
        should_interrupt: Whether to halt execution
        reason: Human-readable explanation
        replacement_value: Optional value to replace current result
        metadata: Flexible dict for additional context (user-extensible)
    """
    
    should_interrupt: bool
    reason: str | None = None
    replacement_value: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def proceed(cls) -> 'InterruptDecision':
        """Create a decision to proceed without interruption."""
        return cls(should_interrupt=False)
    
    @classmethod
    def interrupt(
        cls,
        reason: str,
        *,
        replacement_value: Any = None,
        **metadata: Any,
    ) -> 'InterruptDecision':
        """Create a decision to interrupt with metadata.
        
        Args:
            reason: Why the interrupt occurred
            replacement_value: Optional value to use instead of current
            **metadata: Additional user-defined metadata
        """
        return cls(
            should_interrupt=True,
            reason=reason,
            replacement_value=replacement_value,
            metadata=metadata,
        )


class ProgressItem(BaseModel):
    """Base class for all streaming progress events.
    
    BREAKING CHANGE: Added interrupt_callback for HITL support.
    """
    
    model_config = {"frozen": False, "arbitrary_types_allowed": True}
    
    type: ProgressType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str = ""
    node_id: str = ""
    
    # NEW: Optional interrupt callback (supports both creation and post-creation)
    interrupt_callback: InterruptCallback | None = Field(
        default=None,
        exclude=True,  # Don't serialize the callback
    )
    
    def set_interrupt_callback(self, callback: InterruptCallback) -> 'ProgressItem':
        """Set the interrupt callback (for post-creation attachment).
        
        Returns self for chaining.
        """
        self.interrupt_callback = callback
        return self
    
    async def check_interrupt(self) -> InterruptDecision:
        """Check if this event should cause an interruption.
        
        Returns:
            InterruptDecision with should_interrupt=False if no callback set.
        """
        if self.interrupt_callback is None:
            return InterruptDecision.proceed()
        
        return await self.interrupt_callback(self)
```

**Rationale**: 
- Primary pattern: Set callback in constructor when known
- Fallback pattern: `set_interrupt_callback()` for dynamic attachment
- Convenience methods on `InterruptDecision` for cleaner code

### 2. Interrupt Exception

A special exception to propagate interruptions with checkpoint support.

```python
# src/pydantic_flow/core/errors.py

class FlowCheckpoint(BaseModel):
    """Serializable checkpoint for flow resumption.
    
    Attributes:
        flow_id: Unique identifier for the flow
        execution_mode: DAG or STEPPER mode
        results: Dict of completed node results
        pending_nodes: List of nodes not yet executed
        current_node: Node that was interrupted (if any)
        inputs: Original flow inputs
        metadata: Additional checkpoint context
        timestamp: When checkpoint was created
    """
    
    flow_id: str
    execution_mode: str  # "dag" or "stepper"
    results: dict[str, Any]
    pending_nodes: list[str]
    current_node: str | None = None
    inputs: dict[str, Any]  # Serialized input data
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    
    # Stepper-specific state
    iteration: int | None = None
    frontier: list[str] | None = None


class InterruptionRequested(FlowError):
    """Raised when HITL interruption is requested.
    
    This exception carries the interruption decision and checkpoint state,
    allowing flows to be resumed after human intervention.
    
    Attributes:
        decision: The interrupt decision
        interrupted_at: The progress item where interruption occurred
        checkpoint: Serializable state for resumption
    """
    
    def __init__(
        self,
        decision: InterruptDecision,
        interrupted_at: ProgressItem,
        checkpoint: FlowCheckpoint | None = None,
    ) -> None:
        self.decision = decision
        self.interrupted_at = interrupted_at
        self.checkpoint = checkpoint
        
        reason = decision.reason or "No reason provided"
        location = f"{interrupted_at.node_id}/{interrupted_at.type}"
        super().__init__(
            f"Execution interrupted: {reason} (at {location})"
        )
    
    def can_resume(self) -> bool:
        """Check if this interruption includes resumption state."""
        return self.checkpoint is not None
    
    def get_resume_metadata(self) -> dict[str, Any]:
        """Get metadata about where to resume."""
        if self.checkpoint is None:
            return {}
        
        return {
            "flow_id": self.checkpoint.flow_id,
            "current_node": self.checkpoint.current_node,
            "completed_nodes": list(self.checkpoint.results.keys()),
            "pending_nodes": self.checkpoint.pending_nodes,
            "iteration": self.checkpoint.iteration,
        }
```

**Rationale**: Checkpoints enable full flow resumption, critical for long-running workflows where human approval may take hours/days.

### 3. HITL Node Types

#### HumanApprovalNode

A node that always stops execution and waits for human approval.

```python
# src/pydantic_flow/nodes/human.py (NEW FILE)

from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from pydantic_flow.nodes.base import NodeWithInput
from pydantic_flow.streaming.events import Heartbeat
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart


class HumanDecision[T](BaseModel):
    """Decision from human reviewer."""
    
    approved: bool
    modified_value: T | None = None  # Human can modify the value
    feedback: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# Type alias for human decision functions
HumanDecisionFunc[T] = Callable[[T, dict[str, Any]], Awaitable[HumanDecision[T]]]


class HumanApprovalNode[InputT, OutputT](NodeWithInput[InputT, OutputT]):
    """Node that requires human approval before proceeding.
    
    This node stops execution and calls a user-provided async function
    to get human input. The function can be interactive (CLI, web UI, etc.)
    or delegate to a queue/webhook system.
    
    Example:
        async def approve_weather(data: WeatherInfo, context: dict) -> HumanDecision:
            print(f"Approve this weather data? {data}")
            response = await get_user_input()  # Your UI logic
            return HumanDecision(
                approved=response == "yes",
                feedback="Looks good!"
            )
        
        approval_node = HumanApprovalNode[WeatherInfo, WeatherInfo](
            decision_func=approve_weather,
            input=weather_node.output,
            name="approve_weather"
        )
    """
    
    def __init__(
        self,
        decision_func: HumanDecisionFunc[InputT],
        *,
        input: Any = None,
        timeout_seconds: float | None = None,
        heartbeat_interval: float = 5.0,
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize a HumanApprovalNode.
        
        Args:
            decision_func: Async function that gets human decision
            input: Optional input from another node's output
            timeout_seconds: Optional timeout for human response
            heartbeat_interval: Seconds between heartbeat events while waiting
            name: Optional unique identifier for this node
            run_id: Optional run identifier for tracking execution
        """
        super().__init__(input, name, run_id)
        self.decision_func = decision_func
        self.timeout_seconds = timeout_seconds
        self.heartbeat_interval = heartbeat_interval
    
    async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
        """Stream progress while waiting for human approval.
        
        Yields heartbeats while waiting, then proceeds based on decision.
        """
        actual_run_id = self.run_id or ""
        
        yield StreamStart(
            run_id=actual_run_id,
            node_id=self.name,
            input_preview=self._preview_input(input_data),
        )
        
        # Collect context for the decision
        context = {
            "node_id": self.name,
            "run_id": actual_run_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        
        # Create decision task and heartbeat generator
        decision_task = asyncio.create_task(
            self._get_decision_with_timeout(input_data, context)
        )
        heartbeat_gen = self._emit_heartbeats(actual_run_id)
        
        decision = None
        try:
            # Yield heartbeats while waiting for decision
            while not decision_task.done():
                try:
                    heartbeat = await asyncio.wait_for(
                        heartbeat_gen.__anext__(),
                        timeout=0.1,  # Check decision status frequently
                    )
                    yield heartbeat
                except asyncio.TimeoutError:
                    continue
            
            # Get the decision result
            decision = await decision_task
            
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Human approval timeout after {self.timeout_seconds}s"
            )
        except Exception:
            raise
        finally:
            await heartbeat_gen.aclose()
        
        # Process decision
        if decision.approved:
            # Use modified value if provided, otherwise pass through
            output = decision.modified_value if decision.modified_value is not None else input_data
            
            yield StreamEnd(
                run_id=actual_run_id,
                node_id=self.name,
                result_preview=self._preview_input(output),  # type: ignore
            )
        else:
            # Rejection - raise interruption with checkpoint
            from pydantic_flow.core.errors import InterruptionRequested
            
            raise InterruptionRequested(
                decision=InterruptDecision(
                    should_interrupt=True,
                    reason=decision.feedback or "Human rejected",
                    metadata=decision.metadata,
                ),
                interrupted_at=StreamEnd(
                    run_id=actual_run_id,
                    node_id=self.name,
                ),
                # Checkpoint creation handled by flow
            )
    
    async def _get_decision_with_timeout(
        self,
        input_data: InputT,
        context: dict[str, Any],
    ) -> HumanDecision[InputT]:
        """Get human decision with optional timeout."""
        if self.timeout_seconds:
            return await asyncio.wait_for(
                self.decision_func(input_data, context),
                timeout=self.timeout_seconds,
            )
        return await self.decision_func(input_data, context)
    
    async def _emit_heartbeats(self, run_id: str) -> AsyncIterator[ProgressItem]:
        """Async generator that yields heartbeat events while waiting.
        
        Yields:
            Heartbeat progress items with approval status metadata.
        """
        counter = 0
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            counter += 1
            yield Heartbeat(
                run_id=run_id,
                node_id=self.name,
                message=f"Waiting for human approval ({counter * self.heartbeat_interval:.0f}s)",
                metadata={
                    "status": "waiting_for_human",
                    "elapsed_seconds": counter * self.heartbeat_interval,
                    "heartbeat_count": counter,
                },
            )


class HumanReviewNode[InputT, OutputT](NodeWithInput[InputT, OutputT]):
    """Node that allows human to review and optionally modify data.
    
    Unlike HumanApprovalNode, this always proceeds but gives humans
    a chance to intervene. Think of it as "review and modify" vs "approve/reject".
    
    Example:
        async def review_summary(summary: str, context: dict) -> HumanDecision:
            # Show summary to human, allow editing
            edited = await show_editor(summary)
            return HumanDecision(
                approved=True,
                modified_value=edited if edited != summary else None,
                feedback="Enhanced clarity"
            )
        
        review_node = HumanReviewNode[str, str](
            review_func=review_summary,
            input=summary_node.output,
            default_timeout=30.0,  # Auto-proceed after 30s
            name="review_summary"
        )
    """
    
    def __init__(
        self,
        review_func: HumanDecisionFunc[InputT],
        *,
        input: Any = None,
        default_timeout: float = 30.0,  # Auto-proceed if no response
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize a HumanReviewNode.
        
        Args:
            review_func: Async function that gets human review
            input: Optional input from another node's output
            default_timeout: Seconds before auto-proceeding
            name: Optional unique identifier for this node
            run_id: Optional run identifier for tracking execution
        """
        super().__init__(input, name, run_id)
        self.review_func = review_func
        self.default_timeout = default_timeout
    
    async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
        """Stream progress while getting human review."""
        actual_run_id = self.run_id or ""
        
        yield StreamStart(
            run_id=actual_run_id,
            node_id=self.name,
            input_preview=self._preview_input(input_data),
        )
        
        context = {
            "node_id": self.name,
            "run_id": actual_run_id,
        }
        
        try:
            decision = await asyncio.wait_for(
                self.review_func(input_data, context),
                timeout=self.default_timeout,
            )
            
            # Use modified value if provided, otherwise pass through
            output = decision.modified_value if decision.modified_value is not None else input_data
            
        except asyncio.TimeoutError:
            # Auto-proceed with original value
            output = input_data
        
        yield StreamEnd(
            run_id=actual_run_id,
            node_id=self.name,
            result_preview=self._preview_input(output),  # type: ignore
        )
```

### 4. Node-Level Interrupt Handlers

Nodes can register handlers with priorities to control execution order.

```python
# src/pydantic_flow/nodes/base.py (MODIFIED)

from collections.abc import Awaitable
from collections.abc import Callable

# New type alias
NodeInterruptHandler = Callable[
    ['BaseNode', 'NodeLifecycleEvent', Any],  # node, event, data
    Awaitable[InterruptDecision]
]


class HandlerPriority(IntEnum):
    """Priority levels for interrupt handlers.
    
    Lower values execute first. Critical handlers (0-25) always run.
    """
    
    CRITICAL = 0      # Security, safety checks
    HIGH = 25         # Important validation
    NORMAL = 50       # Standard business logic (default)
    LOW = 75          # Logging, telemetry
    BACKGROUND = 100  # Analytics, metrics


class InterruptHandlerRegistration(BaseModel):
    """Registration info for an interrupt handler."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    handler: NodeInterruptHandler
    priority: int = Field(default=HandlerPriority.NORMAL, ge=0, le=100)
    name: str | None = None  # For debugging
    
    def __lt__(self, other: 'InterruptHandlerRegistration') -> bool:
        """Sort by priority (lower first)."""
        return self.priority < other.priority


class NodeLifecycleEvent(StrEnum):
    """Node lifecycle events that can trigger interruption."""
    
    BEFORE_START = "before_start"
    AFTER_START = "after_start"
    BEFORE_EXECUTION = "before_execution"
    AFTER_EXECUTION = "after_execution"
    BEFORE_END = "before_end"
    AFTER_END = "after_end"


class BaseNode[InputT, OutputT](ABC):
    """Abstract base class for all workflow nodes.
    
    BREAKING CHANGE: Added priority-based interrupt_handlers for HITL support.
    """
    
    def __init__(self, name: str | None = None, run_id: str | None = None) -> None:
        self.name = name or f"{self.__class__.__name__}_{id(self):x}"
        self.run_id = run_id
        self._output: NodeOutput[OutputT] = NodeOutput(node=self)
        self._input_type: type[InputT] = self.__class__.__orig_bases__[0].__args__[0]
        self._output_type: type[OutputT] = self.__class__.__orig_bases__[0].__args__[1]
    
    def register_interrupt_handler(
        self,
        callback: InterruptCallback,
        priority: int = HandlerPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register an interrupt callback handler for this node.
        
        Handlers execute in priority order (lower first). Critical handlers
        (priority 0-25) always execute even if higher priority handler interrupts.
        
        Args:
            callback: Async function that receives ProgressItem and returns
                InterruptDecision.
            priority: Execution priority (0-100, default 50)
            metadata: Optional metadata about the handler.
        """
        # Implementation provided by InterruptibleNodeMixin
        pass
    
    async def _check_lifecycle_interrupt(
        self,
        event: NodeLifecycleEvent,
        data: Any,
    ) -> None:
        """Check all handlers for this lifecycle event in priority order.
        
        Critical handlers (priority 0-25) always run. Higher priority handlers
        can short-circuit remaining handlers by interrupting.
        
        Raises:
            InterruptionRequested: If any handler requests interruption
        """
        interruption: InterruptDecision | None = None
        
        for registration in self._interrupt_handlers:
            # Always run critical handlers
            is_critical = registration.priority <= HandlerPriority.HIGH
            
            # Skip non-critical if already interrupted
            if interruption is not None and not is_critical:
                break
            
            try:
                decision = await registration.handler(self, event, data)
                
                # First interrupt wins (unless overridden by critical)
                if decision.should_interrupt and interruption is None:
                    interruption = decision
                
            except Exception as e:
                # Handler errors don't stop other handlers
                logger.warning(
                    f"Interrupt handler '{registration.name}' failed: {e}"
                )
        
        # Raise if any handler requested interruption
        if interruption is not None:
            synthetic_event = ProgressItem(
                type=ProgressType.START if "start" in event else ProgressType.END,
                run_id=self.run_id or "",
                node_id=self.name,
            )
            raise InterruptionRequested(
                decision=interruption,
                interrupted_at=synthetic_event,
            )
    
    @abstractmethod
    async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the node's logic.
        
        BREAKING CHANGE: Now checks for lifecycle interruptions.
        """
        # Check before start
        await self._check_lifecycle_interrupt(
            NodeLifecycleEvent.BEFORE_START,
            input_data,
        )
        
        yield StreamStart(
            run_id=self.run_id or "",
            node_id=self.name,
            input_preview=self._preview_input(input_data),
        )
        
        # Subclass implements actual streaming logic here
        yield  # type: ignore
        
        yield StreamEnd(
            run_id=self.run_id or "",
            node_id=self.name,
        )
        
        # Check after end
        await self._check_lifecycle_interrupt(
            NodeLifecycleEvent.AFTER_END,
            None,  # No data after end
        )
    
    async def run(self, input_data: InputT) -> OutputT:
        """Execute the node and return the final validated result.
        
        BREAKING CHANGE: Now handles InterruptionRequested exceptions.
        """
        final_result: OutputT | None = None
        tool_result: Any = None
        
        try:
            async for item in self.astream(input_data):
                # Check for event-level interruption
                decision = await item.check_interrupt()
                if decision.should_interrupt:
                    raise InterruptionRequested(
                        decision=decision,
                        interrupted_at=item,
                    )
                
                # Extract result from ToolResult or StreamEnd
                if isinstance(item, ToolResult) and item.result is not None:
                    tool_result = item.result
                elif isinstance(item, StreamEnd) and item.result_preview:
                    try:
                        if hasattr(self._output_type, "model_validate"):
                            final_result = self._output_type.model_validate(
                                item.result_preview
                            )
                        else:
                            final_result = item.result_preview
                    except Exception:
                        final_result = item.result_preview
        
        except InterruptionRequested:
            # Re-raise to allow flow-level handling
            raise
        
        if tool_result is not None:
            final_result = tool_result
        
        if final_result is None:
            msg = f"Node {self.name} did not produce a result"
            raise RuntimeError(msg)
        
        return final_result
```

**Rationale**: Priority system ensures critical handlers (security, safety) always run while allowing flexible ordering for standard business logic.

### 5. Flow-Level Interrupt Handlers

Flows can register handlers that observe node transitions and support resumption.

```python
# src/pydantic_flow/flow/flow.py (MODIFIED)

from collections.abc import Awaitable
from collections.abc import Callable
import uuid

# New type alias
FlowInterruptHandler = Callable[
    ['Flow', 'FlowTransitionEvent', str, Any],  # flow, event, node_name, data
    Awaitable[InterruptDecision]
]

class FlowTransitionEvent(StrEnum):
    """Flow transition events that can trigger interruption."""
    
    FLOW_START = "flow_start"
    NODE_QUEUED = "node_queued"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    FLOW_COMPLETE = "flow_complete"


class Flow[InputT: BaseModel, OutputT: BaseModel]:
    """A workflow orchestrator that manages node execution and dependencies.
    
    BREAKING CHANGE: Added interrupt_handlers and resume support for flow-level HITL.
    """
    
    def __init__(self, *, input_type: type[InputT], output_type: type[OutputT]) -> None:
        self.nodes: list[BaseNode[Any, Any]] = []
        self._execution_order: list[BaseNode[Any, Any]] = []
        self._results: dict[str, Any] = {}
        self._input_type = input_type
        self._output_type = output_type
        self._edges: dict[str, list[str]] = {}
        self._conditional_edges: list[ConditionalEdge[Any]] = []
        self._entry_nodes: list[str] | None = None
        
        # NEW: Flow-level interrupt handlers with priorities
        self._interrupt_handlers: list[InterruptHandlerRegistration] = []
        
        # NEW: Flow ID for checkpoint tracking
        self.flow_id = str(uuid.uuid4())
    
    def add_interrupt_handler(
        self,
        handler: FlowInterruptHandler,
        priority: int = HandlerPriority.NORMAL,
        name: str | None = None,
    ) -> None:
        """Register a flow-level interrupt handler with priority.
        
        The handler will be called at flow transition points and can
        interrupt the entire flow.
        
        Args:
            handler: Async function that receives (flow, event, node_name, data)
                    and returns InterruptDecision
            priority: Execution priority (0-100, default 50)
            name: Optional name for debugging
        """
        registration = InterruptHandlerRegistration(
            handler=handler,
            priority=priority,
            name=name or handler.__name__,
        )
        self._interrupt_handlers.append(registration)
        self._interrupt_handlers.sort()
    
    async def _check_flow_interrupt(
        self,
        event: FlowTransitionEvent,
        node_name: str,
        data: Any,
    ) -> None:
        """Check all handlers for this flow transition in priority order.
        
        Raises:
            InterruptionRequested: If any handler requests interruption
        """
        interruption: InterruptDecision | None = None
        
        for registration in self._interrupt_handlers:
            is_critical = registration.priority <= HandlerPriority.HIGH
            
            if interruption is not None and not is_critical:
                break
            
            try:
                decision = await registration.handler(self, event, node_name, data)
                
                if decision.should_interrupt and interruption is None:
                    interruption = decision
                    
            except Exception as e:
                logger.warning(
                    f"Flow interrupt handler '{registration.name}' failed: {e}"
                )
        
        if interruption is not None:
            synthetic_event = ProgressItem(
                type=ProgressType.START,
                run_id="",
                node_id=node_name,
            )
            
            # Create checkpoint for resumption
            checkpoint = self._create_checkpoint(node_name, data)
            
            raise InterruptionRequested(
                decision=interruption,
                interrupted_at=synthetic_event,
                checkpoint=checkpoint,
            )
    
    def _create_checkpoint(
        self,
        current_node: str | None,
        inputs: Any,
    ) -> FlowCheckpoint:
        """Create a checkpoint of current flow state."""
        # Determine pending nodes
        completed = set(self._results.keys())
        all_nodes = {node.name for node in self.nodes}
        pending = list(all_nodes - completed)
        
        # Serialize inputs (must be BaseModel or dict)
        if hasattr(inputs, "model_dump"):
            serialized_inputs = inputs.model_dump()
        elif isinstance(inputs, dict):
            serialized_inputs = inputs
        else:
            serialized_inputs = {"value": str(inputs)}
        
        return FlowCheckpoint(
            flow_id=self.flow_id,
            execution_mode="dag",  # Will be overridden by stepper if needed
            results=dict(self._results),
            pending_nodes=pending,
            current_node=current_node,
            inputs=serialized_inputs,
            metadata={
                "edges": self._edges,
                "entry_nodes": self._entry_nodes,
            },
        )
    
    async def run(self, inputs: InputT) -> OutputT:
        """Execute the flow with the given inputs.
        
        BREAKING CHANGE: Now checks for flow-level interruptions and supports resumption.
        """
        if not isinstance(inputs, self._input_type):
            expected_name = self._input_type.__name__
            actual_name = type(inputs).__name__
            msg = f"Input type mismatch: expected {expected_name}, got {actual_name}"
            raise TypeError(msg)
        
        self._results = {}
        
        try:
            # Check flow start
            await self._check_flow_interrupt(
                FlowTransitionEvent.FLOW_START,
                "",
                inputs,
            )
            
            for node in self._execution_order:
                # Check before node execution
                await self._check_flow_interrupt(
                    FlowTransitionEvent.NODE_STARTED,
                    node.name,
                    None,
                )
                
                # Determine input data
                if has_multiple_inputs(node):
                    input_data: Any = tuple(
                        self._results[dep.node.name] for dep in node.inputs
                    )
                elif has_input_dependency(node):
                    input_node = node.input.node
                    if input_node.name not in self._results:
                        msg = f"Input node {input_node.name} has not been executed"
                        raise FlowError(msg)
                    input_data = self._results[input_node.name]
                else:
                    input_data = inputs
                
                # Execute the node (may raise InterruptionRequested)
                result = await extract_result_from_stream(node.astream(input_data)
                self._results[node.name] = result
                
                # Check after node execution
                await self._check_flow_interrupt(
                    FlowTransitionEvent.NODE_COMPLETED,
                    node.name,
                    result,
                )
            
            # Check flow completion
            await self._check_flow_interrupt(
                FlowTransitionEvent.FLOW_COMPLETE,
                "",
                self._results,
            )
            
            return self._output_type(**self._results)
        
        except InterruptionRequested as e:
            # Update checkpoint with current execution mode
            if e.checkpoint:
                e.checkpoint.execution_mode = "dag"
            raise
        except Exception as e:
            if isinstance(e, FlowError):
                raise
            msg = f"Flow execution failed: {e}"
            raise FlowError(msg) from e
    
    async def resume(
        self,
        checkpoint: FlowCheckpoint,
        updated_results: dict[str, Any] | None = None,
    ) -> OutputT:
        """Resume flow execution from a checkpoint.
        
        Args:
            checkpoint: Previously created checkpoint
            updated_results: Optional modifications to checkpointed results
                           (e.g., human-modified values)
        
        Returns:
            Flow output after resuming execution
        
        Raises:
            ValueError: If checkpoint is invalid or incompatible
        """
        if checkpoint.flow_id != self.flow_id:
            msg = f"Checkpoint flow_id mismatch: {checkpoint.flow_id} != {self.flow_id}"
            raise ValueError(msg)
        
        # Restore results
        self._results = dict(checkpoint.results)
        
        # Apply any updated results
        if updated_results:
            self._results.update(updated_results)
        
        # Reconstruct inputs
        inputs = self._input_type.model_validate(checkpoint.inputs)
        
        # Find nodes to execute (those in pending_nodes)
        pending_names = set(checkpoint.pending_nodes)
        nodes_to_execute = [
            node for node in self._execution_order
            if node.name in pending_names
        ]
        
        try:
            for node in nodes_to_execute:
                await self._check_flow_interrupt(
                    FlowTransitionEvent.NODE_STARTED,
                    node.name,
                    None,
                )
                
                # Determine input data (same logic as run())
                if has_multiple_inputs(node):
                    input_data: Any = tuple(
                        self._results[dep.node.name] for dep in node.inputs
                    )
                elif has_input_dependency(node):
                    input_node = node.input.node
                    if input_node.name not in self._results:
                        msg = f"Input node {input_node.name} has not been executed"
                        raise FlowError(msg)
                    input_data = self._results[input_node.name]
                else:
                    input_data = inputs
                
                result = await extract_result_from_stream(node.astream(input_data)
                self._results[node.name] = result
                
                await self._check_flow_interrupt(
                    FlowTransitionEvent.NODE_COMPLETED,
                    node.name,
                    result,
                )
            
            await self._check_flow_interrupt(
                FlowTransitionEvent.FLOW_COMPLETE,
                "",
                self._results,
            )
            
            return self._output_type(**self._results)
            
        except InterruptionRequested as e:
            if e.checkpoint:
                e.checkpoint.execution_mode = "dag"
            raise
        except Exception as e:
            if isinstance(e, FlowError):
                raise
            msg = f"Flow resumption failed: {e}"
            raise FlowError(msg) from e
```

**Rationale**: Resume support enables long-running workflows with human approval delays. Checkpoints are fully serializable for storage in databases or message queues.

### 6. Stepper Engine Integration

The stepper engine needs comprehensive interrupt support for loop scenarios.

```python
# src/pydantic_flow/engine/stepper.py (MODIFIED)

class StepperEngine[InputT: BaseModel, OutputT: BaseModel]:
    """Loop-capable execution engine using frontier-based stepping.
    
    BREAKING CHANGE: Added comprehensive interrupt support for HITL in loops.
    """
    
    def __init__(self, config: EngineConfig[InputT, OutputT]) -> None:
        self.nodes_by_name = {node.name: node for node in config.nodes}
        self.edges = config.edges
        self.conditional_edges = config.conditional_edges
        self.entry_nodes = config.entry_nodes
        self.input_type = config.input_type
        self.output_type = config.output_type
        
        # NEW: Track interrupt handlers
        self._interrupt_handlers: list[InterruptHandlerRegistration] = []
    
    async def invoke(
        self,
        inputs: InputT,
        config: RunConfig | None = None,
        interrupt_handlers: list[tuple[FlowInterruptHandler, int, str | None]] | None = None,
    ) -> OutputT:
        """Execute the flow with the given inputs.
        
        Args:
            inputs: Input data matching InputT
            config: Optional execution configuration
            interrupt_handlers: Optional flow-level interrupt handlers as
                              list of (handler, priority, name) tuples
        
        Returns:
            Output data matching OutputT
        
        Raises:
            RecursionLimitError: If max_steps is exceeded
            FlowTimeoutError: If timeout_seconds is exceeded
            RoutingError: If routing targets are invalid
            InterruptionRequested: If HITL interruption occurs
            FlowError: For other execution errors
        
        BREAKING CHANGE: Added interrupt_handlers parameter and interrupt support.
        """
        if config is None:
            config = RunConfig()
        
        # Register interrupt handlers with priorities
        self._interrupt_handlers = []
        if interrupt_handlers:
            for handler, priority, name in interrupt_handlers:
                self._interrupt_handlers.append(
                    InterruptHandlerRegistration(
                        handler=handler,
                        priority=priority,
                        name=name,
                    )
                )
            self._interrupt_handlers.sort()
        
        self._validate_input_type(inputs)
        
        start_time = time.time()
        results: dict[str, Any] = {}
        frontier = set(self.entry_nodes)
        iteration = 0
        events: list[IterationEvent] = []
        
        try:
            # Check flow start interrupt
            await self._check_stepper_interrupt(
                "flow_start",
                "",
                inputs,
                iteration,
                results,
            )
            
            while frontier:
                self._check_limits(iteration, config, start_time, events)
                
                current_frontier = list(frontier)
                
                # Check before frontier execution
                await self._check_stepper_interrupt(
                    "frontier_start",
                    "|".join(current_frontier),
                    None,
                    iteration,
                    results,
                )
                
                frontier = set()
                
                await self._execute_frontier(current_frontier, inputs, results)
                
                # Check after frontier execution
                await self._check_stepper_interrupt(
                    "frontier_complete",
                    "|".join(current_frontier),
                    results,
                    iteration,
                    results,
                )
                
                # Route next (with interrupt support)
                next_frontier, ended = await self._route_next_with_interrupt(
                    current_frontier,
                    results,
                    iteration,
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                if config.trace_iterations:
                    event = IterationEvent(
                        iteration=iteration,
                        frontier=current_frontier,
                        routed_to=list(next_frontier),
                        ended=ended,
                        elapsed_ms=elapsed_ms,
                    )
                    events.append(event)
                
                if ended:
                    break
                
                frontier = next_frontier
                iteration += 1
            
            # Check flow completion
            await self._check_stepper_interrupt(
                "flow_complete",
                "",
                results,
                iteration,
                results,
            )
            
            return self.output_type(**results)
        
        except InterruptionRequested as e:
            # Create checkpoint with stepper-specific state
            checkpoint = self._create_stepper_checkpoint(
                inputs,
                results,
                frontier,
                iteration,
            )
            # Attach checkpoint to exception
            e.checkpoint = checkpoint
            raise
        except FlowError:
            raise
        except Exception as e:
            msg = f"Flow execution failed: {e}"
            raise FlowError(msg) from e
    
    async def _check_stepper_interrupt(
        self,
        event_type: str,
        node_name: str,
        data: Any,
        iteration: int,
        results: dict[str, Any],
    ) -> None:
        """Check interrupt handlers with loop-specific context.
        
        Args:
            event_type: Type of event (flow_start, frontier_start, etc.)
            node_name: Name of current node (or frontier nodes joined)
            data: Data associated with the event
            iteration: Current iteration number
            results: Current results dict
        
        Raises:
            InterruptionRequested: If any handler requests interruption
        """
        interruption: InterruptDecision | None = None
        
        for registration in self._interrupt_handlers:
            is_critical = registration.priority <= HandlerPriority.HIGH
            
            if interruption is not None and not is_critical:
                break
            
            try:
                # Create temporary flow-like object for handler signature
                # In practice, this would be the actual flow
                flow_context = type('FlowContext', (), {
                    'nodes': list(self.nodes_by_name.values()),
                    'results': results,
                    'iteration': iteration,
                })()
                
                decision = await registration.handler(
                    flow_context,  # type: ignore
                    event_type,  # type: ignore
                    node_name,
                    data,
                )
                
                if decision.should_interrupt and interruption is None:
                    # Add iteration context to metadata
                    decision.metadata.update({
                        "iteration": iteration,
                        "event_type": event_type,
                        "stepper_mode": True,
                    })
                    interruption = decision
                    
            except Exception as e:
                logger.warning(
                    f"Stepper interrupt handler '{registration.name}' failed: {e}"
                )
        
        if interruption is not None:
            synthetic_event = ProgressItem(
                type=ProgressType.START,
                run_id="",
                node_id=node_name,
            )
            raise InterruptionRequested(
                decision=interruption,
                interrupted_at=synthetic_event,
            )
    
    async def _route_next_with_interrupt(
        self,
        current_frontier: list[str],
        results: dict[str, Any],
        iteration: int,
    ) -> tuple[set[str], bool]:
        """Route to next frontier with interrupt support.
        
        Allows interrupt handlers to modify routing decisions (e.g., force END).
        """
        next_frontier: set[str] = set()
        ended = False
        
        for node_name in current_frontier:
            # Static edges
            static_targets = self.edges.get(node_name, [])
            next_frontier.update(static_targets)
            
            # Conditional edges with interrupt checks
            for cond_edge in self.conditional_edges:
                if cond_edge.from_node == node_name:
                    # Check before routing decision
                    await self._check_stepper_interrupt(
                        "before_routing",
                        node_name,
                        results,
                        iteration,
                        results,
                    )
                    
                    targets, edge_ended = self._apply_conditional_edge(
                        cond_edge, results
                    )
                    next_frontier.update(targets)
                    ended = ended or edge_ended
                    
                    # Check after routing decision
                    await self._check_stepper_interrupt(
                        "after_routing",
                        node_name,
                        {"targets": list(targets), "ended": edge_ended},
                        iteration,
                        results,
                    )
        
        return next_frontier, ended
    
    def _create_stepper_checkpoint(
        self,
        inputs: InputT,
        results: dict[str, Any],
        frontier: set[str],
        iteration: int,
    ) -> FlowCheckpoint:
        """Create checkpoint with stepper-specific state."""
        # Determine pending nodes
        completed = set(results.keys())
        all_nodes = set(self.nodes_by_name.keys())
        pending = list(all_nodes - completed)
        
        # Serialize inputs
        if hasattr(inputs, "model_dump"):
            serialized_inputs = inputs.model_dump()
        elif isinstance(inputs, dict):
            serialized_inputs = inputs
        else:
            serialized_inputs = {"value": str(inputs)}
        
        return FlowCheckpoint(
            flow_id=str(uuid.uuid4()),  # Would use actual flow ID
            execution_mode="stepper",
            results=dict(results),
            pending_nodes=pending,
            current_node=None,  # Not applicable in stepper
            inputs=serialized_inputs,
            metadata={
                "edges": self.edges,
                "entry_nodes": self.entry_nodes,
            },
            # Stepper-specific state
            iteration=iteration,
            frontier=list(frontier),
        )
    
    async def resume(
        self,
        checkpoint: FlowCheckpoint,
        config: RunConfig | None = None,
        updated_results: dict[str, Any] | None = None,
    ) -> OutputT:
        """Resume stepper execution from checkpoint.
        
        Args:
            checkpoint: Previously created checkpoint
            config: Optional execution configuration
            updated_results: Optional modifications to results
        
        Returns:
            Flow output after resuming
        
        Raises:
            ValueError: If checkpoint is invalid
        """
        if checkpoint.execution_mode != "stepper":
            msg = f"Invalid checkpoint mode: {checkpoint.execution_mode}"
            raise ValueError(msg)
        
        if config is None:
            config = RunConfig()
        
        # Restore state
        results = dict(checkpoint.results)
        if updated_results:
            results.update(updated_results)
        
        # Reconstruct inputs
        inputs = self.input_type.model_validate(checkpoint.inputs)
        
        # Resume from checkpointed iteration and frontier
        iteration = checkpoint.iteration or 0
        frontier = set(checkpoint.frontier or self.entry_nodes)
        
        start_time = time.time()
        events: list[IterationEvent] = []
        
        try:
            while frontier:
                self._check_limits(iteration, config, start_time, events)
                
                current_frontier = list(frontier)
                
                await self._check_stepper_interrupt(
                    "frontier_start",
                    "|".join(current_frontier),
                    None,
                    iteration,
                    results,
                )
                
                frontier = set()
                
                await self._execute_frontier(current_frontier, inputs, results)
                
                await self._check_stepper_interrupt(
                    "frontier_complete",
                    "|".join(current_frontier),
                    results,
                    iteration,
                    results,
                )
                
                next_frontier, ended = await self._route_next_with_interrupt(
                    current_frontier,
                    results,
                    iteration,
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                if config.trace_iterations:
                    event = IterationEvent(
                        iteration=iteration,
                        frontier=current_frontier,
                        routed_to=list(next_frontier),
                        ended=ended,
                        elapsed_ms=elapsed_ms,
                    )
                    events.append(event)
                
                if ended:
                    break
                
                frontier = next_frontier
                iteration += 1
            
            await self._check_stepper_interrupt(
                "flow_complete",
                "",
                results,
                iteration,
                results,
            )
            
            return self.output_type(**results)
            
        except InterruptionRequested as e:
            checkpoint = self._create_stepper_checkpoint(
                inputs,
                results,
                frontier,
                iteration,
            )
            e.checkpoint = checkpoint
            raise
        except FlowError:
            raise
        except Exception as e:
            msg = f"Flow resumption failed: {e}"
            raise FlowError(msg) from e
```

**Rationale**: Comprehensive interrupt support at every stepper transition point - frontiers, routing decisions, and conditional edges. Checkpoints include iteration and frontier state for perfect resumption in loops.

---

## Usage Examples

### Example 1: Simple Approval Node

```python
async def approve_weather(
    weather: WeatherInfo,
    context: dict,
) -> HumanDecision[WeatherInfo]:
    """Show weather to human for approval."""
    print(f"Weather: {weather.temperature}°C, {weather.condition}")
    print("Approve? (y/n): ", end="")
    
    # In real app, this would be async input from web UI
    response = await asyncio.to_thread(input)
    
    return HumanDecision(
        approved=response.lower() == "y",
        feedback="Manual review completed",
    )


# Build flow with approval step
flow = Flow(input_type=Query, output_type=Results)

weather_node = ToolNode[Query, WeatherInfo](
    tool_func=get_weather,
    name="weather"
)

approval_node = HumanApprovalNode[WeatherInfo, WeatherInfo](
    decision_func=approve_weather,
    input=weather_node.output,
    timeout_seconds=300,  # 5 min timeout
    name="approve"
)

summary_node = PromptNode[WeatherInfo, str](
    prompt="Summarize: {temperature}°C, {condition}",
    input=approval_node.output,
    name="summary"
)

flow.add_nodes(weather_node, approval_node, summary_node)

try:
    result = await extract_result_from_stream(flow.astream(Query(location="Paris"))
except InterruptionRequested as e:
    print(f"Flow interrupted: {e.decision.reason}")
```

### Example 2: Event-Level Interruption (Token Limit)

```python
class TokenLimitInterrupter:
    """Interrupt LLM after N tokens."""
    
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.token_count = 0
    
    async def __call__(self, item: ProgressItem) -> InterruptDecision:
        if isinstance(item, TokenChunk):
            self.token_count += 1
            if self.token_count >= self.max_tokens:
                return InterruptDecision(
                    should_interrupt=True,
                    reason=f"Reached token limit of {self.max_tokens}",
                )
        return InterruptDecision(should_interrupt=False)


# Create node with token limit
agent_node = AgentNode[Query, str](
    agent=agent,
    prompt_template="{question}",
    name="agent"
)

# Add interrupt callback to all events
limiter = TokenLimitInterrupter(max_tokens=100)

async for item in agent_node.astream(query):
    item.interrupt_callback = limiter
    decision = await item.check_interrupt()
    if decision.should_interrupt:
        print(f"Stopped at {limiter.token_count} tokens")
        break
    
    if isinstance(item, TokenChunk):
        print(item.text, end="")
```

### Example 3: Node Lifecycle Handler

```python
async def review_before_summary(
    node: BaseNode,
    event: NodeLifecycleEvent,
    data: Any,
) -> InterruptDecision:
    """Review data before summary node executes."""
    if event == NodeLifecycleEvent.BEFORE_EXECUTION:
        print(f"About to summarize: {data}")
        print("Proceed? (y/n): ", end="")
        response = await asyncio.to_thread(input)
        
        if response.lower() != "y":
            return InterruptDecision(
                should_interrupt=True,
                reason="User cancelled summary",
            )
    
    return InterruptDecision(should_interrupt=False)


summary_node = PromptNode[WeatherInfo, str](
    prompt="Summarize: {temperature}°C",
    input=weather_node.output,
    name="summary"
)

summary_node.add_interrupt_handler(review_before_summary)
```

### Example 4: Flow-Level Monitoring

```python
async def audit_flow_transitions(
    flow: Flow,
    event: FlowTransitionEvent,
    node_name: str,
    data: Any,
) -> InterruptDecision:
    """Audit all node transitions, interrupt if needed."""
    
    # Log to audit system
    await log_to_audit_db(event, node_name, data)
    
    # Interrupt if we detect sensitive data
    if event == FlowTransitionEvent.NODE_COMPLETED:
        if detect_pii(data):
            return InterruptDecision(
                should_interrupt=True,
                reason="PII detected - human review required",
                metadata={"node": node_name, "data_preview": str(data)[:100]},
            )
    
    return InterruptDecision(should_interrupt=False)


flow = Flow(input_type=Input, output_type=Output)
flow.add_interrupt_handler(audit_flow_transitions)
```

### Example 5: Streaming with Mid-Flight Interruption

```python
async def process_with_interruption():
    """Process LLM stream with ability to interrupt mid-response."""
    
    node = AgentNode[Query, str](agent=agent, name="agent")
    query = Query(question="Write a long essay...")
    
    # User can press Ctrl+C to interrupt
    user_interrupted = asyncio.Event()
    
    async def interrupt_on_signal(item: ProgressItem) -> InterruptDecision:
        if user_interrupted.is_set():
            return InterruptDecision.interrupt(
                reason="User interrupted",
                user_action="cancel",
            )
        return InterruptDecision.proceed()
    
    try:
        async for item in node.astream(query):
            # Attach interrupt callback (supports both patterns)
            item.interrupt_callback = interrupt_on_signal
            
            if isinstance(item, TokenChunk):
                print(item.text, end="", flush=True)
            
            # Check interrupt after each item
            decision = await item.check_interrupt()
            if decision.should_interrupt:
                print(f"\n[Interrupted: {decision.reason}]")
                break
                
    except KeyboardInterrupt:
        user_interrupted.set()
        print("\n[User stopped generation]")
```

### Example 6: Loop Interruption with Resumption

```python
async def loop_with_approval():
    """Loop workflow that requires approval every 3 iterations."""
    
    flow = Flow(input_type=CounterState, output_type=OutputState)
    tick_node = TickNode(name="tick")
    flow.add_nodes(tick_node)
    flow.set_entry_nodes("tick")
    
    # Router loops until n >= 10
    def router(state: BaseModel) -> T_Route:
        tick_state = getattr(state, "tick", None)
        if tick_state and tick_state.n >= 10:
            return Route.END
        return "tick"
    
    flow.add_conditional_edges("tick", router)
    
    # Interrupt handler: require approval every 3 iterations
    approval_granted = {}
    
    async def require_periodic_approval(
        flow_obj: Flow,
        event: FlowTransitionEvent,
        node_name: str,
        data: Any,
    ) -> InterruptDecision:
        """Interrupt every 3 iterations for human approval."""
        
        if event == "frontier_complete":
            # Get iteration from metadata (added by stepper)
            iteration = getattr(flow_obj, 'iteration', 0)
            
            if iteration > 0 and iteration % 3 == 0:
                # Check if approval already granted for this iteration
                if iteration not in approval_granted:
                    print(f"\n[Iteration {iteration}] Current state: {data}")
                    print("Continue? (y/n): ", end="")
                    response = await asyncio.to_thread(input)
                    
                    if response.lower() == 'y':
                        approval_granted[iteration] = True
                        return InterruptDecision.proceed()
                    else:
                        return InterruptDecision.interrupt(
                            reason=f"Human stopped at iteration {iteration}",
                            iteration=iteration,
                            current_state=data,
                        )
        
        return InterruptDecision.proceed()
    
    # Register with high priority
    flow.add_interrupt_handler(
        require_periodic_approval,
        priority=HandlerPriority.HIGH,
        name="periodic_approval"
    )
    
    # Compile and run
    compiled = flow.compile()
    config = RunConfig(max_steps=50)
    
    checkpoint = None
    try:
        result = await extract_result_from_stream(compiled.astream(
            CounterState(n=0),
            config,
        )
        print(f"\nCompleted: n = {result.tick.n}")
        
    except InterruptionRequested as e:
        print(f"\n[Interrupted: {e.decision.reason}]")
        checkpoint = e.checkpoint
        
        # Show checkpoint info
        if checkpoint:
            print(f"Checkpoint created:")
            print(f"  - Iteration: {checkpoint.iteration}")
            print(f"  - Results: {list(checkpoint.results.keys())}")
            print(f"  - Frontier: {checkpoint.frontier}")
            
            # Simulate human review and modification
            print("\nModifying counter value and resuming...")
            await asyncio.sleep(1)  # Simulate delay
            
            # Resume with modified value
            updated_results = {
                "tick": CounterState(n=checkpoint.results["tick"].n + 1)
            }
            
            result = await compiled.engine.resume(
                checkpoint,
                config,
                updated_results,
            )
            print(f"Resumed and completed: n = {result.tick.n}")
```

### Example 7: Priority-Based Handlers

```python
async def security_and_logging():
    """Demonstrate priority-based handler execution."""
    
    flow = Flow(input_type=Input, output_type=Output)
    
    # Critical security handler (priority 0)
    async def security_check(
        flow_obj: Flow,
        event: FlowTransitionEvent,
        node_name: str,
        data: Any,
    ) -> InterruptDecision:
        """Check for security violations - always runs."""
        if event == FlowTransitionEvent.NODE_COMPLETED:
            if detect_sensitive_data(data):
                return InterruptDecision.interrupt(
                    reason="Security: PII detected",
                    security_level="critical",
                )
        return InterruptDecision.proceed()
    
    # Standard business logic (priority 50)
    async def business_validation(
        flow_obj: Flow,
        event: FlowTransitionEvent,
        node_name: str,
        data: Any,
    ) -> InterruptDecision:
        """Validate business rules."""
        if event == FlowTransitionEvent.NODE_COMPLETED:
            if not validate_business_rules(data):
                return InterruptDecision.interrupt(
                    reason="Business rule violation",
                )
        return InterruptDecision.proceed()
    
    # Low-priority logging (priority 90)
    async def audit_logger(
        flow_obj: Flow,
        event: FlowTransitionEvent,
        node_name: str,
        data: Any,
    ) -> InterruptDecision:
        """Log for audit trail - runs if no interrupt."""
        await log_to_audit_system(event, node_name, data)
        return InterruptDecision.proceed()
    
    # Register handlers with priorities
    flow.add_interrupt_handler(security_check, HandlerPriority.CRITICAL, "security")
    flow.add_interrupt_handler(business_validation, HandlerPriority.NORMAL, "validation")
    flow.add_interrupt_handler(audit_logger, HandlerPriority.BACKGROUND, "audit")
    
    # Execution order: security → validation → audit
    # If security interrupts, validation and audit are skipped
    # If validation interrupts, audit is skipped
    # Security always runs regardless of other interrupts
```

### Example 8: Flexible Metadata

```python
async def metadata_rich_interruption():
    """Show flexible metadata usage."""
    
    async def approval_with_context(
        weather: WeatherInfo,
        context: dict,
    ) -> HumanDecision[WeatherInfo]:
        """Request approval with rich metadata."""
        print(f"Weather: {weather.temperature}°C")
        print(f"Context: {context}")
        
        # Simulate getting response from external system
        response = await get_from_webhook_queue(context["run_id"])
        
        return HumanDecision(
            approved=response["approved"],
            modified_value=WeatherInfo(**response["modified_data"]) if "modified_data" in response else None,
            feedback=response.get("feedback"),
            metadata={
                # User-provided metadata
                "approver_id": response["user_id"],
                "approver_name": response["user_name"],
                "approval_timestamp": response["timestamp"],
                "ip_address": response["ip"],
                "client": response["client"],
                # Any custom fields the user wants
                "department": response.get("department"),
                "project_code": response.get("project_code"),
            }
        )
    
    approval_node = HumanApprovalNode[WeatherInfo, WeatherInfo](
        decision_func=approval_with_context,
        input=weather_node.output,
        timeout_seconds=3600,  # 1 hour
        name="approval"
    )
    
    try:
        result = await extract_result_from_stream(flow.astream(query)
    except InterruptionRequested as e:
        # Access rich metadata
        metadata = e.decision.metadata
        print(f"Rejected by: {metadata.get('approver_name')}")
        print(f"Department: {metadata.get('department')}")
        print(f"All metadata: {metadata}")
        
        # Metadata is serializable in checkpoint
        if e.checkpoint:
            checkpoint_json = e.checkpoint.model_dump_json()
            await save_to_database(checkpoint_json)
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Breaking Changes)
1. Modify `ProgressItem` to add `interrupt_callback` field
2. Create `InterruptDecision` and `InterruptionRequested`
3. Update `extract_result_from_stream(BaseNode.astream()` to check interrupts
4. Update `BaseNode.astream()` to add lifecycle checks
5. Add `NodeLifecycleEvent` enum

### Phase 2: Node-Level HITL
1. Add `NodeInterruptHandler` type and `add_interrupt_handler()` to `BaseNode`
2. Implement `_check_lifecycle_interrupt()` in `BaseNode`
3. Update all node implementations to call lifecycle checks
4. Create `HumanApprovalNode` and `HumanReviewNode`

### Phase 3: Flow-Level HITL
1. Add `FlowInterruptHandler` and `add_interrupt_handler()` to `Flow`
2. Implement `_check_flow_interrupt()` in `extract_result_from_stream(Flow.astream()`
3. Add interrupt hooks to `StepperEngine`
4. Update `CompiledFlow` to pass interrupt handlers

### Phase 4: Testing & Examples
1. Write comprehensive tests for all interrupt scenarios
2. Create example scripts for each HITL pattern
3. Update documentation with HITL guide
4. Performance testing for interrupt overhead

---

## Breaking Changes Summary

### API Changes
- `ProgressItem`: Added `interrupt_callback` field, changed `frozen=False`
- `BaseNode.__init__`: Added `_interrupt_handlers` list
- `BaseNode.astream()`: Added lifecycle interrupt checks
- `extract_result_from_stream(BaseNode.astream()`: Added interrupt exception handling
- `Flow.__init__`: Added `_interrupt_handlers` list
- `extract_result_from_stream(Flow.astream()`: Added flow transition interrupt checks
- `extract_result_from_stream(StepperEngine.astream()`: Added `interrupt_handlers` parameter

### New Exceptions
- `InterruptionRequested`: Special exception for HITL interruptions

### New Classes
- `InterruptDecision`: Return type for interrupt callbacks
- `HumanApprovalNode`: Requires approval to proceed
- `HumanReviewNode`: Optional review/modification
- `NodeLifecycleEvent`: Enum for node lifecycle events
- `FlowTransitionEvent`: Enum for flow transition events

### No Backwards Compatibility
- All changes are breaking
- No deprecation period
- Clean, unified HITL system

---

## Design Rationale

### Why Not Separate Interrupt System?
- **Integration**: Interrupts are part of events, not external
- **Simplicity**: One callback per event, not separate channels
- **Composability**: Callbacks can be chained/composed naturally

### Why Three Interrupt Levels?
- **Event-Level**: Fine-grained (e.g., token limits, real-time monitoring)
- **Node-Level**: Lifecycle hooks (e.g., validation before/after execution)
- **Flow-Level**: Orchestration (e.g., audit trails, global policies)

### Why `frozen=False` for ProgressItem?
- Need to attach callbacks after event creation
- Callbacks are excluded from serialization
- Still immutable from user perspective (only internal callback mutation)

### Why Async Callbacks Everywhere?
- Callbacks may need to query external systems (DB, APIs)
- Callbacks may show UI and wait for user input
- Async is the native pattern for pydantic-flow

### Why Not Context Managers?
- Callbacks are more flexible than context managers
- Can be attached dynamically during streaming
- Don't require wrapping entire node/flow execution

---

## Design Decisions (Resolved)

### 1. Event Callback Attachment
**Decision**: Support both patterns with preference for creation-time setting.
- Primary: Set callback during event creation (cleaner, immutable-friendly)
- Fallback: Allow attachment afterward when dynamic behavior needed
- Implementation: `interrupt_callback` parameter in constructors, plus setter method

### 2. Heartbeat Emission
**Decision**: Async emit heartbeats through the stream to keep connections alive.
- `HumanApprovalNode` uses async generator to yield heartbeats
- Heartbeats include approval status metadata (e.g., "waiting_for_human")
- Configurable interval (default 5 seconds)

### 3. Resume After Interrupt
**Decision**: Full support for resuming flows after interruption.
- Capture flow state (results, current node, inputs) in `InterruptionRequested`
- New `FlowCheckpoint` class for serializable state snapshots
- `Flow.resume()` method to continue from checkpoint
- Stepper engine maintains iteration state for loop resumption

### 4. Handler Priorities
**Decision**: Priority-based handler execution system.
- Handlers registered with priority (0-100, default 50, lower runs first)
- Critical handlers (priority 0-25) always run, can't be skipped
- Standard handlers (26-75) run in order
- Low-priority handlers (76-100) for logging/telemetry
- Short-circuit if high-priority handler interrupts

### 5. Interrupt Metadata Propagation
**Decision**: Lightweight but flexible metadata with user extensions.
- Core fields: `reason`, `replacement_value`, `metadata` dict
- Standard metadata keys: `node_id`, `timestamp`, `event_type`, `user_id`
- User-provided metadata merged into dict (no restrictions)
- Serializable for checkpoint persistence

### 6. Stepper Engine Loop Interrupts
**Decision**: Comprehensive interrupt support in loop scenarios.
- Check interrupts at: frontier execution, routing decisions, conditional edges
- Track loop iteration in interrupt context
- Support interrupt-and-modify for conditional routing (e.g., force `Route.END`)
- Resume with modified iteration count or routing state

---

## Implementation Phases

### Phase 1: Core Infrastructure (Breaking Changes)
**Estimated effort: 3-4 days**

1. Modify `ProgressItem` base class:
   - Add `interrupt_callback` field
   - Add `set_interrupt_callback()` method
   - Implement `check_interrupt()` logic
   - Change `frozen=False`, add `arbitrary_types_allowed=True`
   
2. Create `InterruptDecision` model:
   - Core fields: `should_interrupt`, `reason`, `replacement_value`, `metadata`
   - Add convenience methods: `proceed()`, `interrupt()`
   - Full type annotations
   
3. Create `FlowCheckpoint` model:
   - Fields for DAG and stepper state
   - Serialization support
   - Validation logic
   
4. Create `InterruptionRequested` exception:
   - Extend `FlowError`
   - Carry decision and checkpoint
   - Add `can_resume()` and `get_resume_metadata()` helpers
   
5. Update all progress item types:
   - Ensure all constructors support `interrupt_callback` parameter
   - Test serialization/deserialization
   
6. Create priority and registration types:
   - `HandlerPriority` enum
   - `InterruptHandlerRegistration` model
   - Sorting and comparison logic

### Phase 2: Node-Level HITL
**Estimated effort: 4-5 days**

1. Update `BaseNode`:
   - Add `_interrupt_handlers` list with registrations
   - Implement `add_interrupt_handler()` with priority
   - Implement `_check_lifecycle_interrupt()` with priority ordering
   - Update `astream()` to check lifecycle points
   - Update `run()` to handle `InterruptionRequested`
   
2. Create `NodeLifecycleEvent` enum:
   - Define all lifecycle events
   - Document when each fires
   
3. Create `HumanApprovalNode`:
   - Implement `astream()` with heartbeat support
   - Async generator for heartbeat emission
   - Timeout handling
   - Approval/rejection logic
   - Test with various timeout scenarios
   
4. Create `HumanReviewNode`:
   - Similar to approval but always proceeds
   - Auto-proceed on timeout
   - Modification support
   
5. Update all existing node implementations:
   - Ensure lifecycle checks are called
   - Test interrupt propagation
   - Verify no breaking changes in behavior

### Phase 3: Flow-Level HITL
**Estimated effort: 5-6 days**

1. Update `Flow` class:
   - Add `flow_id` field
   - Add `_interrupt_handlers` with priorities
   - Implement `add_interrupt_handler()` with priority
   - Implement `_check_flow_interrupt()` with priority ordering
   - Implement `_create_checkpoint()`
   - Add checkpoint creation to interrupt raising
   
2. Implement `Flow.resume()`:
   - Checkpoint validation
   - Result restoration
   - Updated result merging
   - Resume execution from pending nodes
   - Full test coverage
   
3. Create `FlowTransitionEvent` enum:
   - Define all transition events
   - Document when each fires
   
4. Update `CompiledFlow`:
   - Pass interrupt handlers to engine
   - Support resume operations
   
5. Test comprehensive scenarios:
   - Interruption at various points
   - Resume after modification
   - Multiple interrupt/resume cycles
   - Checkpoint serialization/deserialization

### Phase 4: Stepper Engine HITL
**Estimated effort: 6-7 days**

1. Update `StepperEngine.__init__()`:
   - Add `_interrupt_handlers` tracking
   
2. Update `extract_result_from_stream(StepperEngine.astream()`:
   - Accept interrupt handlers parameter
   - Implement `_check_stepper_interrupt()`
   - Add interrupt checks at all key points:
     - Flow start
     - Frontier start/complete
     - Before/after routing
     - Flow completion
   - Implement `_create_stepper_checkpoint()`
   - Update exception handling
   
3. Implement `StepperEngine._route_next_with_interrupt()`:
   - Interrupt checks around routing decisions
   - Support for modified routing via interrupts
   
4. Implement `StepperEngine.resume()`:
   - Checkpoint validation for stepper mode
   - State restoration (iteration, frontier)
   - Resume execution logic
   - Full test coverage
   
5. Test loop scenarios:
   - Interruption during different iterations
   - Resume with modified state
   - Conditional edge interruption
   - Route modification via interrupt

### Phase 5: Integration & Testing
**Estimated effort: 3-4 days**

1. Comprehensive test suite:
   - Unit tests for all new components
   - Integration tests for full flows
   - Loop scenario tests
   - Priority ordering tests
   - Checkpoint serialization tests
   - Resume tests with various modifications
   
2. Performance testing:
   - Measure interrupt check overhead
   - Optimize hot paths if needed
   - Benchmark checkpoint creation
   
3. Error handling:
   - Handler failures don't crash flow
   - Timeout handling
   - Invalid checkpoint detection
   - Corrupted state recovery

### Phase 6: Documentation & Examples
**Estimated effort: 2-3 days**

1. Update API documentation:
   - All new classes and methods
   - HITL guide document
   - Migration guide for breaking changes
   
2. Create example scripts:
   - Simple approval workflow
   - Token limit interruption
   - Loop with periodic approval
   - Priority-based handlers
   - Resume after delay
   - Checkpoint persistence
   
3. Update README:
   - HITL feature overview
   - Quick start examples
   - Link to detailed guide
   
4. Create migration guide:
   - Breaking changes list
   - Code migration examples
   - Common patterns update

---

## Breaking Changes Summary

### API Changes
- **`ProgressItem`**: 
  - Added `interrupt_callback` field
  - Changed `frozen=False` (was `True`)
  - Added `arbitrary_types_allowed=True`
  - Added `set_interrupt_callback()` method
  - Added `check_interrupt()` method
  
- **`BaseNode.__init__`**: 
  - Added `_interrupt_handlers` list
  
- **`BaseNode`**: 
  - Added `add_interrupt_handler()` method
  - Added `_check_lifecycle_interrupt()` method
  - Modified `astream()` to check lifecycle interrupts
  - Modified `run()` to handle `InterruptionRequested`
  
- **`Flow.__init__`**: 
  - Added `flow_id` field
  - Added `_interrupt_handlers` list
  
- **`Flow`**: 
  - Added `add_interrupt_handler()` method
  - Added `_check_flow_interrupt()` method
  - Added `_create_checkpoint()` method
  - Added `resume()` method
  - Modified `run()` to check flow interrupts and create checkpoints
  
- **`extract_result_from_stream(StepperEngine.astream()`**: 
  - Added `interrupt_handlers` parameter
  
- **`StepperEngine`**:
  - Added multiple interrupt check methods
  - Added `resume()` method
  - Modified routing logic to support interrupts

### New Exceptions
- **`InterruptionRequested`**: Raised when HITL interruption occurs

### New Classes
- **`InterruptDecision`**: Return type for interrupt callbacks
- **`FlowCheckpoint`**: Serializable state snapshot for resumption
- **`HumanApprovalNode`**: Requires approval to proceed
- **`HumanReviewNode`**: Optional review/modification
- **`NodeLifecycleEvent`**: Enum for node lifecycle events
- **`FlowTransitionEvent`**: Enum for flow transition events
- **`HandlerPriority`**: Enum for handler priority levels
- **`InterruptHandlerRegistration`**: Handler registration with priority

### New Type Aliases
- **`InterruptCallback`**: `Callable[[ProgressItem], Awaitable[InterruptDecision]]`
- **`NodeInterruptHandler`**: `Callable[[BaseNode, NodeLifecycleEvent, Any], Awaitable[InterruptDecision]]`
- **`FlowInterruptHandler`**: `Callable[[Flow, FlowTransitionEvent, str, Any], Awaitable[InterruptDecision]]`
- **`HumanDecisionFunc[T]`**: `Callable[[T, dict[str, Any]], Awaitable[HumanDecision[T]]]`

### No Backwards Compatibility
- All changes are breaking
- No deprecation period
- No wrapper layers
- Clean, unified HITL system from ground up

---

## Design Rationale

### Why Not Separate Interrupt System?
- **Integration**: Interrupts are part of events, not external bolt-on
- **Simplicity**: One callback per event, not separate communication channels
- **Composability**: Callbacks can be chained/composed naturally
- **Consistency**: Same pattern at event/node/flow levels

### Why Three Interrupt Levels?
- **Event-Level**: Fine-grained (e.g., token limits, real-time monitoring)
- **Node-Level**: Lifecycle hooks (e.g., validation before/after execution)
- **Flow-Level**: Orchestration (e.g., audit trails, global policies)
- **Complementary**: Each level addresses different use cases without overlap

### Why `frozen=False` for ProgressItem?
- **Callback Attachment**: Need to attach callbacks after event creation
- **Excluded from Serialization**: Callbacks are excluded, so serialization unchanged
- **Practical Immutability**: Still immutable from user perspective (only internal callback mutation)
- **Better Alternative**: More practical than creating new event instances

### Why Async Callbacks Everywhere?
- **External Systems**: Callbacks may need to query databases, APIs
- **UI Interactions**: Callbacks may show UI and wait for user input
- **Framework Native**: Async is the native pattern for pydantic-flow
- **Future-Proof**: Supports any async operation without redesign

### Why Not Context Managers?
- **Flexibility**: Callbacks are more flexible than context managers
- **Dynamic**: Can be attached dynamically during streaming
- **Composition**: Don't require wrapping entire node/flow execution
- **Multiple**: Can have multiple callbacks at different levels

### Why Priority System?
- **Critical First**: Security/safety checks must always run
- **Ordering**: Business logic needs predictable execution order
- **Short-Circuit**: Optimization for non-critical handlers
- **Extensibility**: Users can integrate with existing priority schemes

### Why Checkpoint-Based Resumption?
- **Long-Running**: Workflows may wait hours/days for approval
- **Persistence**: Checkpoints can be stored in databases
- **Flexibility**: Can modify state before resuming
- **Reliability**: Full state capture prevents data loss

### Why Heartbeats During Approval?
- **Connection Alive**: Keep websockets/HTTP connections open
- **Progress Signal**: Show users that system is still waiting
- **Debugging**: Helps diagnose "stuck" approvals
- **Standard Pattern**: Common in long-running operations

---

## Conclusion

This comprehensive HITL architecture provides:

✅ **Event-level interruption** - Stop streaming operations mid-flight  
✅ **Node-level lifecycle hooks** - Validate at execution boundaries  
✅ **Flow-level orchestration** - Audit, policy, and transition control  
✅ **Priority-based execution** - Critical handlers always run  
✅ **Full resumption support** - Checkpoint and continue after hours/days  
✅ **Loop interruption** - Stop and resume iterative workflows  
✅ **Flexible metadata** - User-extensible context propagation  
✅ **Type-safe throughout** - Full Python 3.14+ type support  
✅ **Async-native** - Natural async/await integration  
✅ **Breaking changes** - Clean design without legacy cruft  

The design maintains pydantic-flow's core principles while adding powerful human-in-the-loop capabilities suitable for production AI workflows requiring human oversight, approval workflows, and long-running processes.

**Total estimated implementation time: 23-29 days**

**Next Steps**: Begin Phase 1 implementation of core infrastructure.

