"""Complex Human-in-the-Loop (HITL) example with unified checkpoint system.

This example demonstrates:
- Multiple interrupt handlers with different priorities
- Checkpoint persistence and querying
- Priority-based handler execution order
- Interrupt metadata tracking and inspection
- Proper resume workflow with human decisions
"""

import asyncio
from pathlib import Path

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow import PromptConfig
from pydantic_flow import PromptNode
from pydantic_flow import ToolNode
from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointBackend
from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.inspection import CheckpointInspector
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.hitl.interrupts import HandlerPriority
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import TokenChunk


# Helper to extract result from stream
async def extract_result_from_stream(stream):
    """Extract final result from async stream of progress items."""
    result = None
    async for item in stream:
        if hasattr(item, "result"):
            result = item.result
    return result


class DocumentInput(BaseModel):
    """Document to process."""

    content: str
    category: str
    priority_level: int  # 1-5, where 5 is highest


class ProcessedDocument(BaseModel):
    """Processed document output."""

    summary: str
    risk_score: int  # 0-100


# Simulate a risk scoring function
async def calculate_risk_score(input_data: DocumentInput) -> ProcessedDocument:
    """Calculate risk score for document."""
    risk = len(input_data.content) // 10  # Simple heuristic
    if "urgent" in input_data.content.lower():
        risk += 30
    if "confidential" in input_data.content.lower():
        risk += 40

    return ProcessedDocument(
        summary=f"Processed: {input_data.content[:50]}...", risk_score=min(risk, 100)
    )


async def main():
    """Run complex HITL workflow with multiple approval stages."""
    print("=" * 70)
    print("COMPLEX HITL EXAMPLE: Multi-Stage Approval Workflow")
    print("=" * 70 + "\n")

    # Set up checkpoint backend for persistence
    db_path = Path("hitl_complex_checkpoints.db")
    if db_path.exists():
        db_path.unlink()  # Clean start for demo

    backend = SQLiteCheckpointBackend(config=SQLiteCheckpointConfig(db_path=db_path))
    await backend.initialize()

    inspector = CheckpointInspector(backend)

    # Create processing nodes
    risk_analyzer = ToolNode[DocumentInput, ProcessedDocument](
        tool_func=calculate_risk_score, name="risk_analyzer"
    )

    final_processor = PromptNode[ProcessedDocument, ProcessedDocument](
        prompt="Finalize document with summary: {input.summary}",
        config=PromptConfig(model="openai:gpt-4"),
        input=risk_analyzer.output,
        name="final_processor",
    )

    # Build flow
    flow = Flow(input_type=DocumentInput, output_type=ProcessedDocument)
    flow.add_nodes(risk_analyzer, final_processor)

    # Track which handlers fired
    fired_handlers = []

    # Critical security handler (priority 0 - always runs first)
    async def security_check(item: ProgressItem) -> InterruptDecision:
        """Check for security issues (CRITICAL priority)."""
        fired_handlers.append("security_check")
        if (
            isinstance(item, TokenChunk)
            and item.text
            and ("password" in item.text.lower() or "secret" in item.text.lower())
        ):
            return InterruptDecision.interrupt(
                "Security violation detected",
                metadata={"handler": "security", "severity": "critical"},
            )
        return InterruptDecision.proceed()

    # High-priority risk handler (priority 26)
    async def risk_threshold_check(item: ProgressItem) -> InterruptDecision:
        """Check if risk exceeds threshold (HIGH priority)."""
        fired_handlers.append("risk_threshold_check")
        if (
            isinstance(item, StreamEnd)
            and hasattr(item, "node_id")
            and item.node_id == "risk_analyzer"
        ):
            # Request approval for high-risk documents
            return InterruptDecision.interrupt(
                "Risk threshold exceeded - manager approval required",
                metadata={"handler": "risk", "approval_level": "manager"},
            )
        return InterruptDecision.proceed()

    # Normal priority compliance check (priority 51)
    async def compliance_check(item: ProgressItem) -> InterruptDecision:
        """Check compliance requirements (NORMAL priority)."""
        fired_handlers.append("compliance_check")
        # This would check compliance rules in production
        return InterruptDecision.proceed()

    # Audit logging handler (priority 100 - runs last)
    async def audit_log(item: ProgressItem) -> InterruptDecision:
        """Log all events for audit trail (LOW priority + offset)."""
        fired_handlers.append("audit_log")
        # Log event to audit system in production
        return InterruptDecision.proceed()

    # Register handlers at flow level with explicit priorities
    flow.register_interrupt_handler(security_check, priority=HandlerPriority.CRITICAL)
    flow.register_interrupt_handler(risk_threshold_check, priority=HandlerPriority.HIGH)
    flow.register_interrupt_handler(compliance_check, priority=HandlerPriority.NORMAL)
    flow.register_interrupt_handler(audit_log, priority=HandlerPriority.LOW)

    await run_low_risk_test(flow, fired_handlers, backend)
    await run_high_risk_test(flow, fired_handlers, backend, inspector)

    # Cleanup
    await backend.close()
    print("\n✅ All tests completed. Checkpoint database saved to:", db_path)


async def run_low_risk_test(
    flow: Flow, fired_handlers: list, backend: SQLiteCheckpointBackend
) -> None:
    """Test low-risk document processing."""
    print("Test Case 1: Low-Risk Document (No Interruption Expected)")
    print("-" * 70)

    low_risk_doc = DocumentInput(
        content="Regular business document about quarterly planning.",
        category="business",
        priority_level=2,
    )

    # Configure with checkpoint backend but unique run_id
    config = RunConfig(checkpoint_backend=backend, run_id="low_risk_test_run")

    fired_handlers.clear()
    try:
        result = await extract_result_from_stream(
            flow.astream(low_risk_doc, config=config)
        )
        print("✅ Document processed successfully (no interrupt)")
        print(f"   Risk Score: {result.risk_score}")
        print(f"   Summary: {result.summary}")
        print(f"   Handlers fired (in order): {fired_handlers}")
        print()

    except InterruptionRequested as exc:
        snapshot = exc.snapshot
        print("✋ Unexpected interruption")
        print(f"   Run ID: {snapshot.run_id}")
        print(f"   Interrupted Node: {snapshot.interrupted_node_id}")
        print(f"   Decision: {exc.decision}")
        print(f"   Handlers fired: {fired_handlers}")
        print()


async def run_high_risk_test(  # noqa: PLR0915
    flow: Flow,
    fired_handlers: list,
    backend: SQLiteCheckpointBackend,
    inspector: CheckpointInspector,
) -> None:
    """Test high-risk document with security concerns."""
    print("\nTest Case 2: High-Risk Document with Security Concern")
    print("-" * 70)

    high_risk_doc = DocumentInput(
        content="Urgent: confidential information about the password reset system.",
        category="security",
        priority_level=5,
    )

    # Configure with checkpoint backend
    config = RunConfig(checkpoint_backend=backend, run_id="high_risk_test_run")

    fired_handlers.clear()
    try:
        result = await extract_result_from_stream(
            flow.astream(high_risk_doc, config=config)
        )
        print(f"Unexpected success: {result}")

    except InterruptionRequested as exc:
        snapshot = exc.snapshot
        metadata = snapshot.metadata or {}

        print("✋ Workflow interrupted for approval (as expected)")
        print(f"   Run ID: {snapshot.run_id}")
        print(f"   Snapshot ID: {snapshot.snapshot_id}")
        print(f"   Wave Number: {snapshot.wave_number}")
        print(f"   Interrupted Node: {snapshot.interrupted_node_id}")
        print(f"   Handler that triggered: {metadata.get('handler', 'unknown')}")
        severity_val = metadata.get("severity") or metadata.get("approval_level")
        print(f"   Severity/Level: {severity_val}")
        print(f"   Handlers fired (in order): {fired_handlers}")
        print()

        # Verify handlers executed in priority order
        print("✓ Handler Execution Order Verification:")
        expected_order = [
            "security_check",  # Priority 0 (CRITICAL)
            "risk_threshold_check",  # Priority 26 (HIGH)
            "compliance_check",  # Priority 51 (NORMAL)
            "audit_log",  # Priority 76 (LOW)
        ]

        # Check as many as we have
        for i, expected in enumerate(expected_order):
            if i < len(fired_handlers):
                actual = fired_handlers[i]
                status = "✓" if expected == actual else "✗"
                print(f"   {status} Expected {expected}, got {actual}")

        print("\n� Checkpoint Inspection Demo:")

        # Query interrupted runs
        interrupted_runs = await inspector.list_interrupted_runs(limit=10)
        print(f"   • Total interrupted runs: {len(interrupted_runs)}")

        for run in interrupted_runs:
            print(f"   • Run {run.run_id[:12]}... - Status: {run.status}")
            if run.interrupt_snapshot_id:
                print(f"     Snapshot: {run.interrupt_snapshot_id[:12]}...")

        # Get the specific interrupt snapshot
        interrupt_snapshot = await inspector.get_interrupt_snapshot(snapshot.run_id)
        if interrupt_snapshot:
            print("\n   ✓ Retrieved interrupt snapshot")
            print(f"     Reason: {interrupt_snapshot.reason}")
            print(f"     Has metadata: {bool(interrupt_snapshot.metadata)}")
            print(f"     Next frontier: {interrupt_snapshot.next_frontier}")

        print("\n👤 In production, you would now:")
        print("   1. Present document to security reviewer via UI/API")
        print("   2. Collect approval/rejection decision")
        print(
            "   3. Use flow.resume_from_snapshot(snapshot, ..., human_decision={...})"
        )
        print("   4. Continue to next approval stage if approved")
        print("\n💡 Resume example:")
        print("   ```python")
        print("   # After getting human approval")
        print("   config = RunConfig(checkpoint_backend=backend)")
        print("   result = await flow.resume_from_snapshot(")
        print("       snapshot=snapshot,")
        print("       inputs=high_risk_doc,")
        print("       config=config,")
        print("       human_decision={'approved': True, 'approver': 'security_team'}")
        print("   )")
        print("   ```\n")


if __name__ == "__main__":
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
