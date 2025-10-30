"""Complex Human-in-the-Loop (HITL) example.

This example demonstrates:
- Multiple interrupt handlers with different priorities
- Conditional routing with approval gates
- Priority-based handler execution
- Checkpoint metadata tracking
"""

import asyncio

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow import PromptConfig
from pydantic_flow import PromptNode
from pydantic_flow import ToolNode
from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.hitl.interrupts import HandlerPriority
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import TokenChunk


class DocumentInput(BaseModel):
    """Document to process."""

    content: str
    category: str
    priority_level: int  # 1-5, where 5 is highest


class ProcessedDocument(BaseModel):
    """Processed document output."""

    summary: str
    risk_score: int  # 0-100


class ApprovalRecord(BaseModel):
    """Record of approvals."""

    approved: bool
    approver: str
    reason: str


# Simulate a risk scoring function
def calculate_risk_score(input_data: DocumentInput) -> ProcessedDocument:
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
            # In real scenario, would check risk score from node state
            # For demo, always request review for high-risk items
            return InterruptDecision.interrupt(
                "Risk threshold exceeded - manager approval required",
                metadata={"handler": "risk", "approval_level": "manager"},
            )
        return InterruptDecision.proceed()

    # Normal priority compliance check (priority 51)
    async def compliance_check(item: ProgressItem) -> InterruptDecision:
        """Check compliance requirements (NORMAL priority)."""
        fired_handlers.append("compliance_check")
        # This would check compliance rules
        return InterruptDecision.proceed()

    # Audit logging handler (priority 100 - runs last)
    async def audit_log(item: ProgressItem) -> InterruptDecision:
        """Log all events for audit trail (LOW priority + offset)."""
        fired_handlers.append("audit_log")
        # Log event to audit system
        return InterruptDecision.proceed()

    # Register handlers at flow level with explicit priorities
    flow.register_interrupt_handler(security_check, priority=HandlerPriority.CRITICAL)
    flow.register_interrupt_handler(risk_threshold_check, priority=HandlerPriority.HIGH)
    flow.register_interrupt_handler(compliance_check, priority=HandlerPriority.NORMAL)
    flow.register_interrupt_handler(audit_log, priority=HandlerPriority.LOW)

    await run_low_risk_test(flow, fired_handlers)
    await run_high_risk_test(flow, fired_handlers)


async def run_low_risk_test(flow: Flow, fired_handlers: list) -> None:
    """Test low-risk document processing."""
    print("Test Case 1: Low-Risk Document")
    print("-" * 70)

    low_risk_doc = DocumentInput(
        content="Regular business document about quarterly planning.",
        category="business",
        priority_level=2,
    )

    fired_handlers.clear()
    try:
        result = await flow.run(low_risk_doc)
        print("✅ Document processed successfully")
        print(f"   Risk Score: {result.risk_score}")
        print(f"   Summary: {result.summary}")
        print(f"   Handlers fired (in order): {fired_handlers}")
        print()

    except InterruptionRequested as exc:
        checkpoint = exc.checkpoint
        print("✋ Workflow interrupted")
        print(f"   Reason: {checkpoint.metadata.get('handler', 'Unknown')}")
        print(f"   Handlers fired (in order): {fired_handlers}")
        print(f"   Checkpoint: {checkpoint.interrupted_node_id}")
        print()


async def run_high_risk_test(flow: Flow, fired_handlers: list) -> None:
    """Test high-risk document with security concerns."""
    print("\nTest Case 2: High-Risk Document with Security Concern")
    print("-" * 70)

    high_risk_doc = DocumentInput(
        content="Urgent: confidential information about the password reset system.",
        category="security",
        priority_level=5,
    )

    fired_handlers.clear()
    try:
        result = await flow.run(high_risk_doc)
        print(f"Unexpected success: {result}")

    except InterruptionRequested as exc:
        checkpoint = exc.checkpoint
        metadata = checkpoint.metadata

        print("✋ Workflow interrupted for approval")
        print(f"   Flow ID: {checkpoint.flow_id}")
        print(f"   Interrupted Node: {checkpoint.interrupted_node_id}")
        print(f"   Handler that triggered: {metadata.get('handler')}")
        severity_val = metadata.get("severity", metadata.get("approval_level"))
        print(f"   Severity: {severity_val}")
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

        for expected, actual in zip(expected_order, fired_handlers, strict=True):
            status = "✓" if expected == actual else "✗"
            print(f"   {status} Expected {expected}, got {actual}")

        print("\n👤 In production, this would:")
        print("   1. Present document to security reviewer")
        print("   2. Collect approval/rejection decision")
        print("   3. Resume workflow with decision")
        print("   4. Continue to next approval stage if approved\n")


if __name__ == "__main__":
    asyncio.run(main())
