"""Tests for node caching capabilities."""

from collections.abc import AsyncIterator
from datetime import timedelta

from pydantic import BaseModel

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart


class SampleInput(BaseModel):
    """Sample input model for testing."""

    value: str


class SampleOutput(BaseModel):
    """Sample output model for testing."""

    result: str


class ConcreteTestNode(BaseNode[SampleInput, SampleOutput]):
    """Concrete test node with minimal astream implementation."""

    async def astream(self, input_data: SampleInput) -> AsyncIterator[ProgressItem]:
        """Minimal streaming implementation for testing."""
        yield StreamStart(run_id=self.run_id or "", node_id=self.name)
        yield StreamEnd(
            run_id=self.run_id or "",
            node_id=self.name,
            result=SampleOutput(result=input_data.value),
        )


def test_node_no_policy():
    """Test BaseNode with no policy set."""
    node = ConcreteTestNode(name="test")
    assert node.cache_policy is None
    assert node.is_cacheable() is False


def test_node_with_disabled_policy():
    """Test BaseNode with disabled policy."""
    policy = CachePolicy(enabled=False, ttl=timedelta(seconds=60))
    node = ConcreteTestNode(name="test", cache_policy=policy)
    assert node.cache_policy is not None
    assert node.is_cacheable() is False


def test_node_with_enabled_policy():
    """Test Node with enabled policy."""
    policy = CachePolicy(enabled=True, ttl=timedelta(seconds=60))
    node = ConcreteTestNode(name="test", cache_policy=policy)
    assert node.cache_policy is not None
    assert node.is_cacheable() is True


def test_node_interrupt_handlers():
    """Test Node interrupt handler registration."""
    node = ConcreteTestNode(name="test")

    # Initially no handlers
    assert node._interrupt_handlers == []

    # Register handlers
    from pydantic_flow.hitl.decisions import InterruptDecision
    from pydantic_flow.hitl.interrupts import HandlerPriority

    async def handler1(item: ProgressItem):
        return InterruptDecision(should_interrupt=False)

    async def handler2(item: ProgressItem):
        return InterruptDecision(should_interrupt=False)

    node.register_interrupt_handler(handler1, priority=HandlerPriority.HIGH)
    node.register_interrupt_handler(handler2, priority=HandlerPriority.LOW)

    assert len(node._interrupt_handlers) == 2

    # Clear handlers
    node.clear_interrupt_handlers()
    assert len(node._interrupt_handlers) == 0
