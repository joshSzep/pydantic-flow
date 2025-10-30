"""Node mixins for composable functionality.

This module provides mixins that can be added to nodes to enable
additional capabilities like caching and interrupt handling.
"""

from __future__ import annotations

from typing import Any

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.hitl.decisions import InterruptCallback
from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.hitl.interrupts import HandlerPriority
from pydantic_flow.hitl.interrupts import InterruptHandlerRegistration
from pydantic_flow.streaming.events import ProgressItem


class InterruptibleNodeMixin:
    """Mixin providing interrupt handler support for nodes.

    Add this mixin to node classes that want to support HITL
    (Human-in-the-Loop) interruption patterns.

    Example:
        ```python
        class MyNode(InterruptibleNodeMixin, BaseNode[InputT, OutputT]):
            def __init__(self, ...):
                super().__init__(...)
                self._interrupt_handlers = []
        ```

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize interrupt handler storage.

        Args:
            *args: Positional arguments passed to super().__init__
            **kwargs: Keyword arguments passed to super().__init__

        """
        super().__init__(*args, **kwargs)
        self._interrupt_handlers: list[InterruptHandlerRegistration] = []

    def register_interrupt_handler(
        self,
        callback: InterruptCallback,
        priority: int = HandlerPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register an interrupt callback handler for this node.

        Handlers are invoked in priority order (lowest first) when
        checking for interrupts. Critical handlers (0-25) always execute.

        Args:
            callback: Async function that receives ProgressItem and returns
                InterruptDecision.
            priority: Priority level (0-100, lower executes first).
            metadata: Optional metadata about the handler.

        """
        registration = InterruptHandlerRegistration(
            callback=callback,
            priority=priority,
            metadata=metadata or {},
        )
        self._interrupt_handlers.append(registration)
        # Keep handlers sorted by priority
        self._interrupt_handlers.sort(key=lambda h: h.priority)

    def clear_interrupt_handlers(self) -> None:
        """Remove all registered interrupt handlers from this node."""
        self._interrupt_handlers.clear()

    async def _check_interrupt_handlers(self, item: ProgressItem) -> InterruptDecision:
        """Check all registered interrupt handlers for this progress item.

        Executes handlers in priority order. If any handler requests
        interruption, returns immediately with that decision.

        Args:
            item: The progress item to check.

        Returns:
            InterruptDecision indicating whether to interrupt.

        """
        for handler in self._interrupt_handlers:
            decision = await handler.callback(item)
            if decision.should_interrupt:
                return decision
        return InterruptDecision.proceed()


class CacheableNode:
    """Mixin for nodes that support caching.

    Add this mixin to node classes that want to opt into caching.
    The cache_policy attribute controls caching behavior.

    Example:
        ```python
        class MyNode(CacheableNode, BaseNode[InputT, OutputT]):
            def __init__(self, ..., cache_policy: CachePolicy | None = None):
                super().__init__(...)
                self.cache_policy = cache_policy
        ```

    """

    cache_policy: CachePolicy | None = None

    def is_cacheable(self) -> bool:
        """Check if this node has caching enabled.

        Returns:
            True if cache_policy is set and enabled.

        """
        return self.cache_policy is not None and self.cache_policy.enabled
