"""Event filtering for checkpoint traces.

Allows filtering specific event types or patterns from checkpoint traces
to reduce storage or exclude sensitive operations.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class EventFilterAction(str, Enum):
    """Action to take when filter matches.

    Attributes:
        EXCLUDE: Exclude event from trace entirely.
        REDACT: Include event but redact sensitive fields.
        ALLOW: Allow event (explicit allow rule).

    """

    EXCLUDE = "exclude"
    REDACT = "redact"
    ALLOW = "allow"


class EventFilterRule(BaseModel):
    """Rule for filtering events.

    Attributes:
        name: Rule name (for debugging).
        event_type_pattern: Regex pattern for event_type matching.
        tool_name_pattern: Regex pattern for tool_call event matching.
        action: What to do when rule matches.
        redact_fields: Fields to redact if action is REDACT.
        priority: Rule priority (higher = checked first).

    Example:
        >>> rule = EventFilterRule(
        ...     name="exclude_db_queries",
        ...     tool_name_pattern=".*database.*",
        ...     action=EventFilterAction.EXCLUDE,
        ... )

    """

    name: str
    event_type_pattern: str | None = None
    tool_name_pattern: str | None = None
    action: EventFilterAction = EventFilterAction.EXCLUDE
    redact_fields: list[str] = Field(default_factory=list)
    priority: int = 0


class EventFilter:
    """Filters events from checkpoint traces based on rules.

    Example:
        >>> filter_config = EventFilter(rules=[
        ...     EventFilterRule(
        ...         name="exclude_tool_calls",
        ...         event_type_pattern="tool_call",
        ...         action=EventFilterAction.EXCLUDE,
        ...     ),
        ...     EventFilterRule(
        ...         name="redact_api_keys",
        ...         tool_name_pattern=".*api.*",
        ...         action=EventFilterAction.REDACT,
        ...         redact_fields=["api_key", "auth_token"],
        ...     ),
        ... ])
        >>> filtered_trace = filter_config.filter_trace(trace)

    """

    def __init__(self, rules: list[EventFilterRule] | None = None):
        """Initialize event filter with rules.

        Args:
            rules: List of filtering rules (sorted by priority).

        """
        self.rules = sorted(rules or [], key=lambda r: r.priority, reverse=True)
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for performance."""
        import re

        self._compiled_rules: list[tuple[EventFilterRule, Any, Any]] = []
        for rule in self.rules:
            event_pattern = (
                re.compile(rule.event_type_pattern) if rule.event_type_pattern else None
            )
            tool_pattern = (
                re.compile(rule.tool_name_pattern) if rule.tool_name_pattern else None
            )
            self._compiled_rules.append((rule, event_pattern, tool_pattern))

    def filter_trace(self, trace: Any) -> Any:
        """Filter events from execution trace.

        Args:
            trace: ExecutionTrace to filter.

        Returns:
            Filtered ExecutionTrace.

        """
        import copy

        filtered_trace = copy.deepcopy(trace)

        # Filter node traces
        filtered_node_traces = []
        for node_trace in filtered_trace.node_traces:
            filtered_events = self.filter_events(node_trace.events)
            if filtered_events:  # Keep node trace if any events remain
                node_trace.events = filtered_events
                filtered_node_traces.append(node_trace)

        filtered_trace.node_traces = filtered_node_traces
        return filtered_trace

    def filter_events(self, events: list[Any]) -> list[Any]:
        """Filter list of events.

        Args:
            events: List of StoredEvent objects.

        Returns:
            Filtered list of events.

        """
        filtered = []
        for event in events:
            action = self._match_event(event)

            if action == EventFilterAction.ALLOW:
                filtered.append(event)
            elif action == EventFilterAction.EXCLUDE:
                continue  # Skip event
            elif action == EventFilterAction.REDACT:
                redacted_event = self._redact_event(event)
                filtered.append(redacted_event)

        return filtered

    def _match_event(self, event: Any) -> EventFilterAction:
        """Match event against rules.

        Args:
            event: StoredEvent to match.

        Returns:
            Action to take (default: ALLOW).

        """
        for rule, event_pattern, tool_pattern in self._compiled_rules:
            # Check event type pattern
            if event_pattern and event_pattern.match(event.event_type):
                return rule.action

            # Check tool name pattern (for tool_call events)
            if (
                tool_pattern
                and event.event_type == "tool_call"
                and "tool_name" in event.data
                and tool_pattern.match(event.data["tool_name"])
            ):
                return rule.action

        # Default: allow all events
        return EventFilterAction.ALLOW

    def _redact_event(self, event: Any) -> Any:
        """Redact sensitive fields from event.

        Args:
            event: StoredEvent to redact.

        Returns:
            Redacted event.

        """
        import copy

        redacted = copy.deepcopy(event)

        # Find matching rule for redaction fields
        for rule, event_pattern, tool_pattern in self._compiled_rules:
            if rule.action != EventFilterAction.REDACT:
                continue

            # Check if rule matches
            matches = False
            if event_pattern and event_pattern.match(event.event_type):
                matches = True
            if (
                tool_pattern
                and event.event_type == "tool_call"
                and "tool_name" in event.data
                and tool_pattern.match(event.data["tool_name"])
            ):
                matches = True

            if matches:
                # Redact specified fields
                for field in rule.redact_fields:
                    if field in redacted.data:
                        redacted.data[field] = "***REDACTED***"

        return redacted


class FilteredCheckpointBackend:
    """Wrapper backend that filters events before saving traces.

    Example:
        >>> from pydantic_flow.checkpoints import SQLiteCheckpointBackend
        >>> base_backend = SQLiteCheckpointBackend(...)
        >>> event_filter = EventFilter(rules=[...])
        >>> filtered_backend = FilteredCheckpointBackend(
        ...     backend=base_backend,
        ...     event_filter=event_filter,
        ... )

    """

    def __init__(self, backend: Any, event_filter: EventFilter):
        """Initialize filtered backend wrapper.

        Args:
            backend: Underlying storage backend.
            event_filter: Event filter configuration.

        """
        self._backend = backend
        self._filter = event_filter

    async def save_trace(self, trace: Any) -> None:
        """Save trace after filtering events.

        Args:
            trace: ExecutionTrace to filter and save.

        """
        filtered_trace = self._filter.filter_trace(trace)
        await self._backend.save_trace(filtered_trace)

    async def save_node_trace(self, node_trace: Any) -> None:
        """Save node trace after filtering events.

        Args:
            node_trace: NodeExecutionTrace to filter and save.

        """
        import copy

        filtered_node_trace = copy.deepcopy(node_trace)
        filtered_node_trace.events = self._filter.filter_events(node_trace.events)
        await self._backend.save_node_trace(filtered_node_trace)

    # Delegate all other methods to underlying backend
    def __getattr__(self, name: str) -> Any:
        """Delegate unknown methods to underlying backend."""
        return getattr(self._backend, name)
