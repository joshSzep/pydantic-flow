"""Helper functions for telemetry instrumentation."""

from contextlib import asynccontextmanager
from contextlib import contextmanager
import time
from typing import Any
from typing import ParamSpec
from typing import TypeVar

from opentelemetry import trace
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode

from pydantic_flow.telemetry.attributes import AttributeKey
from pydantic_flow.telemetry.attributes import Outcome
from pydantic_flow.telemetry.setup import get_active_span
from pydantic_flow.telemetry.setup import get_meter
from pydantic_flow.telemetry.setup import get_tracer
from pydantic_flow.telemetry.setup import is_enabled
from pydantic_flow.telemetry.setup import set_active_span

P = ParamSpec("P")
T = TypeVar("T")


@contextmanager
def create_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    set_as_active: bool = True,
):
    """Create a span with standard error handling.

    Args:
        name: Span name.
        attributes: Initial attributes.
        set_as_active: Whether to set this span as active in context.

    Yields:
        The created span.

    """
    if not is_enabled():
        yield None
        return

    tracer = get_tracer()
    parent_span = get_active_span()
    parent_context = trace.set_span_in_context(parent_span) if parent_span else None

    with tracer.start_as_current_span(
        name, attributes=attributes or {}, context=parent_context
    ) as span:
        token = None
        if set_as_active:
            token = set_active_span(span)

        try:
            yield span
            span.set_attribute(AttributeKey.OUTCOME, Outcome.SUCCESS)
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute(AttributeKey.OUTCOME, Outcome.ERROR)
            span.set_attribute(AttributeKey.ERROR_TYPE, type(e).__name__)
            span.set_attribute(AttributeKey.ERROR_MESSAGE, str(e))
            raise
        finally:
            if token is not None:
                set_active_span(None)


@asynccontextmanager
async def create_span_async(
    name: str,
    attributes: dict[str, Any] | None = None,
    set_as_active: bool = True,
):
    """Async version of create_span.

    Args:
        name: Span name.
        attributes: Initial attributes.
        set_as_active: Whether to set this span as active in context.

    Yields:
        The created span.

    """
    if not is_enabled():
        yield None
        return

    tracer = get_tracer()
    parent_span = get_active_span()
    parent_context = trace.set_span_in_context(parent_span) if parent_span else None

    with tracer.start_as_current_span(
        name, attributes=attributes or {}, context=parent_context
    ) as span:
        token = None
        if set_as_active:
            token = set_active_span(span)

        try:
            yield span
            span.set_attribute(AttributeKey.OUTCOME, Outcome.SUCCESS)
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute(AttributeKey.OUTCOME, Outcome.ERROR)
            span.set_attribute(AttributeKey.ERROR_TYPE, type(e).__name__)
            span.set_attribute(AttributeKey.ERROR_MESSAGE, str(e))
            raise
        finally:
            if token is not None:
                set_active_span(None)


def record_span_event(
    event_name: str,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Record an event on the active span.

    Args:
        event_name: Name of the event.
        attributes: Event attributes.

    """
    if not is_enabled():
        return

    span = get_active_span()
    if span is not None and span.is_recording():
        span.add_event(event_name, attributes=attributes or {})


def record_counter(
    metric_name: str,
    value: int = 1,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Record a counter increment.

    Args:
        metric_name: Metric name.
        value: Value to add (default 1).
        attributes: Metric attributes.

    """
    if not is_enabled():
        return

    meter = get_meter()
    counter = meter.create_counter(metric_name)
    counter.add(value, attributes=attributes or {})


def record_histogram(
    metric_name: str,
    value: float,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Record a histogram value.

    Args:
        metric_name: Metric name.
        value: Value to record.
        attributes: Metric attributes.

    """
    if not is_enabled():
        return

    meter = get_meter()
    histogram = meter.create_histogram(metric_name)
    histogram.record(value, attributes=attributes or {})


@contextmanager
def measure_duration(
    metric_name: str,
    attributes: dict[str, Any] | None = None,
):
    """Context manager to measure duration in milliseconds.

    Args:
        metric_name: Histogram metric name.
        attributes: Metric attributes.

    Yields:
        None

    """
    start = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start) * 1000
        record_histogram(metric_name, duration_ms, attributes)


@asynccontextmanager
async def measure_duration_async(
    metric_name: str,
    attributes: dict[str, Any] | None = None,
):
    """Async context manager to measure duration in milliseconds.

    Args:
        metric_name: Histogram metric name.
        attributes: Metric attributes.

    Yields:
        None

    """
    start = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start) * 1000
        record_histogram(metric_name, duration_ms, attributes)
