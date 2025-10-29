"""Telemetry wrappers for checkpoint stores.

This module provides decorators and wrappers to add observability
to checkpoint operations.
"""

from collections.abc import Awaitable
from collections.abc import Callable
from functools import wraps
from typing import Any
from typing import ParamSpec
from typing import TypeVar

from pydantic_flow.telemetry.attributes import AttributeKey
from pydantic_flow.telemetry.attributes import MetricName
from pydantic_flow.telemetry.attributes import SpanKind
from pydantic_flow.telemetry.helpers import create_span_async
from pydantic_flow.telemetry.helpers import measure_duration_async
from pydantic_flow.telemetry.helpers import record_counter

P = ParamSpec("P")
T = TypeVar("T")


def instrument_checkpoint_save(
    backend_name: str,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Add telemetry to checkpoint save operations.

    Args:
        backend_name: Name of the checkpoint backend.

    Returns:
        Decorator function.

    """

    def decorator(
        func: Callable[P, Awaitable[T]],
    ) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            attrs: dict[str, Any] = {
                str(AttributeKey.CHECKPOINT_BACKEND): backend_name,
            }
            if args and hasattr(args[0], "id"):
                attrs[str(AttributeKey.CHECKPOINT_ID)] = str(args[0].id)  # pyright: ignore[reportAttributeAccessIssue]

            record_counter(MetricName.CHECKPOINT_WRITES, attributes=attrs)

            async with (
                create_span_async(SpanKind.CHECKPOINT_WRITE, attributes=attrs),
                measure_duration_async(
                    MetricName.CHECKPOINT_WRITE_DURATION, attributes=attrs
                ),
            ):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def instrument_checkpoint_get(
    backend_name: str,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Add telemetry to checkpoint get operations.

    Args:
        backend_name: Name of the checkpoint backend.

    Returns:
        Decorator function.

    """

    def decorator(
        func: Callable[P, Awaitable[T]],
    ) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            attrs: dict[str, Any] = {
                str(AttributeKey.CHECKPOINT_BACKEND): backend_name,
            }

            record_counter(MetricName.CHECKPOINT_READS, attributes=attrs)

            async with (
                create_span_async(SpanKind.CHECKPOINT_READ, attributes=attrs),
                measure_duration_async(
                    MetricName.CHECKPOINT_READ_DURATION, attributes=attrs
                ),
            ):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
