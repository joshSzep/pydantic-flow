"""OpenTelemetry integration for pydantic-flow.

This module provides tracing and metrics that mirror the Event → Node → Flow
architecture, making streaming behavior fully observable.

Example:
    ```python
    from pydantic_flow.telemetry import setup_telemetry

    # Simple setup - uses env vars or stdout
    setup_telemetry()

    # Or configure explicitly
    setup_telemetry(
        service_name="my-ai-app",
        otlp_endpoint="http://localhost:4318",
        trace_sample_rate=1.0
    )
    ```

"""

from pydantic_flow.telemetry.attributes import AttributeKey
from pydantic_flow.telemetry.attributes import EventName
from pydantic_flow.telemetry.attributes import MetricName
from pydantic_flow.telemetry.attributes import SpanKind
from pydantic_flow.telemetry.config import TelemetryConfig
from pydantic_flow.telemetry.helpers import traced_cache_lookup
from pydantic_flow.telemetry.helpers import traced_cache_write
from pydantic_flow.telemetry.helpers import traced_node_execution
from pydantic_flow.telemetry.setup import get_meter
from pydantic_flow.telemetry.setup import get_tracer
from pydantic_flow.telemetry.setup import is_enabled
from pydantic_flow.telemetry.setup import setup_telemetry

__all__ = [
    "AttributeKey",
    "EventName",
    "MetricName",
    "SpanKind",
    "TelemetryConfig",
    "get_meter",
    "get_tracer",
    "is_enabled",
    "setup_telemetry",
    "traced_cache_lookup",
    "traced_cache_write",
    "traced_node_execution",
]
