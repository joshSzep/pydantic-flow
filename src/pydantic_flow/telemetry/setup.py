"""Telemetry setup and initialization.

This module provides the main entry point for configuring OpenTelemetry
integration and accessing tracer/meter instances.
"""

from contextvars import ContextVar
from typing import Any

from opentelemetry import metrics
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio

from pydantic_flow.telemetry.config import TelemetryConfig

_config: TelemetryConfig | None = None
_tracer: trace.Tracer | None = None
_meter: metrics.Meter | None = None
_active_span: ContextVar[trace.Span | None] = ContextVar("_active_span", default=None)  # type: ignore[arg-type]


def setup_telemetry(
    *,
    enabled: bool | None = None,
    service_name: str | None = None,
    otlp_endpoint: str | None = None,
    trace_sample_rate: float | None = None,
    export_to_console: bool | None = None,
    export_interval_ms: int | None = None,
) -> TelemetryConfig:
    """Initialize OpenTelemetry tracing and metrics.

    This should be called once at application startup. If not called,
    telemetry will be disabled.

    Args:
        enabled: Enable telemetry. Defaults to env or True.
        service_name: Service name. Defaults to env or "pydantic-flow".
        otlp_endpoint: OTLP HTTP endpoint. Defaults to env or None.
        trace_sample_rate: Sampling rate 0.0-1.0. Defaults to env or 1.0.
        export_to_console: Export to console. Defaults to env or False.
        export_interval_ms: Export interval. Defaults to env or 5000.

    Returns:
        The active telemetry configuration.

    Example:
        ```python
        # Minimal setup - uses environment variables
        setup_telemetry()

        # Or configure explicitly
        setup_telemetry(
            service_name="my-agent",
            otlp_endpoint="http://localhost:4318",
            trace_sample_rate=0.1
        )
        ```

    """
    global _config, _tracer, _meter

    # Build config from args and env
    config_dict: dict[str, Any] = {}
    if enabled is not None:
        config_dict["enabled"] = enabled
    if service_name is not None:
        config_dict["service_name"] = service_name
    if otlp_endpoint is not None:
        config_dict["otlp_endpoint"] = otlp_endpoint
    if trace_sample_rate is not None:
        config_dict["trace_sample_rate"] = trace_sample_rate
    if export_to_console is not None:
        config_dict["export_to_console"] = export_to_console
    if export_interval_ms is not None:
        config_dict["export_interval_ms"] = export_interval_ms

    _config = TelemetryConfig(**config_dict)

    if not _config.enabled:
        return _config

    # Create resource with service name
    resource = Resource.create({"service.name": _config.service_name})

    # Setup tracing
    if _config.otlp_endpoint:
        span_exporter: Any = OTLPSpanExporter(
            endpoint=f"{_config.otlp_endpoint}/v1/traces"
        )
    elif _config.export_to_console:
        span_exporter = ConsoleSpanExporter()
    else:
        # No exporter - telemetry is a no-op
        _tracer = trace.get_tracer(__name__)
        _meter = metrics.get_meter(__name__)
        return _config

    sampler = ParentBasedTraceIdRatio(_config.trace_sample_rate)
    tracer_provider = TracerProvider(resource=resource, sampler=sampler)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer("pydantic_flow", "0.16.0")

    # Setup metrics
    if _config.otlp_endpoint:
        metric_exporter: Any = OTLPMetricExporter(
            endpoint=f"{_config.otlp_endpoint}/v1/metrics"
        )
    else:
        # Console only for traces, metrics would be too verbose
        # Just create a no-op meter
        _meter = metrics.get_meter(__name__)
        return _config

    metric_reader = PeriodicExportingMetricReader(
        metric_exporter, export_interval_millis=_config.export_interval_ms
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    _meter = metrics.get_meter("pydantic_flow", "0.16.0")

    return _config


def get_tracer() -> trace.Tracer:
    """Get the configured tracer instance.

    Returns:
        Tracer instance or a no-op tracer if telemetry is disabled.

    """
    if _tracer is None:
        return trace.get_tracer(__name__)
    return _tracer


def get_meter() -> metrics.Meter:
    """Get the configured meter instance.

    Returns:
        Meter instance or a no-op meter if telemetry is disabled.

    """
    if _meter is None:
        return metrics.get_meter(__name__)
    return _meter


def is_enabled() -> bool:
    """Check if telemetry is enabled.

    Returns:
        True if telemetry was configured and enabled.

    """
    return _config is not None and _config.enabled


def get_active_span() -> trace.Span | None:
    """Get the currently active span from context.

    Returns:
        Active span or None if no span is active.

    """
    return _active_span.get()


def set_active_span(span: trace.Span | None) -> Any:
    """Set the active span in context.

    Args:
        span: Span to set as active or None.

    Returns:
        Context token for resetting.

    """
    return _active_span.set(span)
