"""Telemetry configuration."""

import os

from pydantic import BaseModel
from pydantic import Field


class TelemetryConfig(BaseModel):
    """Configuration for OpenTelemetry integration.

    All settings support environment variable overrides using the PFLOW_TELEMETRY_*
    prefix.

    Attributes:
        enabled: Whether telemetry is enabled. Env: PFLOW_TELEMETRY_ENABLED
        service_name: Service name. Env: PFLOW_TELEMETRY_SERVICE_NAME
        otlp_endpoint: OTLP HTTP endpoint. Env: OTEL_EXPORTER_OTLP_ENDPOINT
        trace_sample_rate: Trace sampling 0.0-1.0.
            Env: PFLOW_TELEMETRY_SAMPLE_RATE
        export_to_console: Export to console if no endpoint.
            Env: PFLOW_TELEMETRY_CONSOLE
        export_interval_ms: Metric export interval.
            Env: PFLOW_TELEMETRY_EXPORT_INTERVAL_MS

    Example:
        ```python
        # Use defaults with env vars
        config = TelemetryConfig()

        # Or configure explicitly
        config = TelemetryConfig(
            service_name="my-agent",
            otlp_endpoint="http://localhost:4318",
            trace_sample_rate=0.1
        )
        ```

    """

    enabled: bool = Field(
        default_factory=lambda: os.getenv("PFLOW_TELEMETRY_ENABLED", "true").lower()
        in ("true", "1", "yes")
    )
    service_name: str = Field(
        default_factory=lambda: os.getenv(
            "PFLOW_TELEMETRY_SERVICE_NAME", "pydantic-flow"
        )
    )
    otlp_endpoint: str | None = Field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    trace_sample_rate: float = Field(
        default_factory=lambda: float(os.getenv("PFLOW_TELEMETRY_SAMPLE_RATE", "1.0")),
        ge=0.0,
        le=1.0,
    )
    export_to_console: bool = Field(
        default_factory=lambda: os.getenv("PFLOW_TELEMETRY_CONSOLE", "false").lower()
        in ("true", "1", "yes")
    )
    export_interval_ms: int = Field(
        default_factory=lambda: int(
            os.getenv("PFLOW_TELEMETRY_EXPORT_INTERVAL_MS", "5000")
        ),
        gt=0,
    )
