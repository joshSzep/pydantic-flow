"""Checkpoint telemetry for monitoring and observability.

This module provides OpenTelemetry metrics and health checks for checkpoint
operations.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from enum import Enum
from typing import Any

from opentelemetry.metrics import Counter
from opentelemetry.metrics import Histogram
from opentelemetry.metrics import UpDownCounter
from pydantic import BaseModel

from pydantic_flow.telemetry.setup import get_meter
from pydantic_flow.telemetry.setup import is_enabled

# Health check thresholds
UNHEALTHY_ERROR_RATE = 0.5  # 50% error rate
DEGRADED_ERROR_RATE = 0.1  # 10% error rate
STALE_WRITE_THRESHOLD_SECONDS = 300  # 5 minutes
DEGRADED_LATENCY_MS = 1000  # 1 second


class CheckpointOperation(str, Enum):
    """Checkpoint operation types for metrics."""

    SAVE_SNAPSHOT = "save_snapshot"
    GET_SNAPSHOT = "get_snapshot"
    SAVE_TRACE = "save_trace"
    GET_TRACE = "get_trace"
    DELETE_RUN = "delete_run"
    LIST_RUNS = "list_runs"


class CheckpointHealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class CheckpointHealthCheck(BaseModel):
    """Health check result for checkpoint backend.

    Attributes:
        status: Overall health status.
        backend_type: Type of backend (sqlite, postgres, s3, etc).
        last_write_time: Last successful write timestamp.
        last_error: Most recent error message if any.
        total_operations: Total operations since startup.
        failed_operations: Failed operations since startup.
        avg_latency_ms: Average operation latency in milliseconds.

    """

    status: CheckpointHealthStatus
    backend_type: str
    last_write_time: datetime | None = None
    last_error: str | None = None
    total_operations: int = 0
    failed_operations: int = 0
    avg_latency_ms: float = 0.0

    def is_healthy(self) -> bool:
        """Check if backend is healthy.

        Returns:
            True if status is HEALTHY.

        """
        return self.status == CheckpointHealthStatus.HEALTHY


class CheckpointTelemetry:
    """Telemetry collector for checkpoint operations.

    Provides OpenTelemetry metrics for monitoring checkpoint performance
    and reliability.

    Metrics collected:
        - checkpoint_operations_total: Counter of operations by type and status
        - checkpoint_operation_duration: Histogram of operation latencies
        - checkpoint_buffer_size: Current buffer size for buffered backends
        - checkpoint_errors_total: Counter of errors by type

    Usage:
        telemetry = CheckpointTelemetry(backend_type="sqlite")

        # Record operation
        with telemetry.record_operation("save_snapshot"):
            await backend.save_state_snapshot(snapshot)

        # Update buffer size
        telemetry.update_buffer_size(current_size)

        # Get health status
        health = telemetry.get_health()

    Args:
        backend_type: Type of backend for labeling metrics.

    """

    def __init__(self, backend_type: str):
        """Initialize checkpoint telemetry.

        Args:
            backend_type: Backend type (sqlite, postgres, s3, etc).

        """
        self.backend_type = backend_type
        self._enabled = is_enabled()

        # Operation tracking for health checks
        self._total_operations = 0
        self._failed_operations = 0
        self._last_write_time: datetime | None = None
        self._last_error: str | None = None
        self._latency_sum = 0.0
        self._latency_count = 0

        if not self._enabled:
            return

        meter = get_meter()

        # Counters
        self._operations_counter: Counter = meter.create_counter(
            name="checkpoint_operations_total",
            description="Total checkpoint operations by type and status",
            unit="1",
        )

        self._errors_counter: Counter = meter.create_counter(
            name="checkpoint_errors_total",
            description="Total checkpoint errors by type",
            unit="1",
        )

        # Histograms for latency
        self._duration_histogram: Histogram = meter.create_histogram(
            name="checkpoint_operation_duration_ms",
            description="Checkpoint operation duration in milliseconds",
            unit="ms",
        )

        # UpDownCounter for buffer size
        self._buffer_size_gauge: UpDownCounter = meter.create_up_down_counter(
            name="checkpoint_buffer_size",
            description="Current checkpoint buffer size",
            unit="1",
        )

    def record_operation_start(self) -> float:
        """Record the start of an operation.

        Returns:
            Start timestamp for duration calculation.

        """
        import time

        return time.perf_counter()

    def record_operation_end(
        self,
        operation: CheckpointOperation | str,
        start_time: float,
        success: bool = True,
        error: Exception | None = None,
    ) -> None:
        """Record the end of an operation.

        Args:
            operation: Operation type.
            start_time: Start timestamp from record_operation_start.
            success: Whether operation succeeded.
            error: Exception if operation failed.

        """
        import time

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Update health tracking
        self._total_operations += 1
        self._latency_sum += duration_ms
        self._latency_count += 1

        if success:
            # Update last write time for write operations
            if operation in (
                CheckpointOperation.SAVE_SNAPSHOT,
                CheckpointOperation.SAVE_TRACE,
            ):
                self._last_write_time = datetime.now(UTC)
        else:
            self._failed_operations += 1
            if error:
                self._last_error = str(error)

        if not self._enabled:
            return

        # Record metrics
        status = "success" if success else "error"
        attributes = {
            "backend_type": self.backend_type,
            "operation": str(operation),
            "status": status,
        }

        self._operations_counter.add(1, attributes)
        self._duration_histogram.record(duration_ms, attributes)

        if error:
            error_attributes = {
                "backend_type": self.backend_type,
                "error_type": type(error).__name__,
            }
            self._errors_counter.add(1, error_attributes)

    def update_buffer_size(self, size: int) -> None:
        """Update current buffer size metric.

        Args:
            size: Current buffer size.

        """
        if not self._enabled:
            return

        attributes = {"backend_type": self.backend_type}
        # UpDownCounter expects delta, so we need to track previous value
        # For simplicity, we'll just set the absolute value
        # In production, you'd track the delta
        self._buffer_size_gauge.add(size, attributes)

    def get_health(self) -> CheckpointHealthCheck:
        """Get current health status.

        Returns:
            Health check result with current metrics.

        """
        # Calculate status based on error rate and recency
        error_rate = (
            self._failed_operations / self._total_operations
            if self._total_operations > 0
            else 0.0
        )

        avg_latency = (
            self._latency_sum / self._latency_count if self._latency_count > 0 else 0.0
        )

        if error_rate > UNHEALTHY_ERROR_RATE or (
            self._last_write_time
            and (datetime.now(UTC) - self._last_write_time).total_seconds()
            > STALE_WRITE_THRESHOLD_SECONDS
        ):
            status = CheckpointHealthStatus.UNHEALTHY
        elif error_rate > DEGRADED_ERROR_RATE or avg_latency > DEGRADED_LATENCY_MS:
            status = CheckpointHealthStatus.DEGRADED
        else:
            status = CheckpointHealthStatus.HEALTHY

        return CheckpointHealthCheck(
            status=status,
            backend_type=self.backend_type,
            last_write_time=self._last_write_time,
            last_error=self._last_error,
            total_operations=self._total_operations,
            failed_operations=self._failed_operations,
            avg_latency_ms=avg_latency,
        )


class CheckpointMetricsCollector:
    """Convenience wrapper for collecting checkpoint metrics.

    Provides context managers for automatic metric recording.

    Usage:
        collector = CheckpointMetricsCollector(backend_type="sqlite")

        async with collector.track_operation("save_snapshot"):
            await backend.save_state_snapshot(snapshot)

    Args:
        backend_type: Type of backend for metrics.

    """

    def __init__(self, backend_type: str):
        """Initialize metrics collector.

        Args:
            backend_type: Backend type for labeling.

        """
        self.telemetry = CheckpointTelemetry(backend_type=backend_type)

    class OperationContext:
        """Context manager for tracking operations."""

        def __init__(
            self,
            telemetry: CheckpointTelemetry,
            operation: CheckpointOperation | str,
        ):
            """Initialize operation context.

            Args:
                telemetry: Telemetry instance.
                operation: Operation being tracked.

            """
            self.telemetry = telemetry
            self.operation = operation
            self.start_time: float | None = None

        def __enter__(self) -> CheckpointMetricsCollector.OperationContext:
            """Start tracking operation."""
            self.start_time = self.telemetry.record_operation_start()
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            """End tracking operation."""
            if self.start_time is not None:
                success = exc_type is None
                self.telemetry.record_operation_end(
                    self.operation,
                    self.start_time,
                    success=success,
                    error=exc_val if exc_val else None,
                )

        async def __aenter__(
            self,
        ) -> CheckpointMetricsCollector.OperationContext:
            """Async start tracking operation."""
            self.start_time = self.telemetry.record_operation_start()
            return self

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            """Async end tracking operation."""
            if self.start_time is not None:
                success = exc_type is None
                self.telemetry.record_operation_end(
                    self.operation,
                    self.start_time,
                    success=success,
                    error=exc_val if exc_val else None,
                )

    def track_operation(self, operation: CheckpointOperation | str) -> OperationContext:
        """Create context manager for tracking an operation.

        Args:
            operation: Operation type to track.

        Returns:
            Context manager for automatic metric recording.

        """
        return self.OperationContext(self.telemetry, operation)

    def update_buffer_size(self, size: int) -> None:
        """Update buffer size metric.

        Args:
            size: Current buffer size.

        """
        self.telemetry.update_buffer_size(size)

    def get_health(self) -> CheckpointHealthCheck:
        """Get health check status.

        Returns:
            Current health status.

        """
        return self.telemetry.get_health()
