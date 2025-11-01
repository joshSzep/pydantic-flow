"""Tests for checkpoint telemetry and monitoring."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import patch

import pytest

from pydantic_flow.checkpoints.telemetry import DEGRADED_ERROR_RATE
from pydantic_flow.checkpoints.telemetry import UNHEALTHY_ERROR_RATE
from pydantic_flow.checkpoints.telemetry import CheckpointHealthStatus
from pydantic_flow.checkpoints.telemetry import CheckpointMetricsCollector
from pydantic_flow.checkpoints.telemetry import CheckpointOperation
from pydantic_flow.checkpoints.telemetry import CheckpointTelemetry


@pytest.fixture
def telemetry():
    """Create telemetry instance for testing."""
    return CheckpointTelemetry(backend_type="test")


def test_telemetry_initialization(telemetry):
    """Test telemetry initializes correctly."""
    assert telemetry.backend_type == "test"
    assert telemetry._total_operations == 0
    assert telemetry._failed_operations == 0


def test_record_successful_operation(telemetry):
    """Test recording successful operation."""
    start = telemetry.record_operation_start()
    telemetry.record_operation_end(
        CheckpointOperation.SAVE_SNAPSHOT, start, success=True
    )

    assert telemetry._total_operations == 1
    assert telemetry._failed_operations == 0
    assert telemetry._last_write_time is not None


def test_record_failed_operation(telemetry):
    """Test recording failed operation."""
    start = telemetry.record_operation_start()
    error = ValueError("test error")
    telemetry.record_operation_end(
        CheckpointOperation.SAVE_SNAPSHOT, start, success=False, error=error
    )

    assert telemetry._total_operations == 1
    assert telemetry._failed_operations == 1
    assert telemetry._last_error == "test error"


def test_health_check_healthy(telemetry):
    """Test health check reports healthy status."""
    # Record successful operations
    for _ in range(10):
        start = telemetry.record_operation_start()
        telemetry.record_operation_end(
            CheckpointOperation.SAVE_SNAPSHOT, start, success=True
        )

    health = telemetry.get_health()
    assert health.status == CheckpointHealthStatus.HEALTHY
    assert health.total_operations == 10
    assert health.failed_operations == 0
    assert health.is_healthy()


def test_health_check_degraded(telemetry):
    """Test health check reports degraded status."""
    # Record operations with some failures (15% failure rate)
    for i in range(20):
        start = telemetry.record_operation_start()
        success = i % 7 != 0  # Fail every 7th operation
        telemetry.record_operation_end(
            CheckpointOperation.SAVE_SNAPSHOT, start, success=success
        )

    health = telemetry.get_health()
    assert health.status == CheckpointHealthStatus.DEGRADED
    assert health.total_operations == 20
    assert health.failed_operations > DEGRADED_ERROR_RATE * 20


def test_health_check_unhealthy(telemetry):
    """Test health check reports unhealthy status."""
    # Record operations with high failure rate
    for i in range(10):
        start = telemetry.record_operation_start()
        success = i < 3  # Only first 3 succeed
        telemetry.record_operation_end(
            CheckpointOperation.SAVE_SNAPSHOT, start, success=success
        )

    health = telemetry.get_health()
    assert health.status == CheckpointHealthStatus.UNHEALTHY
    assert health.failed_operations >= UNHEALTHY_ERROR_RATE * 10


def test_health_check_stale_writes(telemetry):
    """Test health check detects stale writes."""
    # Record a write with old timestamp
    telemetry._last_write_time = datetime.now(UTC) - timedelta(minutes=10)
    telemetry._total_operations = 1

    health = telemetry.get_health()
    assert health.status == CheckpointHealthStatus.UNHEALTHY


def test_buffer_size_update(telemetry):
    """Test buffer size metric updates."""
    # Should not raise errors
    telemetry.update_buffer_size(100)
    telemetry.update_buffer_size(200)


def test_metrics_collector_sync_context():
    """Test metrics collector with synchronous context manager."""
    collector = CheckpointMetricsCollector(backend_type="test")

    with collector.track_operation(CheckpointOperation.SAVE_SNAPSHOT):
        pass  # Simulate operation

    health = collector.get_health()
    assert health.total_operations == 1
    assert health.failed_operations == 0


def test_metrics_collector_sync_context_with_error():
    """Test metrics collector handles errors in sync context."""
    collector = CheckpointMetricsCollector(backend_type="test")

    try:
        with collector.track_operation(CheckpointOperation.SAVE_SNAPSHOT):
            raise ValueError("test error")
    except ValueError:
        pass

    health = collector.get_health()
    assert health.total_operations == 1
    assert health.failed_operations == 1
    assert health.last_error == "test error"


@pytest.mark.asyncio
async def test_metrics_collector_async_context():
    """Test metrics collector with async context manager."""
    collector = CheckpointMetricsCollector(backend_type="test")

    async with collector.track_operation(CheckpointOperation.SAVE_SNAPSHOT):
        pass  # Simulate operation

    health = collector.get_health()
    assert health.total_operations == 1
    assert health.failed_operations == 0


@pytest.mark.asyncio
async def test_metrics_collector_async_context_with_error():
    """Test metrics collector handles errors in async context."""
    collector = CheckpointMetricsCollector(backend_type="test")

    try:
        async with collector.track_operation(CheckpointOperation.SAVE_SNAPSHOT):
            raise ValueError("test error")
    except ValueError:
        pass

    health = collector.get_health()
    assert health.total_operations == 1
    assert health.failed_operations == 1


def test_health_check_model(telemetry):
    """Test health check model properties."""
    health = telemetry.get_health()

    assert health.backend_type == "test"
    assert health.status in CheckpointHealthStatus
    assert health.total_operations >= 0
    assert health.failed_operations >= 0
    assert health.avg_latency_ms >= 0.0


def test_telemetry_disabled():
    """Test telemetry works when OpenTelemetry is disabled."""
    with patch("pydantic_flow.checkpoints.telemetry.is_enabled", return_value=False):
        telemetry = CheckpointTelemetry(backend_type="test")
        start = telemetry.record_operation_start()
        telemetry.record_operation_end(
            CheckpointOperation.SAVE_SNAPSHOT, start, success=True
        )

        # Should still track for health checks
        health = telemetry.get_health()
        assert health.total_operations == 1


def test_operation_enum_values():
    """Test checkpoint operation enum has expected values."""
    assert CheckpointOperation.SAVE_SNAPSHOT == "save_snapshot"
    assert CheckpointOperation.GET_SNAPSHOT == "get_snapshot"
    assert CheckpointOperation.SAVE_TRACE == "save_trace"
    assert CheckpointOperation.GET_TRACE == "get_trace"
    assert CheckpointOperation.DELETE_RUN == "delete_run"
    assert CheckpointOperation.LIST_RUNS == "list_runs"


def test_health_status_enum_values():
    """Test health status enum has expected values."""
    assert CheckpointHealthStatus.HEALTHY == "healthy"
    assert CheckpointHealthStatus.DEGRADED == "degraded"
    assert CheckpointHealthStatus.UNHEALTHY == "unhealthy"
