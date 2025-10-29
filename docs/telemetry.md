# OpenTelemetry Integration

## Overview

pydantic-flow provides first-class OpenTelemetry integration that mirrors the framework's Event → Node → Flow architecture. Telemetry is **streaming-native** and captures the same information developers see during execution.

## Quick Start

### Minimal Setup

```python
from pydantic_flow.telemetry import setup_telemetry

# Uses environment variables or sensible defaults
setup_telemetry()
```

### Configuration

```python
# Console output for development
setup_telemetry(
    service_name="my-agent",
    export_to_console=True,
    trace_sample_rate=1.0
)

# OTLP endpoint for production
setup_telemetry(
    service_name="my-agent",
    otlp_endpoint="http://localhost:4318",
    trace_sample_rate=0.1  # Sample 10% of traces
)
```

### Environment Variables

All settings support environment variable overrides:

- `PFLOW_TELEMETRY_ENABLED` - Enable/disable telemetry (default: true)
- `PFLOW_TELEMETRY_SERVICE_NAME` - Service name (default: "pydantic-flow")
- `OTEL_EXPORTER_OTLP_ENDPOINT` - OTLP endpoint URL
- `PFLOW_TELEMETRY_SAMPLE_RATE` - Trace sampling rate 0.0-1.0 (default: 1.0)
- `PFLOW_TELEMETRY_CONSOLE` - Export to console (default: false)
- `PFLOW_TELEMETRY_EXPORT_INTERVAL_MS` - Metric export interval (default: 5000)

## Architecture

### Span Hierarchy

Telemetry creates a natural hierarchy that maps to execution:

```
FlowRun (root span)
├── NodeRun (research node)
│   ├── CacheLookup
│   ├── ToolCall (via events)
│   └── CacheWrite
├── NodeRun (summary node)
│   ├── CacheLookup
│   ├── AgentCall
│   │   └── Stream events (start, chunks, end)
│   └── CacheWrite
```

### Span Types

- **FlowRun**: Root span for entire flow execution
- **NodeRun**: Span for each node execution attempt
- **CacheLookup**: Cache get operation
- **CacheWrite**: Cache set operation
- **CheckpointRead**: Loading checkpoint from store
- **CheckpointWrite**: Saving checkpoint to store
- **MemoryCompress**: Memory compression operation
- **AgentCall**: LLM/agent invocation (when applicable)
- **HumanGate**: HITL pause/resume

### Stream Events

Stream events are recorded as span events on the active NodeRun:

- `stream.start` - Node execution begins
- `stream.chunk` - Token/text chunk received
- `stream.end` - Node execution completes
- `cache.hit` / `cache.miss` / `cache.write` - Cache operations
- `tool.call` / `tool.result` - Tool invocations
- `human.requested` - HITL interruption
- `memory.compress.pending` / `complete` - Memory compression

## Attributes

### Flow Attributes

- `pflow.flow.id` - Unique flow identifier
- `pflow.run.id` - Execution run identifier
- `pflow.execution.mode` - "dag" or "stepper"

### Node Attributes

- `pflow.node.id` - Node identifier
- `pflow.node.name` - Node name
- `pflow.node.type` - Node class name
- `pflow.loop.iteration` - Iteration number (for loops)
- `pflow.retry.number` - Retry attempt number

### Cache Attributes

- `pflow.cache.backend` - Cache backend class name
- `pflow.cache.key_hash` - Truncated cache key hash
- `pflow.cache.ttl_remaining` - TTL seconds remaining
- `cache.result` - "hit" or "miss"

### Checkpoint Attributes

- `pflow.checkpoint.backend` - Checkpoint store class name
- `pflow.checkpoint.id` - Checkpoint identifier
- `pflow.checkpoint.interrupted` - Whether HITL interrupted

### Status Attributes

- `outcome` - "success", "error", or "cancelled"
- `error.type` - Exception class name
- `error.message` - Error message

## Metrics

### Counters

- `pflow.flow.runs` - Total flow executions
- `pflow.node.executions` - Total node executions
- `pflow.errors` - Total errors
- `pflow.cache.lookups` - Cache lookups
- `pflow.cache.hits` - Cache hits
- `pflow.cache.misses` - Cache misses
- `pflow.cache.writes` - Cache writes
- `pflow.checkpoint.reads` - Checkpoint reads
- `pflow.checkpoint.writes` - Checkpoint writes
- `pflow.hitl.requests` - HITL interruptions
- `pflow.hitl.responses` - HITL responses

### Histograms

- `pflow.flow.duration.ms` - Flow execution duration
- `pflow.node.duration.ms` - Node execution duration
- `pflow.cache.lookup.duration.ms` - Cache lookup latency
- `pflow.cache.write.duration.ms` - Cache write latency
- `pflow.checkpoint.read.duration.ms` - Checkpoint read latency
- `pflow.checkpoint.write.duration.ms` - Checkpoint write latency
- `pflow.memory.compress.duration.ms` - Memory compression duration

## Deployment Patterns

### Development

```python
# Console output - see traces inline
setup_telemetry(export_to_console=True)
```

### Local OTLP Collector

```bash
# Start Jaeger all-in-one
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

```python
setup_telemetry(otlp_endpoint="http://localhost:4318")
```

View traces at http://localhost:16686

### Production

```python
# Point to production OTLP collector
setup_telemetry(
    service_name="production-agent",
    otlp_endpoint="https://otel-collector.example.com",
    trace_sample_rate=0.1  # Sample 10%
)
```

### Kubernetes

```yaml
env:
  - name: OTEL_EXPORTER_OTLP_ENDPOINT
    value: "http://otel-collector:4318"
  - name: PFLOW_TELEMETRY_SAMPLE_RATE
    value: "0.1"
```

## Performance

### Overhead

When disabled, telemetry has **near-zero overhead** - all imports are local and checks short-circuit immediately.

```python
# Telemetry completely disabled
setup_telemetry(enabled=False)
```

### Sampling

Use sampling in production to reduce overhead:

```python
setup_telemetry(trace_sample_rate=0.1)  # Sample 10% of traces
```

ParentBased sampling ensures child spans follow parent sampling decision.

### Cardinality

Metric attributes use **short, stable keys** to minimize cardinality:

- ✅ `node_type=PromptNode` - Low cardinality
- ❌ `node_id=prompt_abc123xyz` - High cardinality

Use exemplars to link metrics to specific traces instead of high-cardinality labels.

## Querying

### Find Slow Flows

```
# TraceQL (Tempo/Jaeger)
{ name="flow_run" } | duration > 5s

# PromQL (Prometheus)
histogram_quantile(0.95, pflow_flow_duration_ms_bucket)
```

### Cache Hit Rate

```
# PromQL
rate(pflow_cache_hits[5m]) / rate(pflow_cache_lookups[5m])
```

### Error Rate

```
# PromQL
rate(pflow_errors{node_type="PromptNode"}[5m])
```

### HITL Frequency

```
# PromQL
rate(pflow_hitl_requests[1h])
```

## Troubleshooting

### No traces appearing

1. Check telemetry is enabled: `is_enabled()` returns `True`
2. Verify OTLP endpoint is reachable
3. Check sampling rate is not 0.0
4. Verify collector is running and accepting traces

### Missing metrics

1. Metrics export interval may not have elapsed (default 5s)
2. Check OTLP endpoint accepts metrics at `/v1/metrics`
3. Verify meter provider is configured

### High cardinality warnings

If you see cardinality warnings:

1. Reduce dynamic attributes (node IDs, checkpoint IDs)
2. Use exemplars instead of labels for trace linking
3. Aggregate metrics in collector before exporting

## Best Practices

1. **Always call setup_telemetry()** at application startup
2. **Use environment variables** for configuration in production
3. **Sample in production** - 10% is usually sufficient
4. **Monitor cardinality** - keep metric label sets small
5. **Use exemplars** for high-cardinality trace linking
6. **Set service name** to identify your application
7. **Test with console export** before deploying
8. **Disable in tests** to avoid overhead (unless testing telemetry)

## Example

See `examples/telemetry_demo.py` for a complete working example.
