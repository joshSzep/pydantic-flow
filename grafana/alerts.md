# Pydantic-Flow Alert Rules

## Prometheus Alert Rules

Copy these rules to your Prometheus `alerts.yml` file:

```yaml
groups:
  - name: pydantic_flow
    interval: 30s
    rules:
      # High error rate
      - alert: PydanticFlowHighErrorRate
        expr: rate(pflow_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in pydantic-flow"
          description: "Error rate is {{ $value | humanize }} errors/sec (threshold: 0.1)"

      # P95 flow duration increase
      - alert: PydanticFlowSlowExecution
        expr: histogram_quantile(0.95, rate(pflow_flow_duration_ms_bucket[5m])) > 30000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow flow execution detected"
          description: "P95 flow duration is {{ $value | humanize }}ms (threshold: 30000ms)"

      # Cache hit rate degradation
      - alert: PydanticFlowLowCacheHitRate
        expr: |
          (
            rate(pflow_cache_hits_total[10m]) /
            rate(pflow_cache_lookups_total[10m])
          ) < 0.5
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value | humanizePercentage }} (threshold: 50%)"

      # High HITL request rate (unusual activity)
      - alert: PydanticFlowHighHITLRate
        expr: rate(pflow_hitl_requests_total[5m]) > 1
        for: 5m
        labels:
          severity: info
        annotations:
          summary: "Unusually high HITL request rate"
          description: "HITL requests at {{ $value | humanize }} requests/sec"

      # Checkpoint write failures
      - alert: PydanticFlowCheckpointWriteErrors
        expr: rate(pflow_errors_total{operation="checkpoint_write"}[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Checkpoint write failures detected"
          description: "Checkpoint writes failing at {{ $value | humanize }} failures/sec"

      # Node execution timeout increase
      - alert: PydanticFlowNodeTimeout
        expr: rate(pflow_errors_total{error_type="TimeoutError"}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Node execution timeouts increasing"
          description: "Timeout rate: {{ $value | humanize }} timeouts/sec"

      # Memory compression failures
      - alert: PydanticFlowMemoryCompressionError
        expr: rate(pflow_errors_total{error_type="MemoryCompressionError"}[5m]) > 0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Memory compression errors detected"
          description: "Memory compression failing for some flows"

      # No flow executions (possible outage)
      - alert: PydanticFlowNoExecutions
        expr: rate(pflow_flow_runs_total[10m]) == 0
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "No flow executions detected"
          description: "No flows have executed in the last 10 minutes"

      # High retry rate
      - alert: PydanticFlowHighRetryRate
        expr: |
          (
            rate(pflow_node_executions_total{retry_number!="0"}[5m]) /
            rate(pflow_node_executions_total[5m])
          ) > 0.3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High node retry rate"
          description: "{{ $value | humanizePercentage }} of executions are retries"
```

## Grafana Alerts

These alerts can also be configured directly in Grafana:

### Critical Alerts

1. **No Flow Executions**
   - Query: `rate(pflow_flow_runs_total[10m]) == 0`
   - Threshold: 0 for 10 minutes
   - Action: Page on-call

2. **Checkpoint Write Failures**
   - Query: `rate(pflow_errors_total{operation="checkpoint_write"}[5m])`
   - Threshold: > 0.01 for 2 minutes
   - Action: Alert ops team

### Warning Alerts

3. **High Error Rate**
   - Query: `rate(pflow_errors_total[5m])`
   - Threshold: > 0.1 for 2 minutes
   - Action: Notify team

4. **Slow Flow Execution**
   - Query: `histogram_quantile(0.95, rate(pflow_flow_duration_ms_bucket[5m]))`
   - Threshold: > 30000ms for 5 minutes
   - Action: Investigate performance

5. **High Retry Rate**
   - Query: `rate(pflow_node_executions_total{retry_number!="0"}[5m]) / rate(pflow_node_executions_total[5m])`
   - Threshold: > 30% for 5 minutes
   - Action: Check service health

### Info Alerts

6. **Low Cache Hit Rate**
   - Query: `rate(pflow_cache_hits_total[10m]) / rate(pflow_cache_lookups_total[10m])`
   - Threshold: < 50% for 10 minutes
   - Action: Review cache configuration

7. **High HITL Rate**
   - Query: `rate(pflow_hitl_requests_total[5m])`
   - Threshold: > 1 req/sec for 5 minutes
   - Action: Monitor for anomalies

## Alert Response Playbook

### High Error Rate
1. Check Grafana for error types and affected nodes
2. Search traces in Jaeger for failed executions
3. Review logs for stack traces
4. Check external service health (LLM APIs, databases)

### Slow Flow Execution
1. Identify slow nodes in flame graphs
2. Check cache hit rate - may need to warm cache
3. Review LLM API latencies
4. Check for database query performance

### Checkpoint Failures
1. Verify checkpoint store connectivity
2. Check storage quota and permissions
3. Review checkpoint payload sizes
4. Test checkpoint store manually

### No Flow Executions
1. Check service health endpoints
2. Verify incoming request traffic
3. Review application logs for startup errors
4. Check dependency availability
