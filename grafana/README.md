# Grafana Dashboards and Alerts

This directory contains Grafana dashboards and Prometheus alert rules for monitoring pydantic-flow applications.

## Dashboards

### Flow Overview (`dashboards/flow-overview.json`)

Comprehensive overview dashboard showing:

- **Flow Execution Rate**: Flows executed per second by execution mode
- **Flow Duration P95**: 95th and 50th percentile latencies
- **Node Executions by Type**: Rate of executions by node type
- **Cache Hit Rate**: Cache effectiveness gauge
- **Error Rate**: Errors per second by type (with alert)
- **HITL Requests**: Human-in-the-loop interruptions
- **Checkpoint Operations**: Read/write rates for checkpoints
- **Node Duration Heatmap**: Distribution of node execution times

### Durability & Checkpoints (`dashboards/durability.json`)

Deep-dive dashboard for checkpoint operations and durability system:

- **Checkpoint Write Rate by Mode**: Track ASYNC/SYNC/EXIT checkpoint frequency
- **Checkpoint Write Latency**: P50/P95/P99 persistence latency with alerts
- **Checkpoint Writes by Node**: Identify nodes generating most checkpoints
- **Checkpoint Size Distribution**: Monitor checkpoint size trends
- **SYNC vs ASYNC Comparison**: Compare mode performance side-by-side
- **Durability Mode Distribution**: Pie chart of mode usage
- **Checkpoint Failure Rate**: Track and alert on write failures
- **Background Queue Size**: Monitor async checkpoint backlog
- **Latency by Node Table**: Detailed breakdown of checkpoint overhead per node

**Use Cases:**
- Optimize checkpoint performance
- Detect checkpoint store bottlenecks
- Compare durability mode tradeoffs
- Debug background checkpoint issues
- Monitor production checkpoint health

### Importing Dashboards

#### Via Grafana UI

1. Open Grafana
2. Go to Dashboards → Import
3. Upload the JSON file or paste the contents
4. Select your Prometheus data source
5. Click Import

#### Via Provisioning

Add to your Grafana provisioning config:

```yaml
# /etc/grafana/provisioning/dashboards/pydantic-flow.yaml
apiVersion: 1

providers:
  - name: 'pydantic-flow'
    folder: 'Pydantic Flow'
    type: file
    options:
      path: /path/to/grafana/dashboards
```

## Alerts

See `alerts.md` for Prometheus alert rules and response playbooks.

### Key Alerts

- **PydanticFlowHighErrorRate**: Error rate > 0.1/sec for 2 minutes
- **PydanticFlowSlowExecution**: P95 duration > 30 seconds for 5 minutes
- **PydanticFlowLowCacheHitRate**: Cache hit rate < 50% for 10 minutes
- **PydanticFlowCheckpointWriteErrors**: Checkpoint write failures
- **PydanticFlowNoExecutions**: No flows executed for 10 minutes

## Metrics Reference

### Flow Metrics

- `pflow_flow_runs_total`: Total flow executions (counter)
- `pflow_flow_duration_ms`: Flow execution duration (histogram)

### Node Metrics

- `pflow_node_executions_total`: Total node executions (counter)
- `pflow_node_duration_ms`: Node execution duration (histogram)

### Cache Metrics

- `pflow_cache_lookups_total`: Cache lookup attempts (counter)
- `pflow_cache_hits_total`: Cache hits (counter)
- `pflow_cache_misses_total`: Cache misses (counter)
- `pflow_cache_writes_total`: Cache writes (counter)
- `pflow_cache_lookup_duration_ms`: Cache lookup latency (histogram)
- `pflow_cache_write_duration_ms`: Cache write latency (histogram)

### Checkpoint Metrics

- `pflow_checkpoint_reads_total`: Checkpoint reads (counter)
- `pflow_checkpoint_writes_total`: Checkpoint writes (counter)
  - Labels: `node_id`, `checkpoint_durability_mode`, `checkpoint_size_bytes`
- `pflow_checkpoint_read_duration_ms`: Read latency (histogram)
- `pflow_checkpoint_write_duration_ms`: Write latency (histogram)
  - Labels: `node_id`, `checkpoint_durability_mode`
- `pflow_checkpoint_errors_total`: Checkpoint operation failures (counter)
- `pflow_background_checkpoint_tasks`: Active async checkpoint tasks (gauge)

### HITL Metrics

- `pflow_hitl_requests_total`: HITL interruptions (counter)
- `pflow_hitl_responses_total`: HITL responses (counter)

### Error Metrics

- `pflow_errors_total`: Total errors (counter with error_type label)

## Example Queries

### Cache Hit Rate

```promql
rate(pflow_cache_hits_total[5m]) / rate(pflow_cache_lookups_total[5m])
```

### P95 Flow Duration

```promql
histogram_quantile(0.95, rate(pflow_flow_duration_ms_bucket[5m]))
```

### Error Rate by Node Type

```promql
rate(pflow_errors_total[5m]) by (pflow_node_type)
```

### Node Executions per Minute

```promql
rate(pflow_node_executions_total[1m]) * 60
```

### Slowest Nodes (P99)

```promql
histogram_quantile(0.99, rate(pflow_node_duration_ms_bucket[5m])) by (pflow_node_type)
```

### Checkpoint Write Rate by Durability Mode

```promql
rate(pflow_checkpoint_writes_total[5m]) by (checkpoint_durability_mode)
```

### P95 Checkpoint Latency by Node

```promql
histogram_quantile(0.95, rate(pflow_checkpoint_write_duration_ms_bucket[5m])) by (node_id)
```

### Background Checkpoint Queue Depth

```promql
pflow_background_checkpoint_tasks
```

## Customization

### Adding Custom Panels

1. Clone an existing dashboard
2. Add new panels with your metrics
3. Export and save the updated JSON
4. Share with your team

### Modifying Alerts

1. Edit alert thresholds in `alerts.md`
2. Apply to your Prometheus configuration
3. Test alerts using `amtool` or Prometheus UI
4. Configure notification channels in Alertmanager

## Troubleshooting

### No Data in Dashboards

1. Verify Prometheus is scraping metrics from your application
2. Check that telemetry is enabled: `setup_telemetry(enabled=True)`
3. Confirm OTLP endpoint is correct
4. Check Prometheus targets are healthy

### Alerts Not Firing

1. Verify alert rules are loaded in Prometheus
2. Check alert rule syntax with `promtool check rules`
3. Inspect alert state in Prometheus UI
4. Verify Alertmanager is receiving alerts

### High Cardinality Warnings

If you see high cardinality warnings:

1. Review metric labels - avoid dynamic values
2. Use recording rules to pre-aggregate
3. Adjust retention policies
4. Consider sampling more aggressively
