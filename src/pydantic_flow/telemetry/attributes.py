"""Telemetry attribute keys, event names, and metric names.

This module defines the vocabulary for spans, events, and metrics to ensure
consistent naming across the framework.
"""

from enum import StrEnum


class SpanKind(StrEnum):
    """Semantic span kinds for pydantic-flow operations."""

    FLOW_RUN = "flow_run"
    NODE_RUN = "node_run"
    CACHE_LOOKUP = "cache_lookup"
    CACHE_WRITE = "cache_write"
    CHECKPOINT_READ = "checkpoint_read"
    CHECKPOINT_WRITE = "checkpoint_write"
    MEMORY_COMPRESS = "memory_compress"
    AGENT_CALL = "agent_call"
    RETRIEVER_QUERY = "retriever_query"
    HUMAN_GATE = "human_gate"


class EventName(StrEnum):
    """Stream event names recorded as span events."""

    STREAM_START = "stream.start"
    STREAM_CHUNK = "stream.chunk"
    STREAM_END = "stream.end"
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"
    CACHE_WRITE = "cache.write"
    CACHE_ERROR = "cache.error"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    HUMAN_REQUESTED = "human.requested"
    HUMAN_RESPONSE = "human.response"
    MEMORY_COMPRESS_PENDING = "memory.compress.pending"
    MEMORY_COMPRESS_COMPLETE = "memory.compress.complete"
    CHECKPOINT_SAVED = "checkpoint.saved"
    ERROR = "error"


class AttributeKey(StrEnum):
    """Attribute keys for spans and metrics.

    Uses pflow.* namespace for framework-specific attributes and standard
    semantic conventions for general concepts.
    """

    # Flow attributes
    FLOW_ID = "pflow.flow.id"
    FLOW_NAME = "pflow.flow.name"
    RUN_ID = "pflow.run.id"
    EXECUTION_MODE = "pflow.execution.mode"

    # Node attributes
    NODE_ID = "pflow.node.id"
    NODE_NAME = "pflow.node.name"
    NODE_TYPE = "pflow.node.type"
    LOOP_ITERATION = "pflow.loop.iteration"
    ROUTE_BRANCH = "pflow.route.branch"
    RETRY_NUMBER = "pflow.retry.number"
    RESUME_FROM_CHECKPOINT = "pflow.resume.from_checkpoint"

    # Cache attributes
    CACHE_KEY_HASH = "pflow.cache.key_hash"
    CACHE_POLICY = "pflow.cache.policy"
    CACHE_BACKEND = "pflow.cache.backend"
    CACHE_RESULT = "cache.result"
    CACHE_TTL_REMAINING = "pflow.cache.ttl_remaining"

    # Checkpoint attributes
    CHECKPOINT_BACKEND = "pflow.checkpoint.backend"
    CHECKPOINT_ID = "pflow.checkpoint.id"
    CHECKPOINT_KEY = "pflow.checkpoint.key"
    CHECKPOINT_IS_INTERRUPTED = "pflow.checkpoint.interrupted"
    CHECKPOINT_DURABILITY_MODE = "pflow.checkpoint.durability_mode"
    CHECKPOINT_SIZE_BYTES = "pflow.checkpoint.size_bytes"

    # LLM attributes (align with semantic conventions)
    LLM_MODEL = "llm.model.name"
    LLM_PROVIDER = "llm.provider"
    LLM_TEMPERATURE = "llm.temperature"
    LLM_TOKENS_PROMPT = "llm.tokens.prompt"
    LLM_TOKENS_COMPLETION = "llm.tokens.completion"

    # Memory attributes
    MEMORY_MESSAGES_BEFORE = "pflow.memory.messages.before"
    MEMORY_MESSAGES_AFTER = "pflow.memory.messages.after"
    MEMORY_TOKENS_SAVED = "pflow.memory.tokens.saved"
    MEMORY_COMPRESSION_RATIO = "pflow.memory.compression.ratio"

    # Status and outcome
    OUTCOME = "outcome"
    ERROR_TYPE = "error.type"
    ERROR_MESSAGE = "error.message"

    # HITL attributes
    HITL_PROMPT = "pflow.hitl.prompt"
    HITL_INPUT_TYPE = "pflow.hitl.input_type"
    HITL_APPROVED = "pflow.hitl.approved"


class MetricName(StrEnum):
    """Metric names for counters, histograms, and gauges."""

    # Counters
    FLOW_RUNS = "pflow.flow.runs"
    NODE_EXECUTIONS = "pflow.node.executions"
    ERRORS = "pflow.errors"
    CACHE_LOOKUPS = "pflow.cache.lookups"
    CACHE_HITS = "pflow.cache.hits"
    CACHE_MISSES = "pflow.cache.misses"
    CACHE_WRITES = "pflow.cache.writes"
    CHECKPOINT_READS = "pflow.checkpoint.reads"
    CHECKPOINT_WRITES = "pflow.checkpoint.writes"
    HITL_REQUESTS = "pflow.hitl.requests"
    HITL_RESPONSES = "pflow.hitl.responses"

    # Histograms
    FLOW_DURATION = "pflow.flow.duration.ms"
    NODE_DURATION = "pflow.node.duration.ms"
    CACHE_LOOKUP_DURATION = "pflow.cache.lookup.duration.ms"
    CACHE_WRITE_DURATION = "pflow.cache.write.duration.ms"
    CHECKPOINT_READ_DURATION = "pflow.checkpoint.read.duration.ms"
    CHECKPOINT_WRITE_DURATION = "pflow.checkpoint.write.duration.ms"
    MEMORY_COMPRESS_DURATION = "pflow.memory.compress.duration.ms"
    LLM_TOKENS_PROMPT = "pflow.llm.tokens.prompt"
    LLM_TOKENS_COMPLETION = "pflow.llm.tokens.completion"

    # Gauges
    STREAM_INFLIGHT = "pflow.stream.inflight"
    LOOP_DEPTH = "pflow.loop.depth"


# Outcome values
class Outcome(StrEnum):
    """Standard outcome values for the outcome attribute."""

    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
