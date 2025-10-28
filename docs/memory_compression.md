# Memory Compression

**pydantic-flow** provides a pluggable memory compression system that automatically manages conversation context when approaching LLM token limits. The compression system is type-safe, streaming-native, and fully integrated with the framework's Human-in-the-Loop (HITL) capabilities.

## Overview

As conversational AI agents process longer interactions, conversation history can grow beyond token limits. Memory compression solves this by intelligently condensing older messages while preserving recent context and system messages.

### Key Features

- **Pluggable Strategies**: Protocol-based design allows custom compression implementations
- **Built-in Compressors**: Sliding window, LLM-based summarization, and hybrid strategies
- **Automatic Triggering**: Compression activates automatically based on token estimates
- **HITL Integration**: Human approval/rejection of compression decisions
- **Streaming Events**: Real-time visibility into compression operations
- **Type-Safe**: Full Pydantic validation throughout
- **Metrics Tracking**: Comprehensive compression history and statistics

## Quick Start

### Basic Configuration

```python
from pydantic_flow import Flow, MemoryConfig, SlidingWindowCompressor

# Create flow with compression enabled
flow = Flow[InputType, OutputType](
    memory_config=MemoryConfig(
        enable_conversation_memory=True,
        compressor=SlidingWindowCompressor(window_size=10),
        emit_compression_events=True,
    )
)
```

### With LLM-Based Summarization

```python
from pydantic_ai import Agent
from pydantic_flow import SummarizationCompressor

# Create summarization agent
summarizer = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Summarize the conversation concisely, preserving key points.",
)

# Use summarization compressor
flow = Flow[InputType, OutputType](
    memory_config=MemoryConfig(
        enable_conversation_memory=True,
        compressor=SummarizationCompressor(agent=summarizer),
    )
)
```

## Architecture

### Compression Flow

1. **Trigger**: `ConversationMemory.extend()` checks if compression is needed
2. **Pending Event**: Emits `MemoryCompressionPending` (interruptible)
3. **Compression**: Compressor reduces message history
4. **Complete Event**: Emits `MemoryCompressionComplete` with metrics (interruptible)
5. **Update**: Memory updated with compressed messages

### Components

```
┌─────────────────────────────────────────────────────────┐
│                    MemoryConfig                         │
│  - compressor: MemoryCompressor | None                 │
│  - emit_compression_events: bool                        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                ConversationMemory                       │
│  - maybe_compress() -> CompressionMetrics | None       │
│  - compression_history: list[CompressionMetrics]       │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                MemoryCompressor Protocol                │
│  - should_compress(messages) -> bool                    │
│  - compress(messages) -> tuple[list, Metrics]           │
│  - name() -> str                                        │
└─────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
  ┌──────────┐   ┌─────────────────┐  ┌──────────────┐
  │ Sliding  │   │ Summarization   │  │   Hybrid     │
  │  Window  │   │   Compressor    │  │  Compressor  │
  └──────────┘   └─────────────────┘  └──────────────┘
```

## Built-in Compressor Strategies

### SlidingWindowCompressor

Keeps only the N most recent messages. Fast and simple, ideal for conversational flows where recent context is sufficient.

```python
from pydantic_flow import SlidingWindowCompressor

compressor = SlidingWindowCompressor(
    window_size=10,                    # Keep last 10 messages
    preserve_system_messages=True,     # Keep system prompts
    max_tokens=4000,                   # Trigger at 4K tokens
)
```

**Pros:**
- Extremely fast (no LLM calls)
- Predictable memory usage
- Simple configuration

**Cons:**
- Loses older context entirely
- No semantic understanding
- Fixed window size

### SummarizationCompressor

Uses an LLM agent to create concise summaries of older messages, preserving semantic information.

```python
from pydantic_ai import Agent
from pydantic_flow import SummarizationCompressor

summarizer = Agent("openai:gpt-4o-mini")

compressor = SummarizationCompressor(
    agent=summarizer,
    prompt_template="""
    Summarize this conversation history concisely:
    
    {messages}
    
    Preserve key facts, decisions, and context.
    """,
    preserve_system_messages=True,
    max_tokens=4000,
)
```

**Pros:**
- Preserves semantic meaning
- Intelligent condensation
- Customizable via prompts

**Cons:**
- Slower (requires LLM call)
- Additional API costs
- Less predictable

### HybridCompressor

Combines both strategies: uses sliding window for small histories, switches to summarization for larger ones.

```python
from pydantic_flow import HybridCompressor

compressor = HybridCompressor(
    summarization_threshold=15,        # Switch to summarization at 15+ messages
    window_size=10,                    # Sliding window size
    agent=summarizer,                  # Agent for summarization
)
```

**Pros:**
- Best of both worlds
- Adaptive to context size
- Cost-effective

**Cons:**
- More complex configuration
- Strategy switching overhead

## Custom Compressor Implementation

Implement the `MemoryCompressor` protocol for custom strategies:

```python
from typing import Any
from pydantic import BaseModel
from pydantic_flow import MemoryCompressor, CompressionMetrics, BaseMemoryCompressor

class TokenBudgetCompressor(BaseMemoryCompressor, BaseModel):
    """Compress to stay within token budget."""
    
    target_tokens: int = 2000
    
    def name(self) -> str:
        return f"token_budget_{self.target_tokens}"
    
    def should_compress(
        self,
        messages: list[dict[str, Any] | Any],
    ) -> bool:
        """Trigger when over budget."""
        tokens = self.estimate_tokens(messages)
        return tokens > self.target_tokens
    
    async def compress(
        self,
        messages: list[dict[str, Any] | Any],
    ) -> tuple[list[dict[str, Any] | Any], CompressionMetrics]:
        """Remove messages until under budget."""
        import time
        start = time.time()
        
        # Partition messages
        system_msgs, compressible, recent = self.partition_messages(
            messages, keep_recent=5
        )
        
        # Keep removing oldest until under budget
        compressed = compressible.copy()
        while self.estimate_tokens(system_msgs + compressed + recent) > self.target_tokens:
            if not compressed:
                break
            compressed.pop(0)
        
        result = system_msgs + compressed + recent
        
        # Create metrics
        tokens_before = self.estimate_tokens(messages)
        tokens_after = self.estimate_tokens(result)
        
        return result, CompressionMetrics(
            messages_before=len(messages),
            messages_after=len(result),
            estimated_tokens_before=tokens_before,
            estimated_tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
            compression_strategy=self.name(),
            compression_ratio=tokens_after / tokens_before if tokens_before > 0 else 1.0,
            compression_time_ms=(time.time() - start) * 1000,
        )
```

## HITL Integration

Compression events can be intercepted for human approval:

### Approving Compression Before Execution

```python
from pydantic_flow import InterruptDecision, MemoryCompressionPending

async def approve_compression(event: MemoryCompressionPending) -> InterruptDecision:
    """Ask human before compressing."""
    print(f"About to compress {event.message_count} messages")
    print(f"Estimated tokens: {event.estimated_tokens}")
    print(f"Strategy: {event.compressor_name}")
    print(f"Reason: {event.compression_reason}")
    
    response = input("Approve compression? (y/n): ")
    
    if response.lower() == 'y':
        return InterruptDecision.proceed("User approved")
    else:
        return InterruptDecision.interrupt("User rejected compression")

# Register handler
flow.register_interrupt_handler(
    ProgressType.MEMORY_COMPRESSION_PENDING,
    approve_compression,
)
```

### Validating Compression Results

```python
from pydantic_flow import MemoryCompressionComplete

async def validate_compression(event: MemoryCompressionComplete) -> InterruptDecision:
    """Review compression results."""
    metrics = event.metrics
    
    print(f"Compression complete:")
    print(f"  Messages: {metrics.messages_before} → {metrics.messages_after}")
    print(f"  Tokens saved: {metrics.tokens_saved}")
    print(f"  Reduction: {metrics.percentage_reduction:.1f}%")
    print(f"  Time: {metrics.compression_time_ms:.0f}ms")
    
    # Show preview of compressed messages
    if event.compressed_messages_preview:
        print(f"\nFirst compressed message:")
        print(event.compressed_messages_preview[0])
    
    response = input("Accept compression? (y/n): ")
    
    if response.lower() == 'y':
        return InterruptDecision.proceed("Results validated")
    else:
        # Revert to original messages
        return InterruptDecision.interrupt(
            "Compression rejected",
            metadata={"revert": True}
        )

flow.register_interrupt_handler(
    ProgressType.MEMORY_COMPRESSION_COMPLETE,
    validate_compression,
)
```

### Using Alternative Compressor

```python
from pydantic_flow import SlidingWindowCompressor

async def switch_strategy(event: MemoryCompressionPending) -> InterruptDecision:
    """Use different strategy based on context."""
    
    if event.message_count > 50:
        # Use aggressive compression for large histories
        alternative = SlidingWindowCompressor(window_size=5)
        return InterruptDecision.interrupt(
            "Switching to aggressive strategy",
            replacement_value=alternative,
        )
    
    return InterruptDecision.proceed()
```

## Configuration

### MemoryConfig Options

```python
from pydantic_flow import MemoryConfig

config = MemoryConfig(
    # Enable conversation memory
    enable_conversation_memory=True,
    
    # Compressor instance (None = no compression)
    compressor=SlidingWindowCompressor(window_size=10),
    
    # Emit compression events for observability/HITL
    emit_compression_events=True,
)
```

### Compressor Configuration

All built-in compressors inherit from `BaseMemoryCompressor`:

```python
class BaseMemoryCompressor:
    preserve_system_messages: bool = True  # Keep system prompts
    max_tokens: int = 4000                 # Compression trigger threshold
    keep_recent: int = 5                   # Always preserve N recent messages
```

## Event Handling

### Observing Compression Events

```python
async for event in flow.astream(input_data):
    if event.type == ProgressType.MEMORY_COMPRESSION_PENDING:
        print(f"⏳ Compression pending: {event.message_count} messages")
        
    elif event.type == ProgressType.MEMORY_COMPRESSION_COMPLETE:
        metrics = event.metrics
        print(f"✅ Compressed: {metrics.percentage_reduction:.1f}% reduction")
        print(f"   Saved {metrics.tokens_saved} tokens in {metrics.compression_time_ms:.0f}ms")
```

### Compression Metrics

Track compression history:

```python
# Access compression history
history = flow._conversation_memory.compression_history

for metrics in history:
    print(f"Strategy: {metrics.compression_strategy}")
    print(f"Reduction: {metrics.percentage_reduction:.1f}%")
    print(f"Messages removed: {metrics.messages_removed}")
```

## Best Practices

### 1. Choose the Right Strategy

- **SlidingWindow**: Short conversations, recent context sufficient
- **Summarization**: Long conversations needing semantic preservation
- **Hybrid**: General purpose, adapts to conversation length

### 2. Set Appropriate Thresholds

```python
# Conservative: compress less frequently
compressor = SlidingWindowCompressor(max_tokens=8000, window_size=20)

# Aggressive: compress early and often
compressor = SlidingWindowCompressor(max_tokens=2000, window_size=5)
```

### 3. Monitor Compression Metrics

```python
def analyze_compression_patterns(memory):
    """Analyze compression effectiveness."""
    history = memory.compression_history
    
    if not history:
        return
    
    avg_reduction = sum(m.percentage_reduction for m in history) / len(history)
    avg_time = sum(m.compression_time_ms for m in history) / len(history)
    total_saved = sum(m.tokens_saved for m in history)
    
    print(f"Compression Statistics:")
    print(f"  Count: {len(history)}")
    print(f"  Avg reduction: {avg_reduction:.1f}%")
    print(f"  Avg time: {avg_time:.0f}ms")
    print(f"  Total tokens saved: {total_saved}")
```

### 4. Handle Compression Failures Gracefully

```python
class RobustCompressor(BaseMemoryCompressor, BaseModel):
    fallback: MemoryCompressor
    
    async def compress(self, messages):
        try:
            return await super().compress(messages)
        except Exception as e:
            # Fall back to simple strategy
            return await self.fallback.compress(messages)
```

### 5. Test Compression Strategies

```python
def test_compressor_effectiveness(compressor, test_messages):
    """Evaluate compression quality."""
    original_tokens = compressor.estimate_tokens(test_messages)
    
    compressed, metrics = await compressor.compress(test_messages)
    
    assert metrics.compression_ratio < 0.8, "Insufficient compression"
    assert len(compressed) > 0, "Over-compressed"
    assert metrics.compression_time_ms < 1000, "Too slow"
```

## Performance Considerations

### Token Estimation

Built-in estimators use a simple heuristic (4 chars/token). For more accuracy:

```python
class AccurateCompressor(BaseMemoryCompressor):
    def estimate_tokens(self, messages):
        # Use actual tokenizer
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        
        text = "\n".join(self._extract_content(m) for m in messages)
        return len(enc.encode(text))
```

### Async Compression

All compressors support async to avoid blocking:

```python
async def compress(self, messages):
    # Long-running compression
    summary = await self.agent.run("Summarize: " + format_messages(messages))
    return create_compressed_messages(summary)
```

### Caching Summaries

For repeated compressions:

```python
class CachingCompressor(SummarizationCompressor):
    _cache: dict = {}
    
    async def compress(self, messages):
        key = hash(tuple(self._extract_content(m) for m in messages))
        
        if key in self._cache:
            return self._cache[key]
        
        result = await super().compress(messages)
        self._cache[key] = result
        return result
```

## Troubleshooting

### Compression Not Triggering

**Problem**: Messages accumulate without compression.

**Solutions**:
- Check `max_tokens` threshold isn't too high
- Verify `compressor` is set in `MemoryConfig`
- Ensure `enable_conversation_memory=True`
- Check token estimation accuracy

### Over-Compression

**Problem**: Too much context lost.

**Solutions**:
- Increase `window_size` for sliding window
- Increase `keep_recent` to preserve more messages
- Use summarization instead of sliding window
- Adjust `max_tokens` threshold higher

### Slow Compression

**Problem**: Compression causes noticeable delays.

**Solutions**:
- Use `SlidingWindowCompressor` (no LLM calls)
- Switch to faster LLM model for summarization
- Increase `max_tokens` to compress less frequently
- Consider async execution patterns

### Inconsistent Quality

**Problem**: Summarization quality varies.

**Solutions**:
- Refine `prompt_template` for summarizer
- Use more capable LLM model
- Add few-shot examples to prompt
- Validate results with HITL approval

### Memory Leaks

**Problem**: Memory usage grows over time.

**Solutions**:
- Ensure compression is actually running (check metrics)
- Clear compression history periodically:
  ```python
  memory._compression_history.clear()
  ```
- Use appropriate `window_size` for workload

## Examples

See the `/examples` directory for complete working examples:

- `memory_compression_basic.py` - Sliding window and summarization
- `memory_compression_approval.py` - HITL approval workflow  
- `memory_compression_custom.py` - Custom compressor implementation
- `memory_compression_metrics.py` - Monitoring and analytics

## API Reference

### Core Types

- `MemoryCompressor` - Protocol defining compressor interface
- `BaseMemoryCompressor` - Abstract base class with helpers
- `CompressionMetrics` - Compression statistics and metadata

### Built-in Compressors

- `SlidingWindowCompressor` - Keep N recent messages
- `SummarizationCompressor` - LLM-based summarization
- `HybridCompressor` - Adaptive strategy selection

### Events

- `MemoryCompressionPending` - Before compression (interruptible)
- `MemoryCompressionComplete` - After compression (interruptible)

### Configuration

- `MemoryConfig` - Memory and compression settings
- `ConversationMemory` - Enhanced with compression support

---

For more examples and use cases, see the [examples directory](../examples/) and the [API documentation](https://pydantic-flow.dev).
