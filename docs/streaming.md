># Streaming Architecture

## Overview

**pydantic-flow** is now streaming-native by default. Every `Node` exposes an async stream of progress as its primary interface, with non-streaming results produced by consuming the stream internally.

## Design Principles

### Streaming First

The primary interface for every node is `astream()`:

```python
async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
    """Stream progress items while executing."""
```

The `run()` method is a convenience wrapper that consumes the stream:

```python
async def run(self, input_data: InputT) -> OutputT:
    """Consume the stream and return final result."""
```

### Small Progress Vocabulary

Nodes emit a focused set of progress item types:

- **StreamStart**: Execution begins with input preview
- **TokenChunk**: Text token from LLM
- **PartialFields**: Incremental structured field updates
- **ToolCall**: Tool invocation declared
- **ToolArgProgress**: Tool arguments forming
- **ToolResult**: Tool execution completed
- **RetrievalItem**: Search/retrieval result
- **NonFatalError**: Recoverable error or warning
- **StreamEnd**: Execution completes with result preview
- **Heartbeat**: Liveness signal during long operations

### Agent Integration

The framework integrates user-supplied `pydantic-ai` agents without wrapping or renaming them:

```python
from pydantic_ai import Agent
from pydantic_flow import AgentNode

# User creates their agent
agent = Agent("openai:gpt-4", instructions="Be helpful")

# Node uses the agent directly
node = AgentNode(agent=agent, prompt_template="{question}")
```

Observer utilities translate agent streaming into progress items.

## Core Components

### BaseNode

All nodes inherit from `BaseNode` which provides:

```python
class BaseNode[InputT, OutputT](ABC):
    @abstractmethod
    async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
        """Primary streaming interface."""
        
    async def run(self, input_data: InputT) -> OutputT:
        """Convenience method that consumes astream()."""
```

### AgentNode

Wraps a pydantic-ai agent for streaming execution:

```python
node = AgentNode[Query, str](
    agent=agent,                    # User-supplied pydantic-ai agent
    prompt_template="{question}",   # Optional template
    name="answer_node"
)

# Stream tokens
async for token in iter_tokens(node.astream(query)):
    print(token, end="")

# Or get final result
result = await node.run(query)
```

### LLMNode

Agent node with structured output support:

```python
node = LLMNode[Query, Answer](
    agent=agent,                    # Agent configured for structured output
    prompt_template="Answer: {question}",
    name="structured_answer"
)

# Stream includes tokens and partial fields
async for item in node.astream(query):
    if isinstance(item, TokenChunk):
        print(item.text, end="")
    elif isinstance(item, PartialFields):
        print(f"\nFields: {item.fields}")
```

### RetrieverNode

Incremental retrieval with progressive results:

```python
async def retriever_fn(query):
    # Yield results as they're found
    for item in search_database(query):
        yield {
            "id": item.id,
            "content": item.content,
            "score": item.score
        }

node = RetrieverNode(retriever_fn=retriever_fn)

# Downstream nodes can react before all results are in
async for item in node.astream(query):
    if isinstance(item, RetrievalItem):
        process_early(item)
```

## Streaming Parser

Tolerant incremental JSON parsing:

```python
from pydantic_flow.streaming import StreamingParser, PartialFields

parser = StreamingParser(MyModel)

async for item in parser.astream_parse(text_stream):
    if isinstance(item, PartialFields):
        # Act on partial fields
        print(f"Partial: {item.fields}")
    elif isinstance(item, MyModel):
        # Final validated model
        print(f"Complete: {item}")
```

Handles:
- Malformed partial JSON with repair strategies
- Incremental field extraction
- Final validation
- Non-fatal errors for recovery attempts

## Ergonomic Helpers

### Token Streaming

```python
from pydantic_flow import iter_tokens

async for token in iter_tokens(node.astream(input)):
    print(token, end="", flush=True)
```

### Field Observation

```python
from pydantic_flow import iter_fields

async for fields in iter_fields(node.astream(input)):
    update_ui(fields)
```

### Collect Results

```python
from pydantic_flow import collect_all_tokens, collect_final_result

# All tokens as string
text = await collect_all_tokens(node.astream(input))

# Final result only
result = await collect_final_result(node.astream(input))
```

## Flow Streaming

Flows can forward progress from child nodes with source tagging:

```python
# Future enhancement
async for item in flow.astream(input):
    print(f"[{item.node_id}] {item.type}: {item}")
```

Supports:
- Natural backpressure via async iteration
- Cancellation propagation to active nodes
- Observer subscribers for logging/metrics

## Observer Pattern

Subscribe to progress without coupling to specific telemetry:

```python
# Future enhancement
class MyObserver:
    async def on_progress(self, item: ProgressItem):
        log.info(f"{item.node_id}: {item.type}")

flow.add_observer(MyObserver())
```

All progress items include:
- `timestamp`: When the event occurred
- `run_id`: Unique execution identifier
- `node_id`: Source node identifier

## Migration from Non-Streaming

### Before (Hypothetical Old API)

```python
result = await node.execute(input)
```

### After (Streaming-Native)

```python
# Same interface for compatibility
result = await node.run(input)

# Or stream for visibility
async for item in node.astream(input):
    handle_progress(item)
```

No breaking changes for simple `run()` usage. Streaming is opt-in via `astream()`.

## Examples

### Simple Streaming

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_flow import AgentNode, iter_tokens

class Query(BaseModel):
    question: str

agent = Agent("openai:gpt-4")
node = AgentNode[Query, str](agent=agent)

query = Query(question="What is AI?")

# Stream tokens
async for token in iter_tokens(node.astream(query)):
    print(token, end="")
```

### Structured Output

```python
class Answer(BaseModel):
    summary: str
    confidence: float

agent = Agent("openai:gpt-4", output_type=Answer)
node = LLMNode[Query, Answer](agent=agent, prompt_template="{question}")

# Get validated result
result = await node.run(query)
print(f"Summary: {result.summary}")
```

### Incremental Retrieval

```python
async def search(query):
    for result in db.search(query.text):
        yield {"id": result.id, "content": result.text, "score": result.score}

retriever = RetrieverNode(retriever_fn=search)
llm = LLMNode(agent=agent, prompt_template="Summarize: {results}")

# Connect nodes
llm.input = retriever.output

# Both stream progress
async for item in llm.astream(query):
    if isinstance(item, RetrievalItem):
        print(f"Found: {item.content[:50]}")
    elif isinstance(item, TokenChunk):
        print(item.text, end="")
```

## Best Practices

### Error Handling

Nodes emit `NonFatalError` for recoverable issues:

```python
async for item in node.astream(input):
    if isinstance(item, NonFatalError):
        if not item.recoverable:
            # Fatal error, stream will raise exception
            break
        # Log and continue
        log.warning(item.message)
```

### Cancellation

Streams support clean cancellation:

```python
async with timeout(10):
    async for item in node.astream(input):
        process(item)
# Stream closes cleanly on timeout
```

### Heartbeats

Long-running nodes emit heartbeats:

```python
async for item in node.astream(input):
    if isinstance(item, Heartbeat):
        update_progress_bar(item.message)
```

### Backpressure

Async iteration provides natural backpressure:

```python
async for item in node.astream(input):
    # Slow processing slows production
    await slow_operation(item)
```

## Performance

Streaming adds minimal overhead:
- Progress items are lightweight frozen dataclasses
- `run()` consumes stream efficiently
- No buffering unless explicitly requested
- Same final result as hypothetical non-streaming path

## Testing

### Stream Coherence

```python
items = []
async for item in node.astream(input):
    items.append(item)

assert items[0].type == ProgressType.START
assert items[-1].type == ProgressType.END
```

### Result Parity

```python
# Streaming
tokens = await collect_all_tokens(node.astream(input))

# Non-streaming
result = await node.run(input)

# Should match (modulo whitespace)
assert tokens.strip() == result.strip()
```

### Fault Injection

```python
# Test malformed JSON
async def bad_stream():
    yield '{"field": "value'  # Incomplete

parser = StreamingParser(MyModel)
with pytest.raises(ValueError):
    async for _ in parser.astream_parse(bad_stream()):
        pass
```

## Future Enhancements

- Flow-level streaming with source tagging
- Structured streaming for complex types
- Built-in retry with streaming visibility
- Distributed tracing integration
- Streaming aggregation nodes
