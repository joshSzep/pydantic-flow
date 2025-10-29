# Caching

**pydantic-flow** includes first-class caching for LLM responses, embeddings, and other node outputs. The caching layer is type-safe, streaming-native, and supports multiple backends with observability built in.

## Features

- **LLM Response Caching**: Cache completions with automatic key generation from prompts, models, and parameters
- **Embedding Caching**: Cache vector embeddings with content-based hashing
- **Pluggable Backends**: In-memory (with LRU + TTL), SQLite (persistent local), and Redis (distributed) backends
- **Per-Node Policies**: Configure TTL, scope, and behavior per node
- **Stampede Protection**: Singleflight pattern prevents thundering herds
- **Stream Replay**: Optional capture and instant replay of streaming events
- **Observability**: Built-in events for cache hits, misses, writes, and errors
- **Type Safety**: Full Pydantic models with IDE support

## Quickstart

### In-Memory Caching

```python
from datetime import timedelta
from pydantic_flow import Flow
from pydantic_flow.cache import CachePolicy, InMemoryCache
from pydantic_flow.nodes import PromptNode, PromptConfig

# Create a cache backend
cache_backend = InMemoryCache(max_entries=5000)

# Define default policy (optional)
default_policy = CachePolicy(
    enabled=True,
    ttl=timedelta(hours=12),
)

# Create flow with caching
flow = Flow(
    cache_backend=cache_backend,
    default_cache_policy=default_policy,
)

# Add nodes with per-node cache policies
prompt = PromptNode(
    prompt="Summarize: {text}",
    config=PromptConfig(model="gpt-4"),
    cache_policy=CachePolicy(
        ttl=timedelta(minutes=30),  # Override default
    ),
)

flow.add_node(prompt)

# Compiled flow will use caching
compiled = flow.compile()
```

### SQLite Caching (Persistent Local)

```python
from pydantic_flow.cache import SQLiteCache

# Create SQLite cache backend with persistence
cache_backend = SQLiteCache(
    db_path=".pydantic-flow-cache.db",  # Local file
    cleanup_interval=300.0,               # Clean expired entries every 5min
)

# Use context manager for automatic cleanup
async with cache_backend:
    flow = Flow(
        cache_backend=cache_backend,
        default_cache_policy=CachePolicy(
            ttl=timedelta(days=7),  # Persist for a week
        ),
    )
    # ... use flow
```

### Redis Caching (Distributed)

```python
from redis.asyncio import Redis
from pydantic_flow.cache.redis import RedisCache

# Create Redis client
redis_client = Redis(host="localhost", port=6379, decode_responses=False)

# Create Redis cache backend
cache_backend = RedisCache(
    redis=redis_client,
    key_prefix="pf",
    lock_ttl_ms=5000,
    compression_threshold=1024,  # Compress values > 1KB
)

flow = Flow(
    cache_backend=cache_backend,
    default_cache_policy=CachePolicy(),
)
```

## Cache Policies

Configure caching behavior per-node using `CachePolicy`:

```python
from pydantic_flow.cache import CachePolicy, CacheScope

# Basic policy
policy = CachePolicy(
    enabled=True,                      # Enable caching
    ttl=timedelta(hours=1),            # Expire after 1 hour
)

# Namespaced caching
policy = CachePolicy(
    scope=CacheScope.NAMESPACE("production"),
    ttl=timedelta(days=7),
)

# Version-tagged caching
policy = CachePolicy(
    node_version="v2.1",               # Invalidate on version change
    ttl=timedelta(hours=24),
)

# Extra key material
policy = CachePolicy(
    extra_key_material={
        "user_tier": "premium",
        "feature_flags": ["new_model"],
    },
)

# Stream replay (experimental)
policy = CachePolicy(
    store_streams=True,                 # Capture and replay events
    ttl=timedelta(minutes=15),
)

# Bypass cache
policy = CachePolicy(
    bypass=True,                        # Always execute, never cache
)
```

## Cache Key Generation

Cache keys are deterministic hashes of:

### LLM Keys Include
- Provider and model identifier
- All messages (after template rendering)
- System prompt
- Sampling parameters (temperature, top_p, seed)
- Tool schemas and invocation mode
- Node version (if specified)
- Extra key material (if provided)
- Environment label (if configured)

### Embedding Keys Include
- Provider and model identifier
- Input text (hashed)
- Dimension and normalization settings
- Chunking version (if specified)
- Node version and extra material

### Example

```python
from pydantic_flow.cache.key import build_llm_cache_key

key = build_llm_cache_key(
    provider="openai",
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    seed=42,
)
# Result: "pf:global:a3f8b9c2d..."
```

## Invalidation

### Delete by Key

```python
await flow.cache_delete("pf:global:a3f8b9c2d...")
```

### Namespace Invalidation

```python
# Invalidate all keys in "production" namespace
count = await flow.cache_invalidate("production")
print(f"Invalidated {count} keys")
```

### Version Bumping

```python
# Old version
prompt = PromptNode(
    prompt="...",
    cache_policy=CachePolicy(node_version="v1.0"),
)

# New version - automatically invalidates old cached results
prompt = PromptNode(
    prompt="...",
    cache_policy=CachePolicy(node_version="v1.1"),
)
```

## Observability

Cache operations emit events you can observe:

```python
from pydantic_flow.streaming.events import (
    CacheHit,
    CacheMiss,
    CacheWrite,
    CacheError,
)

async for event in compiled.astream(inputs):
    if isinstance(event, CacheHit):
        print(f"Cache hit: {event.key}")
        print(f"TTL remaining: {event.ttl_remaining}s")
    elif isinstance(event, CacheMiss):
        print(f"Cache miss: {event.key}")
    elif isinstance(event, CacheWrite):
        print(f"Cached {event.value_size_bytes} bytes")
    elif isinstance(event, CacheError):
        print(f"Cache error: {event.error}")
```

### OpenTelemetry Integration

Cache operations automatically add span attributes:

- `cache.key`: Cache key used
- `cache.hit`: Boolean indicating hit/miss
- `cache.backend`: Backend name (InMemoryCache, RedisCache)
- `cache.ttl`: TTL in seconds
- `cache.lookup_ms`: Lookup duration
- `cache.write_ms`: Write duration

## Stream Replay (Experimental)

By default, only the final result is cached. Enable stream replay to cache and replay the entire event stream:

```python
policy = CachePolicy(
    store_streams=True,
    ttl=timedelta(minutes=10),
)

prompt = PromptNode(
    prompt="...",
    cache_policy=policy,
)

# First execution: streams normally, captures events
async for event in flow.astream(inputs):
    print(event)

# Second execution: replays captured events instantly (no delays)
async for event in flow.astream(inputs):
    print(event)  # Same events, from cache
```

**Caveats:**
- Higher memory usage (stores all events)
- Events replayed without original timing
- Best for short-lived caches
- Not recommended for long-running operations

## Backend Comparison

| Feature | InMemoryCache | SQLiteCache | RedisCache |
|---------|--------------|-------------|------------|
| **Use Case** | Single process, dev/test | Single server, local persistence | Multi-process/server, production |
| **Persistence** | ❌ Memory only | ✅ SQLite file | ✅ Redis persistence |
| **TTL** | ✅ Yes | ✅ Yes | ✅ Yes |
| **LRU Eviction** | ✅ Yes | ❌ No (use TTL only) | ❌ No (use Redis TTL) |
| **Stampede Protection** | ✅ In-process | ✅ In-process | ✅ Distributed locks |
| **Compression** | ❌ No | ❌ No | ✅ Optional (>1KB) |
| **Namespace Invalidation** | ✅ Yes | ✅ Yes | ✅ Yes (via SCAN) |
| **Concurrency** | Async locks | WAL mode + locks | Distributed |
| **Setup Complexity** | None | None (auto-creates file) | Requires Redis server |
| **Best For** | Development, testing | Single-server apps, local caching | Distributed systems, high scale |

## Best Practices

### 1. Choose Appropriate TTLs

```python
# Short TTL for rapidly changing data
user_profile = PromptNode(
    prompt="Get user profile for {user_id}",
    cache_policy=CachePolicy(ttl=timedelta(minutes=5)),
)

# Long TTL for stable data
embedding = EmbeddingNode(
    text="{document}",
    cache_policy=CachePolicy(ttl=timedelta(days=30)),
)
```

### 2. Use Namespaces for Isolation

```python
# Per-environment caching
dev_policy = CachePolicy(scope=CacheScope.NAMESPACE("dev"))
prod_policy = CachePolicy(scope=CacheScope.NAMESPACE("prod"))

# Per-user caching
user_policy = CachePolicy(
    scope=CacheScope.NAMESPACE(f"user:{user_id}"),
)
```

### 3. Version Your Prompts

```python
# Embed version in policy, not prompt
policy = CachePolicy(
    node_version="2024-01-15-v1",
    ttl=timedelta(hours=24),
)

prompt = PromptNode(
    prompt="Analyze sentiment: {text}",
    cache_policy=policy,
)
```

### 4. Monitor Cache Effectiveness

```python
hits = 0
misses = 0

async for event in flow.astream(inputs):
    if isinstance(event, CacheHit):
        hits += 1
    elif isinstance(event, CacheMiss):
        misses += 1

hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
print(f"Cache hit rate: {hit_rate:.1%}")
```

### 5. Handle Secrets and PII

```python
# Don't cache sensitive data
sensitive_prompt = PromptNode(
    prompt="Process API key: {api_key}",
    cache_policy=CachePolicy(enabled=False),
)

# Or use short TTL and namespacing
user_data = PromptNode(
    prompt="Get data for {user_id}",
    cache_policy=CachePolicy(
        ttl=timedelta(seconds=30),
        scope=CacheScope.NAMESPACE(f"user:{user_id}"),
    ),
)
```

## Advanced Patterns

### Conditional Caching

```python
def should_cache(input_data):
    # Only cache for premium users
    return input_data.get("user_tier") == "premium"

# Implement via custom node or policy bypass flag
policy = CachePolicy(
    bypass=not should_cache(inputs),
)
```

### Cache Warming

```python
# Pre-populate cache with common queries
common_queries = [
    "What is AI?",
    "Explain machine learning",
    "Define neural networks",
]

for query in common_queries:
    await flow.run({"text": query})
```

### Tiered Caching

```python
# Hot path: in-memory, short TTL
hot_policy = CachePolicy(ttl=timedelta(minutes=5))

# Warm path: Redis, medium TTL
warm_policy = CachePolicy(ttl=timedelta(hours=1))

# Cold path: no cache
cold_policy = CachePolicy(enabled=False)
```

## Troubleshooting

### Cache Not Working

1. Verify backend is configured:
   ```python
   assert flow._cache_backend is not None
   ```

2. Check policy is enabled:
   ```python
   assert node.cache_policy is not None
   assert node.cache_policy.enabled
   ```

3. Verify cache events:
   ```python
   async for event in flow.astream(inputs):
       print(f"Event: {event.type}")
   ```

### High Memory Usage (InMemoryCache)

```python
# Reduce max_entries
cache = InMemoryCache(max_entries=1000)

# Reduce cleanup_interval
cache = InMemoryCache(cleanup_interval=30.0)

# Use shorter TTLs
policy = CachePolicy(ttl=timedelta(minutes=10))
```

### Redis Connection Issues

```python
from redis.asyncio import Redis
from redis.exceptions import ConnectionError

try:
    redis_client = Redis(
        host="localhost",
        port=6379,
        socket_connect_timeout=5,
        socket_keepalive=True,
    )
    cache = RedisCache(redis=redis_client)
except ConnectionError as e:
    print(f"Redis unavailable: {e}")
    # Fallback to in-memory
    cache = InMemoryCache()
```

## API Reference

See inline documentation for:
- `pydantic_flow.cache.base` - Core types and abstractions
- `pydantic_flow.cache.memory` - In-memory backend
- `pydantic_flow.cache.redis` - Redis backend
- `pydantic_flow.cache.key` - Key generation utilities
- `pydantic_flow.cache.middleware` - Execution middleware
