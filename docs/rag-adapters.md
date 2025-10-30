# RAG Adapters

**pydantic-flow** includes a comprehensive, type-safe RAG (Retrieval-Augmented Generation) adapter layer designed for seamless integration with the streaming-native flow architecture.

## Design Principles

- **Type Safety**: All public interfaces use Pydantic models
- **Async Throughout**: Every operation is async-first
- **Minimal APIs**: Small, focused interfaces with no global state
- **Streaming Native**: Integrates with pydantic-flow's streaming vocabulary
- **Pluggable**: Easy to add new providers, stores, and loaders

## Architecture

The RAG layer consists of five main components:

### 1. Documents (`pydantic_flow.rag.docs`)

Core data models for RAG operations:

```python
from pydantic_flow.rag.docs import Document, Metadata

metadata = Metadata(
    source="article.md",
    chunk_index=0,
    total_chunks=5,
)

document = Document(
    id="doc_001",
    content="pydantic-flow is a type-safe AI agent framework...",
    metadata=metadata,
)
```

### 2. Embeddings (`pydantic_flow.rag.embeddings`)

Abstract interface for embedding providers:

```python
from pydantic_flow.rag.embeddings import EmbeddingProvider

class EmbeddingProvider(ABC):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts."""
        ...
    
    def dim(self) -> int:
        """Return embedding dimension."""
        ...
```

**Built-in Providers:**

- **OpenAIEmbeddings**: Uses AsyncOpenAI client
- **CohereEmbeddings**: Cohere API via httpx
- **HuggingFaceEmbeddings**: sentence-transformers models
- **OllamaEmbeddings**: Local Ollama server

Example:

```python
from pydantic_flow.rag.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536
)

vectors = await embeddings.embed(["hello", "world"])
print(f"Dimension: {embeddings.dim()}")
```

#### Vector Stores

Three implementations of `VectorStore`:

- **HNSWMemoryStore**: Fast in-memory HNSW (Hierarchical Navigable Small World) index
- **PostgresPGVectorStore**: PostgreSQL-backed store with pgvector extension

Example:

```python
from pydantic_flow.rag.vectors import HNSWMemoryStore

store = HNSWMemoryStore(dim=1536, max_elements=10000)

await store.upsert([
    ("doc1", vector1, document1),
    ("doc2", vector2, document2),
])

results = await store.query(query_vector, k=5)
for result in results:
    print(f"Score: {result.score}, Content: {result.document.content}")
```

### 4. Retrievers (`pydantic_flow.rag.retrievers`)

Coordinate embeddings and vector stores for semantic search:

```python
from pydantic_flow.rag.retrievers import VectorRetriever

retriever = VectorRetriever(
    embedding_provider=embeddings,
    vector_store=store,
    default_k=5,
    filter={"source": "docs/"}
)

documents = await retriever.retrieve("What is pydantic-flow?", k=3)
```

### 5. Loaders (`pydantic_flow.rag.loaders`)

Load and chunk documents from various sources:

**FSLoader** - Filesystem with chunking:

```python
from pydantic_flow.rag.loaders import FSLoader

loader = FSLoader(
    path="docs/",
    chunk_size=1000,
    chunk_overlap=200,
    extensions=[".md", ".txt"]
)

documents = await loader.load()
```

**WebLoader** - HTTP/HTTPS with text extraction:

```python
from pydantic_flow.rag.loaders import WebLoader

loader = WebLoader(
    url="https://example.com/article",
    chunk_size=1000
)

documents = await loader.load()
```

## Flow Integration

### VectorRetrieverNode

Streams `RetrievalItem` events for each retrieved document:

```python
from pydantic_flow import Flow
from pydantic_flow.rag.nodes import VectorRetrieverNode
from pydantic_flow.rag.nodes.retriever import QueryInput

retriever_node = VectorRetrieverNode(
    retriever=retriever,
    name="semantic-search"
)

# Streaming usage
query = QueryInput(query="AI agents", k=3)
async for item in retriever_node.astream(query):
    if isinstance(item, RetrievalItem):
        print(f"Retrieved: {item.content[:50]}...")

# Non-streaming usage
result = await retriever_node.run(query)
print(f"Found {len(result.documents)} documents")
```

### EmbeddingNode

Materializes embeddings for downstream use:

```python
from pydantic_flow.rag.nodes import EmbeddingNode
from pydantic_flow.rag.nodes.embedding import EmbeddingInput

embedding_node = EmbeddingNode(
    embedding_provider=embeddings,
    name="embed-texts"
)

input_data = EmbeddingInput(texts=["hello", "world"])
result = await embedding_node.run(input_data)
print(f"Generated {len(result.embeddings)} embeddings of dim {result.dimensions}")
```

### Complete Flow Example

```python
from pydantic_flow import Flow
from pydantic_flow.nodes import PromptNode
from pydantic_flow.rag.embeddings import OpenAIEmbeddings
from pydantic_flow.rag.loaders import FSLoader
from pydantic_flow.rag.nodes import VectorRetrieverNode
from pydantic_flow.rag.retrievers import VectorRetriever
from pydantic_flow.rag.vectors import HNSWMemoryStore

# 1. Ingest documents
loader = FSLoader(path="docs/", chunk_size=1000)
documents = await loader.load()

# 2. Setup RAG pipeline
embeddings = OpenAIEmbeddings(dimensions=1536)
store = HNSWMemoryStore(dim=1536)

# Generate and upsert embeddings
for doc in documents:
    vecs = await embeddings.embed([doc.content])
    await store.upsert([(doc.id, vecs[0], doc)])

# 3. Create retriever
retriever = VectorRetriever(
    embedding_provider=embeddings,
    vector_store=store,
    default_k=3
)

# 4. Build flow
flow = Flow()

retriever_node = VectorRetrieverNode(
    retriever=retriever,
    name="retriever"
)

# Use retrieved context in prompt
prompt_node = PromptNode(
    prompt="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
    model="openai:gpt-4o",
    name="answer"
)

flow.add_node(retriever_node)
flow.add_node(prompt_node)
flow.add_edge(retriever_node.name, prompt_node.name)

# 5. Execute with streaming
compiled = flow.compile()

async for item in compiled.astream({
    "retriever": QueryInput(query="What is pydantic-flow?", k=3)
}):
    if isinstance(item, RetrievalItem):
        print(f"📄 Retrieved: {item.content[:80]}...")
    elif isinstance(item, TokenChunk):
        print(item.text, end="", flush=True)
```

## Adding New Components

### Custom Embedding Provider

```python
from pydantic_flow.rag.embeddings import EmbeddingProvider

class CustomEmbeddings(EmbeddingProvider):
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Load your model
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Your embedding logic
        return embeddings
    
    def dim(self) -> int:
        return 768
```

### Custom Vector Store

```python
from pydantic_flow.rag.vectors import VectorStore, SearchResult

class CustomVectorStore(VectorStore):
    async def upsert(self, items):
        # Your upsert logic
        pass
    
    async def delete(self, ids):
        # Your delete logic
        pass
    
    async def query(self, vector, k, filter=None):
        # Your search logic
        return results
    
    def embedding_dim(self):
        return self.dim
```

### Custom Loader

```python
from pydantic_flow.rag.loaders import Loader
from pydantic_flow.rag.docs import Document

class CustomLoader(Loader):
    async def load(self) -> list[Document]:
        # Your loading logic
        return documents
```

## Testing

Run RAG tests:

```bash
# Basic tests (mock providers, HNSW only)
pytest tests/rag/

# With optional dependencies
pytest tests/rag/ --run-optional
```

Optional test markers guard tests requiring:
- External API keys (OpenAI, Cohere)
- Heavy dependencies (sentence-transformers)
- External services (PostgreSQL with pgvector)

## Performance Tips

1. **Batch embeddings**: Always embed in batches rather than one at a time
2. **HNSW for speed**: Use HNSWMemoryStore for fast in-memory operations
3. **PGVector for persistence**: Use PostgresPGVectorStore when durability matters
4. **Chunk appropriately**: Balance chunk size vs retrieval granularity (500-1500 chars typical)
5. **Filter metadata**: Use metadata filters to reduce search space

## Dependencies

All RAG dependencies are included in pydantic-flow core:
- `hnswlib` - HNSW memory store
- `httpx` - HTTP client for API calls
- `openai` - OpenAI embeddings
- `sentence-transformers` - HuggingFace embeddings
- `asyncpg` + `pgvector` - PostgreSQL vector store

No additional installation required:

```bash
pip install pydantic-flow
```
