# pydantic-flow Examples

This directory contains example scripts demonstrating various features of the pydantic-flow framework.

## Available Examples

### 🔍 RAG (Retrieval-Augmented Generation)

#### Document Splitters and Reranking

**File:** `rag_splitters_simple.py`

Simple RAG pipeline demonstrating:
- Sentence-based document splitting with overlap
- Lexical reranking baseline
- MMR diversification to reduce redundancy
- Lambda comparison (relevance vs diversity balance)

**Run with:**
```bash
cd examples
uv run python rag_splitters_simple.py
```

**File:** `rag_markdown_docs.py`

Technical documentation RAG example:
- Markdown heading-based splitting
- Preserves heading hierarchy in metadata
- Query-relevant section retrieval
- MMR diversification across different sections

**Run with:**
```bash
cd examples
uv run python rag_markdown_docs.py
```

**File:** `rag_comprehensive.py`

Complete demonstration of all RAG features:
- All three splitter strategies (Token, Sentence, Markdown)
- Lexical reranking baseline
- MMR diversification
- Source-aware selection
- Side-by-side comparisons

**Run with:**
```bash
cd examples
uv run python rag_comprehensive.py
```

**Key concepts:**
- **TokenSplitter**: Approximate token-based chunking with overlap
- **SentenceSplitter**: Preserves sentence and paragraph boundaries
- **MarkdownHeadingSplitter**: Splits by ATX headings with hierarchy metadata
- **LexicalReranker**: Dependency-free TF-IDF baseline reranker
- **CohereReranker**: Optional Cohere Rerank API integration
- **mmr_select()**: Maximal Marginal Relevance for diversity
- **diversify_by_source()**: Source-aware selection constraints

#### Vector Search

**File:** `rag_simple.py`

Complete example of RAG pipeline with streaming:
- Load markdown documents from filesystem
- Generate embeddings (mock provider for demo)
- Upsert to HNSW in-memory vector store
- Query with streaming retrieval results
- Display both streaming and non-streaming modes

**Run with:**
```bash
cd examples
uv run python rag_simple.py
```

**Key concepts:**
- FSLoader for document ingestion and chunking
- HNSWMemoryStore for fast in-memory search
- VectorRetriever combining embeddings and store
- VectorRetrieverNode streaming RetrievalItem events

### 🧠 Memory Modes

**Files:** `memory_modes_simple.py`, `memory_modes_advanced.py`

Examples demonstrating FlowNode memory modes for controlling conversation history in hierarchical flows:

#### Simple Examples (`memory_modes_simple.py`)
Basic demonstrations of all three memory modes:
- **SHARED**: Sub-flow uses parent's memory directly (default)
- **ISOLATED**: Sub-flow gets separate memory (optionally seeded with parent's history)
- **READONLY**: Sub-flow can read but not modify parent's memory

**Run with:**
```bash
cd examples
uv run python memory_modes_simple.py
```

#### Advanced Examples (`memory_modes_advanced.py`)
Real-world use cases for each memory mode:
1. **Parallel Research** (ISOLATED): Multiple independent tasks without cross-contamination
2. **Background Enrichment** (READONLY): Enrich context without polluting main conversation
3. **Sequential Conversation** (SHARED): Multi-step workflows with full context sharing

**Run with:**
```bash
cd examples
uv run python memory_modes_advanced.py
```

**Key concepts:**
- Context variable propagation through async flows
- Memory isolation for parallel sub-flows
- Read-only access for background operations
- Seeding isolated memory with parent context

### 🔄 Human-in-the-Loop (HITL)

**Directory:** `HITL.md` | **Files:** `hitl_simple.py`, `hitl_complex.py`

Comprehensive examples of interrupt handlers, approval workflows, and priority-based execution:
- Conditional interruption based on progress events
- Multi-stage approval with priority handlers
- Security checks, risk analysis, compliance validation
- Checkpoint creation and metadata tracking

**See:** [HITL.md](HITL.md) for detailed documentation

### �📝 `example.py`
**Complete Weather Workflow Demo**

A comprehensive demonstration of the pydantic-flow framework featuring:
- Type-safe node composition with generics
- Multiple workflow patterns (API-based and LLM-based)
- Automatic dependency resolution
- Pydantic model validation
- DAG execution ordering

**Run with:**
```bash
cd examples
uv run python example.py
```

**Features demonstrated:**
- `ToolNode` for API calls
- `PromptNode` for LLM interactions
- `ParserNode` for data transformation
- Flow orchestration with multiple nodes
- Strongly typed input/output models

### 🔍 `type_safety_demo.py`
**Type Safety Demonstration**

Shows the improved type safety features in the Flow class:
- Generic type parameters `Flow[InputT, OutputT]`
- BaseModel constraints for inputs and outputs
- IDE auto-completion support
- Compile-time type checking

**Run with:**
```bash
cd examples
uv run python type_safety_demo.py
```

**Features demonstrated:**
- Strongly typed flow creation
- BaseModel output validation
- Type annotation patterns
- Type safety benefits

### 🏗️ `hierarchical_flows.py`
**Hierarchical Flow Architecture**

A sophisticated example demonstrating hierarchical flow composition using FlowNode:
- Multi-level flow architecture (Level 1: Sub-flows, Level 2: Master pipeline)
- Real-world AI content creation pipeline
- Reusable sub-flow components
- Complex workflow orchestration

**Run with:**
```bash
cd examples
uv run python hierarchical_flows.py
```

**Features demonstrated:**
- `FlowNode` for sub-flow composition
- 4-phase content creation pipeline (Research → Planning → Writing → Publishing)
- Individual sub-flow testing and isolation
- Hierarchical flow reusability across different content types
- Type-safe flow boundaries with complex data models
- Enterprise-level workflow patterns

**Architecture:**
```
Content Creation Pipeline
├── Research Flow (ContentRequest → ResearchResults)
│   ├── Research Data Gathering
│   └── Research Validation
├── Planning Flow (ResearchResults → PlanningResults)
│   ├── Content Outline Creation
│   └── Content Strategy Development
├── Writing Flow (PlanningResults → WritingResults)
│   ├── Draft Content Writing
│   └── Content Review & Editing
└── Publishing Flow (WritingResults → PublishingResults)
    ├── Content Formatting
    └── Publishing Assets Creation
```

### 🔄 `sub_flow_example.py`
**Sub-flow Composition Basics**

Basic demonstration of using FlowNode to compose sub-flows:
- Simple sub-flow wrapping
- Multi-level nested flows
- Sub-flow reusability patterns

**Run with:**
```bash
cd examples
uv run python sub_flow_example.py
```

**Features demonstrated:**
- Basic FlowNode usage
- Nested flow composition
- Sub-flow reuse across contexts

### 📋 `prompt_library_demo.py`
**Comprehensive Prompt Library Showcase**

A thorough demonstration of the prompt library capabilities:
- Multiple template formats (f-string, Jinja2, Mustache)
- Chat message templates with role-based formatting
- Output parsers for structured data extraction
- Template validation and variable collection
- Message join strategies for different LLM providers

**Run with:**
```bash
cd examples
python prompt_library_demo.py
```

**Features demonstrated:**
- `PromptTemplate` with f-string, Jinja2, and Mustache formats
- `ChatPromptTemplate` for multi-turn conversations
- `JsonModelParser` for validating JSON outputs against Pydantic models
- `DelimitedParser` for pipe-delimited text parsing
- `AsIsParser` for pass-through text handling
- Template variable collection and validation utilities
- `JoinStrategy` options (OpenAI, Anthropic, Simple)
- Type-safe input/output with Pydantic models
- Extra variable support beyond model definitions

## Getting Started

1. Navigate to the examples directory:
   ```bash
   cd examples
   ```

2. Run any example:
   ```bash
   uv run python example.py
   # or
   uv run python type_safety_demo.py
   ```

3. Examine the source code to understand how pydantic-flow components work together

## Requirements

All examples use the same dependencies as the main pydantic-flow project:

- Python 3.14+
- The pydantic-flow framework itself

The examples are designed to work out of the box with the project's existing environment setup.