# DocVault - System Architecture

> **Last Updated:** 2026-02-12
> **Status:** All Milestones Completed (M1-M7) - Project Feature-Complete

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Layer-by-Layer Breakdown](#layer-by-layer-breakdown)
4. [Data Flow](#data-flow)
5. [Key Design Decisions](#key-design-decisions)
6. [Technology Stack](#technology-stack)
7. [Scalability Considerations](#scalability-considerations)

---

## Overview

DocVault is a **local-first RAG (Retrieval-Augmented Generation)** system designed for querying enterprise documentation across multiple projects. The system converts documents into searchable vector representations and uses AI to answer questions based on the indexed content.

### Core Principles

1. **Local-First**: Works 100% locally without external API dependencies
2. **Flexible Scalability**: Easy migration to commercial models or own server
3. **Multi-Project Support**: Handle documentation from multiple projects with context preservation
4. **Multi-Format**: Support PDFs, HTML, and Markdown documents
5. **Incremental Development**: Built milestone by milestone with verification at each step

---

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                               │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │   FastAPI       │  │   Streamlit     │  │   CLI Tool      │      │
│  │   REST API      │  │   Web UI        │  │   Terminal      │      │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘      │
└───────────┼──────────────────────┼──────────────────────┼─────────────┘
            │                      │                      │
            └──────────────────────┴──────────────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE LAYER                              │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  Query Processing → Retrieval → Context Assembly → Generation │   │
│  └───────────────────────────────────────────────────────────────┘   │
└───────────┬─────────────────────────────────────────────┬───────────────┘
            │                                             │
            ▼                                             ▼
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│     LLM PROVIDER LAYER          │    │     VECTOR SEARCH LAYER         │
│                                 │    │                                 │
│  ┌────────────────────────┐    │    │  ┌────────────────────────┐    │
│  │   LLM Factory          │    │    │  │   Qdrant Client        │    │
│  │   (Strategy Pattern)   │    │    │  │   (Vector Search)      │    │
│  └────────┬───────────────┘    │    │  └────────┬───────────────┘    │
│           │                     │    │           │                     │
│  ┌────────┴───────────────┐    │    │  ┌────────┴───────────────┐    │
│  │ • Ollama Local         │    │    │  │ • Similarity Search    │    │
│  │ • Ollama Server        │    │    │  │ • Metadata Filtering   │    │
│  │ • OpenAI API           │    │    │  │ • Collection Mgmt      │    │
│  │ • Anthropic API        │    │    │  └────────────────────────┘    │
│  └────────────────────────┘    │    └─────────────┬───────────────────┘
└─────────────────────────────────┘                  │
                                                     │
┌─────────────────────────────────────────────────────────────────────────┐
│                         EMBEDDING LAYER                                 │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │   EmbeddingService (sentence-transformers)                     │   │
│  │   • Model: paraphrase-multilingual-MiniLM-L12-v2               │   │
│  │   • Dimension: 384                                              │   │
│  │   • Languages: English + Spanish                                │   │
│  └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                   ▲
                                   │
┌─────────────────────────────────────────────────────────────────────────┐
│                      DOCUMENT PROCESSING LAYER                          │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │ PDF Parser   │  │ HTML Parser  │  │  MD Parser   │                │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                │
│         │                  │                  │                         │
│         └──────────────────┴──────────────────┘                         │
│                            │                                            │
│                 ┌──────────▼──────────┐                                │
│                 │  Text Extraction    │                                │
│                 │  & Preprocessing    │                                │
│                 └──────────┬──────────┘                                │
│                            │                                            │
│                 ┌──────────▼──────────┐                                │
│                 │  Chunking Strategy  │                                │
│                 │  (~500 tokens)      │                                │
│                 └─────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA STORAGE LAYER                              │
│                                                                         │
│  ┌──────────────────────┐         ┌──────────────────────┐            │
│  │   Qdrant Storage     │         │   Document Store     │            │
│  │   (Vectors)          │         │   (Original Files)   │            │
│  │  ./data/qdrant/      │         │  ./data/documents/   │            │
│  └──────────────────────┘         └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Layer-by-Layer Breakdown

### 1. Application Layer (M7) ✅

**Purpose:** User-facing interfaces

**Components:**
- **FastAPI REST API:** Programmatic access with 4 endpoints
  - `GET /health` — Health check with pipeline status
  - `POST /query` — RAG query with answer + sources
  - `POST /query/stream` — Streaming LLM response (text/plain)
  - `GET /sources` — Indexed collection info
- **Interactive CLI:** Terminal REPL with rich formatting (panels, markdown, colors)
  - Commands: `/sources`, `/help`, `/exit`
  - Lazy pipeline initialization
  - Score-colored source citations

**Key Features:**
- Pydantic request/response validation with Field constraints
- Lifespan management for pipeline initialization
- StreamingResponse for real-time LLM output
- 503 Service Unavailable when pipeline not initialized

---

### 2. RAG Pipeline Layer (M7) ✅

**Purpose:** Orchestrate the retrieval and generation process

**Architecture:**
```python
class RAGPipeline:
    def __init__(self, embedding_service, vector_db, llm_provider, config)
    def query(query_text, top_k, temperature, max_tokens, streaming) -> RAGResponse | Iterator[str]
    def get_indexed_sources() -> dict
```

**Workflow:**
```
User Query → Embedding (M2) → Vector Search (M3) → Context Assembly → LLM Generation (M6) → RAGResponse
```

**Components:**
- **RAGPipeline:** Main orchestrator with dependency injection
- **Source:** Retrieved chunk with metadata and similarity score
- **RAGResponse:** Answer + sources + timing metadata
- **RAGConfig:** Pipeline configuration (top_k, temperature, max_tokens, min_similarity)

---

### 3. LLM Provider Layer (M6) ✅

**Purpose:** Flexible LLM abstraction using Strategy Pattern

**Architecture:**
```python
# Abstract interface
class LLMProvider(ABC):
    def generate(self, prompt: str, context: str, temperature: float, max_tokens: int) -> str
    def generate_stream(self, prompt: str, context: str, ...) -> Iterator[str]
    def get_model_info(self) -> dict[str, str]
    def format_prompt_with_context(self, prompt: str, context: str) -> str  # concrete

# Implementations
- OllamaProvider     # localhost:11434 or custom server URL
- OpenAIProvider     # OpenAI API (GPT-4, GPT-3.5)
- AnthropicProvider  # Anthropic API (Claude models)

# Factory
LLMProviderFactory.create_provider(provider_type) → LLMProvider
```

**Key Design Decision:**
- **No vendor lock-in:** Switch providers with a single config change
- **Sync + streaming:** Both `generate()` and `generate_stream()` for all providers
- **RAG-ready:** Built-in `format_prompt_with_context()` template
- **Cost flexibility:** Start free (local), scale as needed

---

### 4. Vector Search Layer (M3) ✅

**Purpose:** Fast semantic similarity search

**Technology:** Qdrant Vector Database

**Capabilities:**
- **HNSW Index:** Fast approximate nearest neighbor search
- **Metadata Filtering:** Filter by project, date, type, etc.
- **Collection Management:** Separate collections per project
- **Persistence:** Local storage in `./data/qdrant_storage/`

**Search Process:**
```
Query Text → Embedding (384 dims) → Qdrant Search (Top-K) → Relevant Chunks
```

**Performance:**
- Search latency: ~10-50ms
- Throughput: ~1000 queries/sec
- Storage: ~1KB per document chunk

---

### 5. Embedding Layer (M2) ✅

**Purpose:** Convert text to vector representations

**Technology:** sentence-transformers

**Model:** `paraphrase-multilingual-MiniLM-L12-v2`
- **Dimensions:** 384
- **Languages:** English + Spanish
- **Size:** ~120MB
- **Speed:** ~50ms per embedding (CPU)

**Key Features:**
- **Batch processing:** Process multiple texts efficiently
- **L2 normalization:** Optimized for cosine similarity
- **Local execution:** No API calls, fully private
- **Caching:** Model cached after first load

**API:**
```python
service = EmbeddingService()

# Single text
embedding = service.generate_embedding("Hello world")
# → [0.123, -0.456, ..., 0.789]  # 384 floats

# Batch (efficient)
embeddings = service.generate_batch_embeddings(
    ["text1", "text2", "text3"],
    batch_size=32
)
```

---

### 6. Document Processing Layer (M4-M5)

**Purpose:** Extract and prepare text from various document formats

**Components:**

#### M4: Parsers ✅
- **PDF Parser:** Extract text from PDFs (pypdf) — page-by-page extraction with metadata
- **HTML Parser:** Extract content from web pages (BeautifulSoup4 + lxml) — boilerplate removal
- **Markdown Parser:** Parse Markdown documents (python-frontmatter) — YAML frontmatter support
- **Parser Factory:** Automatic format detection by file extension
- **ParsedDocument:** Standard output dataclass for all parsers

**Features:**
- Preserve document structure (headings, sections)
- Extract metadata (title, author, date, page count)
- Handle special characters and encodings
- Remove HTML boilerplate (scripts, nav, sidebar, ads)
- YAML frontmatter extraction for Markdown

#### M5: Ingestion Pipeline ✅
- **TextChunker:** Paragraph-first chunking (~500 tokens, 50 overlap)
- **IngestionStateManager:** JSON-based incremental indexing with mtime detection
- **IngestionPipeline:** End-to-end orchestrator (parse → chunk → embed → store)
- **Deterministic UUIDs:** uuid5 for chunk IDs, enabling safe re-indexing
- **File Discovery:** Extension filtering, skip patterns, recursive scanning

**Chunking Strategy:**
```
Document (10,000 tokens)
    ↓
Chunk 1: tokens   0-500  (50 overlap) → Embedding 1
Chunk 2: tokens 450-950  (50 overlap) → Embedding 2
Chunk 3: tokens 900-1400 (50 overlap) → Embedding 3
...
```

**Why overlap?**
- Prevent context loss at chunk boundaries
- Improve retrieval of information spanning multiple chunks

---

### 7. Data Storage Layer (M3-M5)

**Purpose:** Persist documents and vectors

**Components:**

1. **Qdrant Vector Storage** (M3)
   - Location: `./data/qdrant_storage/`
   - Format: Optimized binary format (HNSW index)
   - Content: Embeddings (384 dims) + metadata
   - Gitignored: Can be regenerated from documents

2. **Document Storage** (M4-M5)
   - Location: `./data/documents/`
   - Format: Original files (PDF, HTML, MD)
   - Structure: Organized by project
   - Versioned: Keep original documents in git (if small)

---

## Data Flow

### Ingestion Flow (Documents → Vector DB)

```
┌─────────────────┐
│ PDF/HTML/MD     │ User places documents in data/documents/
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ [M4] Parser     │ Extract text based on file type
│ • PDFParser     │
│ • HTMLParser    │
│ • MDParser      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ [M5] Chunker    │ Split into ~500 token chunks with 50 token overlap
│ • Tokenization  │
│ • Overlap       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ [M2] Embeddings │ Generate 384-dim vectors
│ EmbeddingService│ Batch processing for efficiency
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ [M3] Qdrant     │ Store vectors + metadata
│ Insert vectors  │ Create HNSW index
└─────────────────┘
```

**Example:**
```python
# M5: Ingestion script
python scripts/ingest_documents.py data/documents/project1/ --collection "project1"

# Process:
1. Find all PDF/HTML/MD files
2. Parse each file → extract text
3. Chunk text → [chunk1, chunk2, ..., chunkN]
4. Generate embeddings → [emb1, emb2, ..., embN]
5. Insert to Qdrant with metadata
```

---

### Query Flow (User Question → Answer)

```
┌─────────────────┐
│ User Query      │ "How to install dependencies?"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ [M2] Embeddings │ Convert query to 384-dim vector
│ Query Embedding │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ [M3] Qdrant     │ Similarity search (cosine)
│ Vector Search   │ Return Top-K chunks (K=5)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ [M7] Context    │ Assemble retrieved chunks into context
│ Assembly        │ Add metadata, format for LLM
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ [M6] LLM        │ Generate answer using:
│ Generation      │ • User query
│                 │ • Retrieved context
│                 │ • System prompt
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Response        │ Answer + cited sources
└─────────────────┘
```

**Example:**
```python
# M7: Query script
python scripts/query_rag.py "How to install dependencies?"

# Process:
1. Query → Embedding: [0.1, 0.2, ...]
2. Qdrant search → Top 5 chunks
3. Assemble context from chunks
4. LLM generates answer using context
5. Return answer + sources
```

---

## Key Design Decisions

### 1. Local-First Architecture

**Decision:** Start with local execution, scale later

**Rationale:**
- **Cost:** $0 vs $0.13/1M tokens (OpenAI)
- **Privacy:** Enterprise docs never leave the machine
- **Latency:** No network calls for embeddings
- **Development:** Easier to develop and test locally

**Trade-off:**
- Quality: Local models < API models
- Hardware: Requires decent CPU/RAM
- **Acceptable** for enterprise RAG use case

---

### 2. Flexible LLM Layer (Strategy Pattern)

**Decision:** Abstract LLM providers behind common interface

**Rationale:**
- **No vendor lock-in:** Switch OpenAI → Anthropic → Ollama easily
- **Cost optimization:** Start local, upgrade to API as needed
- **Redundancy:** Fallback if one provider fails
- **A/B testing:** Compare different models easily

**Implementation:**
```python
# Config change only — .env file
LLM_PROVIDER=ollama_local  # or openai, anthropic, ollama_server

# Code stays the same
from src.llm import LLMProviderFactory

llm = LLMProviderFactory.create_provider()
response = llm.generate(prompt="What is Python?", context=retrieved_chunks)

# Streaming
for chunk in llm.generate_stream(prompt="Explain RAG", context=context):
    print(chunk, end="")
```

---

### 3. 384-Dimensional Embeddings

**Decision:** Use 384 dims instead of 768 or 1536

**Rationale:**
- **Performance:** 2x faster search than 768 dims
- **Storage:** 2x less RAM/disk than 768 dims
- **Quality:** Sufficient for RAG use case
- **Cost:** If we switch to API, 384 dims cost less

**Comparison:**
| Model | Dims | Quality | Speed | Cost |
|-------|------|---------|-------|------|
| MiniLM-L12 (ours) | 384 | Good | Fast | Free |
| BERT-base | 768 | Better | Medium | Free |
| text-embedding-3-large | 1536 | Best | Slow | $0.13/1M |

---

### 4. Chunking Strategy: 500 tokens + 50 overlap

**Decision:** Fixed-size chunks with small overlap

**Rationale:**
- **500 tokens:** Fits in LLM context window (~1/8 of 4K context)
- **50 token overlap:** Prevents context loss at boundaries
- **Fixed size:** Simpler than semantic chunking
- **Efficient:** Easy to implement and test

**Alternative approaches (not chosen):**
- Semantic chunking (by paragraphs/sections)
  - Pro: Natural boundaries
  - Con: Variable size, complex logic
- Smaller chunks (200 tokens)
  - Pro: More precise retrieval
  - Con: More chunks to search, less context per chunk
- Larger chunks (1000 tokens)
  - Pro: More context per chunk
  - Con: Less precise retrieval

---

### 5. Qdrant for Vector Database

**Decision:** Use Qdrant instead of Pinecone, Weaviate, etc.

**Rationale:**
- **Local-first:** Can run in Docker locally
- **Scalable:** Easy to move to cloud later
- **Performance:** Excellent HNSW implementation
- **Simple API:** Easy to use
- **Open source:** No vendor lock-in

**Alternatives (not chosen):**
- Pinecone: Cloud-only, monthly cost
- Weaviate: More complex setup
- Chroma: Less mature, fewer features
- FAISS: No metadata filtering, manual persistence

---

## Technology Stack

### Core Technologies

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Language | Python | 3.10+ | Modern type hints, async support |
| Config | Pydantic | 2.x | Type-safe configuration |
| Embeddings | sentence-transformers | 5.2+ | Local embedding generation |
| Vector DB | Qdrant | Latest | Vector similarity search |
| PDF Parsing | pypdf | 6.7+ | PDF text and metadata extraction |
| HTML Parsing | BeautifulSoup4 + lxml | 4.14+ | HTML content extraction |
| MD Parsing | python-frontmatter | 1.1+ | Markdown YAML frontmatter |
| LLM (local) | Ollama SDK | Latest | Local LLM inference |
| LLM (OpenAI) | openai SDK | Latest | OpenAI GPT models |
| LLM (Anthropic) | anthropic SDK | Latest | Anthropic Claude models |
| API | FastAPI | Latest | REST API endpoints |
| API Server | uvicorn | Latest | ASGI server for FastAPI |
| CLI | rich | Latest | Terminal formatting (panels, markdown) |
| HTTP Client | httpx | Latest | Required by FastAPI TestClient |
| Testing | pytest | Latest | Unit and integration tests |

### Development Tools

- **Type checking:** mypy (implicit via Pydantic)
- **Linting:** ruff (planned)
- **Formatting:** black (planned)
- **Documentation:** Markdown + docstrings
- **CI/CD:** GitHub Actions (planned)

---

## Scalability Considerations

### Horizontal Scaling Path

```
Phase 1: Single Machine (Current)
┌────────────────────────────┐
│  All-in-One Server         │
│  • FastAPI                 │
│  • Qdrant (Docker)         │
│  • Ollama (Local)          │
└────────────────────────────┘
Capacity: ~1000 docs, ~10 req/s


Phase 2: Distributed Components
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  API Server │  │   Qdrant    │  │   Ollama    │
│  (FastAPI)  │→│  (Cloud)    │  │  (Server)   │
└─────────────┘  └─────────────┘  └─────────────┘
Capacity: ~100K docs, ~100 req/s


Phase 3: Multi-Region
┌─────────────────────────────────────────┐
│         Load Balancer                   │
└──────┬─────────┬─────────┬──────────────┘
       │         │         │
   ┌───▼───┐ ┌──▼───┐ ┌──▼───┐
   │ API 1 │ │ API 2│ │ API 3│
   └───┬───┘ └───┬──┘ └──┬───┘
       └─────────┼────────┘
                 ▼
        ┌────────────────┐
        │ Qdrant Cluster │
        │  (Replicated)  │
        └────────────────┘
Capacity: ~1M docs, ~1000 req/s
```

### Performance Targets

| Metric | Phase 1 (Current) | Phase 2 | Phase 3 |
|--------|-------------------|---------|---------|
| Documents | 1K | 100K | 1M |
| Queries/sec | 10 | 100 | 1000 |
| Query latency (p95) | 500ms | 200ms | 100ms |
| Ingestion speed | 10 docs/min | 100 docs/min | 1000 docs/min |

### Cost Considerations

**Phase 1 (Local):**
- Hardware: 4GB RAM, 4 CPU cores
- Cost: $0/month (using existing hardware)
- Limits: ~1K documents, ~10 queries/sec

**Phase 2 (Cloud):**
- Qdrant Cloud: ~$50-100/month
- Ollama Server: ~$100-200/month (4 CPU + 16GB RAM)
- API Server: ~$20-50/month
- **Total:** ~$170-350/month
- Capacity: ~100K documents, ~100 queries/sec

**Phase 3 (Enterprise):**
- Qdrant Cluster: ~$500-1000/month
- OpenAI API: ~$500-2000/month (depending on volume)
- Load Balancer + Multi-region: ~$200-500/month
- **Total:** ~$1200-3500/month
- Capacity: ~1M documents, ~1000 queries/sec

---

## Next Steps

**Current Status:** All milestones completed (M1-M7). Project is feature-complete.

See individual milestone documents for detailed implementation:
- [Milestone 1: Foundation](milestone-01-foundation.md) ✅
- [Milestone 2: Embeddings](milestone-02-embeddings.md) ✅
- [Milestone 3: Vector DB](milestone-03-vector-db.md) ✅
- [Milestone 4: Parsers](milestone-04-parsers.md) ✅
- [Milestone 5: Ingestion](milestone-05-ingestion.md) ✅
- [Milestone 6: Flexible LLM](milestone-06-llm.md) ✅
- [Milestone 7: Complete RAG](milestone-07-rag.md) ✅
