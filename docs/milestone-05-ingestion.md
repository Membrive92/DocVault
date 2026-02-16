# Milestone 5: Document Ingestion Pipeline

**Status:** ✅ Completed
**Dependencies:** M2 (Embeddings), M3 (Vector DB), M4 (Parsers)
**Goal:** Build end-to-end pipeline to ingest, chunk, embed, and index documents

---

## Overview

This milestone integrates all previous components (M2 Embeddings, M3 Vector DB, M4 Parsers) into a complete document ingestion pipeline. It scans directories, parses documents, chunks text, generates embeddings, and stores them in the vector database with incremental indexing support.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      IngestionPipeline                          │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
      ┌────────────────────────────────────────┐
      │  1. DISCOVERY                          │
      │  - Scan directories (recursive)        │
      │  - Filter by extension (.pdf/.html/.md)│
      │  - Skip patterns (.git, __pycache__)   │
      │  - Skip already indexed (mtime check)  │
      └────────────────────────────────────────┘
                            │
                            ▼
      ┌────────────────────────────────────────┐
      │  2. PARSING (M4)                       │
      │  - ParserFactory.parse() → ParsedDoc   │
      │  - Extract text + metadata             │
      │  - Validate content                    │
      └────────────────────────────────────────┘
                            │
                            ▼
      ┌────────────────────────────────────────┐
      │  3. CHUNKING                           │
      │  - Paragraph-first splitting           │
      │  - ~500 tokens per chunk (2000 chars)  │
      │  - 50 token overlap (200 chars)        │
      │  - Sentence fallback for giant blocks  │
      │  - Filter below 100 tokens minimum     │
      └────────────────────────────────────────┘
                            │
                            ▼
      ┌────────────────────────────────────────┐
      │  4. EMBEDDING (M2)                     │
      │  - Batch processing (32 per batch)     │
      │  - 384-dim vectors (MiniLM-L12)        │
      │  - L2 normalized for cosine similarity │
      └────────────────────────────────────────┘
                            │
                            ▼
      ┌────────────────────────────────────────┐
      │  5. INDEXING (M3)                      │
      │  - Deterministic UUID5 per chunk       │
      │  - Store in Qdrant with metadata       │
      │  - Update state (JSON + mtime)         │
      └────────────────────────────────────────┘
```

## Implementation

### Module Structure

```
src/ingestion/
├── __init__.py           # Module exports
├── config.py             # Constants (chunk size, overlap, extensions)
├── models.py             # ChunkMetadata, IngestionResult, IngestionSummary
├── chunker.py            # TextChunker (paragraph-first + sentence fallback)
├── state_manager.py      # IngestionStateManager (JSON persistence + mtime)
└── pipeline.py           # IngestionPipeline (orchestrator)
```

### Key Components

#### TextChunker (`chunker.py`)
- Paragraph-first strategy: splits on `\n\n`, preserves natural boundaries
- Overlap: carries last paragraph of previous chunk to next chunk
- Sentence fallback: splits by `.!?` for paragraphs exceeding 2x chunk size
- Configurable: chunk_size, overlap, min_chunk_size
- Token approximation: 4 chars ≈ 1 token (no external tokenizer dependency)

#### IngestionStateManager (`state_manager.py`)
- JSON file tracking indexed files with modification time (mtime)
- `is_indexed()`: checks if file is indexed AND unchanged since last indexing
- `mark_indexed()`: records file path, mtime, chunk count, metadata
- `remove_indexed()`: un-marks a file
- Uses `path.resolve()` for canonical paths (handles symlinks on all platforms)

#### IngestionPipeline (`pipeline.py`)
- Dependency injection: accepts optional EmbeddingService and QdrantDatabase
- `ingest_file()`: single file through all 5 stages
- `ingest_directory()`: batch processing with skip/force options
- Deterministic UUID5 based on `file_path::chunk::index` (safe re-indexing)
- Per-file error handling: one failure doesn't abort the batch

#### Data Models (`models.py`)
- `ChunkMetadata`: stored in Qdrant per vector point (chunk_text, source_file, chunk_index, etc.)
- `IngestionResult`: per-file result (status: success/skipped/failed)
- `IngestionSummary`: directory-level totals (processed, skipped, failed, total_chunks)

### Configuration (`config.py`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `CHUNK_SIZE` | 500 tokens | Target chunk size |
| `CHUNK_OVERLAP` | 50 tokens | Overlap between consecutive chunks |
| `MIN_CHUNK_SIZE` | 100 tokens | Minimum chunk size (smaller discarded) |
| `CHARS_PER_TOKEN` | 4 | Token approximation factor |
| `BATCH_SIZE` | 32 | Embedding batch size |
| `SUPPORTED_EXTENSIONS` | .pdf .html .htm .md .markdown | File types to process |
| `SKIP_PATTERNS` | __pycache__ .git node_modules .venv | Directories to skip |
| `INDEX_STATE_FILE` | data/index_state.json | State persistence path |

## Usage

```python
from src.ingestion import IngestionPipeline

# Initialize pipeline (creates default services if not injected)
pipeline = IngestionPipeline()

# Ingest a single file
result = pipeline.ingest_file("data/documents/guide.md")
print(f"Created {result.chunks_created} chunks")

# Ingest a directory
summary = pipeline.ingest_directory(
    "data/documents",
    recursive=True,
    force_reindex=False
)
print(f"Processed: {summary.processed}, Skipped: {summary.skipped}")

# With dependency injection (for testing)
from src.database import QdrantDatabase
from src.embeddings import EmbeddingService

pipeline = IngestionPipeline(
    embedding_service=EmbeddingService(),
    vector_db=QdrantDatabase(in_memory=True),
)
```

## Testing

### Unit Tests (`tests/unit/test_ingestion.py`) — 30 tests

| Class | Tests | What's Tested |
|-------|-------|---------------|
| `TestTextChunker` | 11 | Empty/short/medium/long text, overlap, paragraph boundaries, sentence splitting, custom params, validation errors |
| `TestIngestionStateManager` | 7 | Index tracking, mtime detection, persistence, remove, stats |
| `TestModels` | 6 | ChunkMetadata, IngestionResult, IngestionSummary creation and defaults |
| `TestIngestionPipeline` | 6 | All stages called (mocked), file discovery, skip/force, errors captured |

### Integration Tests (`tests/integration/test_ingestion_integration.py`) — 6 tests

| Test | What's Verified |
|------|-----------------|
| `test_ingest_markdown_and_search` | MD → chunk → embed → store → semantic search returns relevant result |
| `test_ingest_html_and_search` | HTML → chunk → embed → store → search works |
| `test_incremental_indexing_skips_unchanged` | Second run skips already-indexed files |
| `test_force_reindex_processes_again` | force_reindex=True re-processes everything |
| `test_metadata_stored_correctly` | Qdrant metadata has all expected fields |
| `test_directory_ingestion_summary` | Multi-file directory returns correct summary counts |

### Running Tests

```bash
# M5 unit tests only (fast, no ML model)
pytest tests/unit/test_ingestion.py -v

# M5 integration tests (loads ML model)
pytest tests/integration/test_ingestion_integration.py -v

# All tests
pytest tests/ -v
```

## Design Decisions

### UUID5 Instead of UUID4
Using `uuid5(NAMESPACE_URL, f"{path.resolve()}::chunk::{i}")` generates deterministic IDs. The same file at the same path always produces the same UUIDs, enabling Qdrant to overwrite (upsert) on re-indexing instead of duplicating vectors.

### No CLI (Deferred to M7)
Per AGENTS.md: no API/CLI until M7. The pipeline is a Python API only, callable from tests and future M7 endpoints.

### Typed Return Values
Instead of raw `dict` returns, we use typed dataclasses (`IngestionResult`, `IngestionSummary`) for IDE autocomplete and runtime safety.

### State Uses path.resolve()
Canonical paths via `resolve()` handle symlinks and relative paths consistently across platforms (Windows backslashes, Unix forward slashes).

## Verification Criteria

- [x] TextChunker splits text into chunks of ~500 tokens with 50-token overlap
- [x] Paragraph boundaries are preserved during chunking
- [x] IngestionStateManager tracks indexed files with mtime
- [x] Modified files are detected and re-indexed
- [x] IngestionPipeline orchestrates M2+M3+M4 end-to-end
- [x] Deterministic UUID5 chunk IDs support re-indexing
- [x] File discovery filters by extension and skip patterns
- [x] Per-file error handling doesn't abort batch
- [x] 30 unit tests pass
- [x] 6 integration tests pass with real services
- [x] All 129 project tests pass

## Next Steps (M6)

With ingestion complete, M6 will implement the flexible LLM layer:
- Strategy Pattern for provider abstraction
- Ollama (local), OpenAI, Anthropic implementations
- Provider factory for easy switching
- Prompt templates for RAG queries

---

**Related Files:**
- `src/ingestion/config.py` — Pipeline configuration constants
- `src/ingestion/models.py` — ChunkMetadata, IngestionResult, IngestionSummary
- `src/ingestion/chunker.py` — TextChunker with paragraph-first strategy
- `src/ingestion/state_manager.py` — JSON-based index state tracking
- `src/ingestion/pipeline.py` — Main IngestionPipeline orchestrator
- `tests/unit/test_ingestion.py` — 30 unit tests
- `tests/integration/test_ingestion_integration.py` — 6 integration tests
