# Milestone 8 — Phase 1: Backend API

**Status:** ✅ Done
**Goal:** Extend the FastAPI backend with CORS, document management, and ingestion endpoints

---

## Overview

Phase 1 prepares the backend to support the web frontend. The existing M7 API only had 4 endpoints oriented to querying (health, query, query/stream, sources). The frontend needs endpoints to upload documents, list them, delete them, trigger ingestion, check ingestion status, and view public configuration.

## What was implemented

### CORS Middleware

Added `CORSMiddleware` to allow requests from the Vite dev server (`localhost:5173`):

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### New Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/documents/upload` | Upload files (multipart/form-data) to `data/documents/`. Validates supported extensions. |
| GET | `/documents` | List files with metadata (name, size, format, modification date). Filters unsupported formats. |
| DELETE | `/documents/{filename}` | Delete a document file. Returns 404 if not found. |
| POST | `/ingest` | Trigger ingestion of all documents or a specific file. Supports `force_reindex`. |
| GET | `/ingest/status` | Returns the summary of the last ingestion run. |
| GET | `/config` | Returns public configuration (provider, model, top_k, etc.). No API keys exposed. |

### New Pydantic Models

```python
class IngestRequest(BaseModel):
    file_path: Optional[str] = None  # None = ingest all
    force_reindex: bool = False

class DocumentInfo(BaseModel):
    filename: str
    format: str
    size_bytes: int
    modified_at: str

class UploadResponse(BaseModel):
    filename: str
    size_bytes: int
    message: str

class IngestResponse(BaseModel):
    processed: int
    skipped: int
    failed: int
    total_chunks: int
    results: list[dict]

class ConfigResponse(BaseModel):
    llm_provider: str
    llm_model: Optional[str]
    rag_top_k: int
    rag_min_similarity: float
    api_host: str
    api_port: int
```

### New Dependency

`python-multipart==0.0.22` — Required by FastAPI for `UploadFile` (multipart/form-data).

### Reused Components

- `IngestionPipeline.ingest_file()` / `ingest_directory()` from `src/ingestion/pipeline.py`
- `settings.documents_dir` from `config/settings.py`
- `SUPPORTED_EXTENSIONS` from `src/ingestion/config.py`

## Tests (15 new, 24 total API tests)

| Test | What it verifies |
|------|-----------------|
| Upload valid markdown | POST /documents/upload with .md file succeeds |
| Upload valid PDF | POST /documents/upload with .pdf file succeeds |
| Upload unsupported format | POST /documents/upload with .exe returns 400 |
| Upload .txt unsupported | POST /documents/upload with .txt returns 400 |
| List empty directory | GET /documents returns empty list |
| List with files | GET /documents returns metadata, filters unsupported formats |
| Delete existing file | DELETE /documents/{name} removes file |
| Delete missing file | DELETE /documents/{name} returns 404 |
| Ingest all directory | POST /ingest calls ingest_directory |
| Ingest specific file | POST /ingest with file_path calls ingest_file |
| Ingest file not found | POST /ingest with missing file returns 404 |
| Ingest pipeline not initialized | POST /ingest returns 503 |
| Status no prior ingestion | GET /ingest/status returns info message |
| Status after ingestion | GET /ingest/status returns last summary |
| Config returns public settings | GET /config returns settings without API keys |

## Files Modified

| File | Changes |
|------|---------|
| `src/api/server.py` | CORS middleware + 6 endpoints + 5 Pydantic models + IngestionPipeline init in lifespan |
| `tests/unit/test_api.py` | 15 new tests + updated fixtures for mock ingestion pipeline + temp documents dir |
| `requirements.txt` | Added `python-multipart==0.0.22` |

## Verification

```bash
pytest tests/unit/test_api.py -v    # 24 passed
pytest tests/unit/ -v               # 170 passed (all unit tests)
```

---

**Next:** [Phase 2 — Frontend Foundation](milestone-08-phase2-frontend-foundation.md)
