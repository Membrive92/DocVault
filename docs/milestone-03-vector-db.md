# Milestone 3: Vector Database (Qdrant)

**Status:** ✅ Completed
**Dependencies:** M1 (Foundation), M2 (Embeddings)
**Goal:** Integrate Qdrant vector database for efficient similarity search

---

## Overview

This milestone establishes the vector database layer using Qdrant, which stores and retrieves document embeddings efficiently. Qdrant provides HNSW (Hierarchical Navigable Small World) indexing for approximate nearest neighbor search at scale.

## Why Qdrant?

### Selected Features
- **Local-first**: Can run entirely in memory or with local persistence
- **Production-ready**: Scales from prototypes to millions of vectors
- **Rich filtering**: Metadata filtering combined with vector search
- **Multiple distance metrics**: Cosine similarity, Euclidean, Dot Product
- **Python-native client**: Clean API without complexity
- **No external dependencies**: Self-contained binary

### Alternatives Considered
- **FAISS**: Powerful but C++ focused, harder to integrate
- **Pinecone**: Cloud-only, not local-first
- **Weaviate**: Heavier footprint, more complex setup
- **Chroma**: Good alternative, but Qdrant has better performance at scale

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VectorDatabase                       │
│  (Abstract interface for vector operations)             │
└─────────────────────────────────────────────────────────┘
                           ▲
                           │
                           │ implements
                           │
┌─────────────────────────────────────────────────────────┐
│                   QdrantDatabase                        │
│                                                         │
│  - __init__(collection_name, vector_size, in_memory)    │
│  - initialize_collection()                              │
│  - insert_vectors(ids, vectors, metadata)               │
│  - search_similar(query_vector, limit, score_threshold) │
│  - delete_by_id(ids)                                    │
│  - get_collection_info()                                │
└─────────────────────────────────────────────────────────┘
                           │
                           │ uses
                           ▼
              ┌────────────────────────┐
              │   Qdrant Client        │
              │   (qdrant-client lib)  │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Qdrant Storage       │
              │  (data/qdrant_storage) │
              └────────────────────────┘
```

## Implementation Details

### Database Configuration (`src/database/config.py`)

```python
DEFAULT_COLLECTION_NAME = "docvault_documents"
VECTOR_SIZE = 384           # Must match embedding dimension from M2
DISTANCE_METRIC = "Cosine"  # Compatible with L2-normalized embeddings
HNSW_M = 16                 # Edges per node (speed vs accuracy)
HNSW_EF_CONSTRUCT = 100     # Construction time search depth
DEFAULT_STORAGE_PATH = "data/qdrant_storage"
```

**HNSW Parameters Explained:**
- **m=16**: Each vector connected to 16 others. Higher = better recall, slower search.
- **ef_construct=100**: Build quality. Higher = better graph, slower indexing.

### Abstract Interface (`src/database/vector_database.py`)

Strategy pattern interface allowing future database swaps:

```python
class VectorDatabase(ABC):
    @abstractmethod
    def initialize_collection(self) -> None: ...
    @abstractmethod
    def insert_vectors(self, ids, vectors, metadata) -> None: ...
    @abstractmethod
    def search_similar(self, query_vector, limit, score_threshold) -> list[dict]: ...
    @abstractmethod
    def delete_by_id(self, ids) -> None: ...
    @abstractmethod
    def get_collection_info(self) -> dict: ...
```

### Qdrant Implementation (`src/database/qdrant_database.py`)

Key design decisions:
1. **In-memory mode**: For fast testing without disk I/O
2. **Cosine distance**: Matches L2-normalized embeddings from M2
3. **HNSW index**: Best performance for approximate search
4. **Lazy collection creation**: Only creates if doesn't exist
5. **Validation**: Checks vector dimensions, input lengths before operations
6. **Error hierarchy**: `ValueError` for bad input, `RuntimeError` for system failures

### Two Storage Modes

- **In-memory** (`in_memory=True`): All data in RAM, lost on exit. Used for tests.
- **Persistent** (`in_memory=False`): Saved to `data/qdrant_storage/`. Survives restarts.

## Integration with M2 (Embeddings)

```python
from src.embeddings import EmbeddingService
from src.database import QdrantDatabase

# Initialize services
embedding_service = EmbeddingService()
vector_db = QdrantDatabase(in_memory=True)

# Index a document
text = "Machine learning is a subset of artificial intelligence"
embedding = embedding_service.generate_embedding(text)

vector_db.insert_vectors(
    ids=[str(uuid4())],       # Qdrant requires valid UUIDs for string IDs
    vectors=[embedding],
    metadata=[{"text": text, "source": "example.pdf", "page": 1}]
)

# Search
query = "What is AI?"
query_embedding = embedding_service.generate_embedding(query)
results = vector_db.search_similar(query_embedding, limit=5)
```

## Performance Considerations

### HNSW Index Tuning
- **m=16**: Good default for most use cases
- **ef_construct=100**: Balance between quality and speed
- Can be adjusted in config based on dataset size

### Memory Usage
- ~1KB per vector (384 float32 dimensions)
- 10,000 documents ≈ 10MB
- 1,000,000 documents ≈ 1GB

### Search Speed
- ~1ms for collections < 100K vectors
- ~10ms for collections < 1M vectors
- HNSW scales logarithmically

## Configuration Updates

**`.env.example`** — New Qdrant section:
```env
QDRANT_COLLECTION_NAME=docvault_documents
QDRANT_STORAGE_PATH=./data/qdrant_storage
QDRANT_IN_MEMORY=False
```

**`config/settings.py`** — New Qdrant fields:
```python
qdrant_collection_name: str = Field(default="docvault_documents")
qdrant_storage_path: Path = Field(default=Path("data/qdrant_storage"))
qdrant_in_memory: bool = Field(default=False)
```

## Testing

### Unit Tests (`tests/unit/test_vector_database.py`)

19 tests covering:
- Collection initialization (4 tests)
- Vector insertion with validation (5 tests)
- Similarity search and ordering (6 tests)
- Metadata preservation (1 test)
- Deletion operations (2 tests)
- Collection info (1 test)

```bash
pytest tests/unit/test_vector_database.py -v
```

### Integration Tests (`tests/integration/test_vector_db_integration.py`)

7 tests using real embeddings from M2:
- Insert real embeddings and verify count
- Semantic search (AI, DevOps, programming queries)
- Score threshold filtering
- Collection info after operations
- Delete after insert

```bash
pytest tests/integration/test_vector_db_integration.py -v
```

## Verification Criteria

**M3 is complete when:**
- [x] Qdrant client installed and working
- [x] QdrantDatabase class implements VectorDatabase interface
- [x] Collection creation works (in-memory and persistent)
- [x] Vector insertion works (single and batch)
- [x] Similarity search returns accurate results
- [x] Score threshold filters irrelevant results
- [x] All 19 unit tests pass
- [x] All 7 integration tests pass (real M2 + M3)
- [x] Documentation updated (README.md, AGENTS.md)

## Next Steps (M4)

With vector database ready, M4 will implement document parsers to extract text from:
- PDF files
- HTML pages
- Markdown documents

This will enable us to ingest real documentation in M5.

---

**Related Files:**
- `src/database/config.py` - Database configuration
- `src/database/vector_database.py` - Abstract interface
- `src/database/qdrant_database.py` - Qdrant implementation
- `src/database/__init__.py` - Module exports
- `tests/unit/test_vector_database.py` - Unit tests (19)
- `tests/integration/test_vector_db_integration.py` - Integration tests (7)
- `config/settings.py` - Qdrant settings added
- `.env.example` - Qdrant env variables added
