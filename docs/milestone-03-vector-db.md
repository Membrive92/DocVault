# Milestone 3: Vector Database (Qdrant)

**Status:** 🚧 Next
**Dependencies:** M1 (Foundation), M2 (Embeddings)
**Goal:** Integrate Qdrant vector database for efficient similarity search

---

## Overview

This milestone establishes the vector database layer using Qdrant, which will store and retrieve document embeddings efficiently. Qdrant provides HNSW (Hierarchical Navigable Small World) indexing for approximate nearest neighbor search at scale.

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
│                                                          │
│  - __init__(collection_name, embedding_dim)             │
│  - initialize_collection()                              │
│  - insert_vectors(vectors, metadata)                    │
│  - search_similar(query_vector, limit, filter)          │
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

## Implementation Plan

### Task 1: Install Qdrant Client
```bash
pip install qdrant-client
```

**Why qdrant-client?**
- Official Python client
- Supports both in-memory and server modes
- Clean async/sync API
- Type hints included

### Task 2: Create Database Configuration

**File:** `src/database/config.py`

```python
"""
Configuration for vector database.
"""

from __future__ import annotations

# Qdrant configuration
DEFAULT_COLLECTION_NAME = "docvault_documents"
VECTOR_SIZE = 384  # Must match embedding dimension from M2
DISTANCE_METRIC = "Cosine"  # Cosine similarity (others: Dot, Euclid)

# HNSW index parameters
HNSW_CONFIG = {
    "m": 16,  # Number of edges per node (trade-off: speed vs accuracy)
    "ef_construct": 100,  # Construction time search depth
}

# Storage configuration
STORAGE_PATH = "data/qdrant_storage"  # Local persistence
IN_MEMORY_MODE = False  # Set to True for testing
```

**HNSW Parameters Explained:**
- **m=16**: Each vector connected to 16 others. Higher = better recall, slower search.
- **ef_construct=100**: Build quality. Higher = better graph, slower indexing.

### Task 3: Create Abstract Vector Database Interface

**File:** `src/database/vector_database.py`

```python
"""
Abstract interface for vector database operations.

This allows swapping Qdrant for alternatives without changing downstream code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class VectorDatabase(ABC):
    """
    Abstract base class for vector database implementations.

    This interface defines the contract that any vector database
    must implement to work with DocVault.
    """

    @abstractmethod
    def initialize_collection(self) -> None:
        """Create or connect to the vector collection."""
        pass

    @abstractmethod
    def insert_vectors(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, Any]]
    ) -> None:
        """Insert vectors with metadata into the collection."""
        pass

    @abstractmethod
    def search_similar(
        self,
        query_vector: list[float],
        limit: int = 5,
        filter: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete_by_id(self, ids: list[str]) -> None:
        """Delete vectors by their IDs."""
        pass

    @abstractmethod
    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection."""
        pass
```

### Task 4: Implement Qdrant Database

**File:** `src/database/qdrant_database.py`

```python
"""
Qdrant implementation of the vector database interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from .config import (
    DEFAULT_COLLECTION_NAME,
    DISTANCE_METRIC,
    HNSW_CONFIG,
    IN_MEMORY_MODE,
    STORAGE_PATH,
    VECTOR_SIZE,
)
from .vector_database import VectorDatabase


logger = logging.getLogger(__name__)


class QdrantDatabase(VectorDatabase):
    """
    Qdrant implementation of vector database.

    Supports both in-memory mode (for testing) and persistent storage.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = VECTOR_SIZE,
        in_memory: bool = IN_MEMORY_MODE
    ) -> None:
        """
        Initialize Qdrant database connection.

        Args:
            collection_name: Name of the vector collection
            vector_size: Dimension of vectors (must match embeddings)
            in_memory: If True, use in-memory storage (testing only)
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.in_memory = in_memory

        logger.info(f"Initializing Qdrant database: {collection_name}")

        # Create client
        if in_memory:
            logger.info("Using in-memory mode")
            self.client = QdrantClient(":memory:")
        else:
            storage_path = Path(STORAGE_PATH)
            storage_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using persistent storage: {storage_path}")
            self.client = QdrantClient(path=str(storage_path))

        # Initialize collection
        self.initialize_collection()

    def initialize_collection(self) -> None:
        """Create collection if it doesn't exist."""
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_exists = any(
            col.name == self.collection_name for col in collections
        )

        if collection_exists:
            logger.info(f"Collection '{self.collection_name}' already exists")
            return

        # Create collection with HNSW index
        logger.info(f"Creating collection '{self.collection_name}'")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,  # Cosine similarity
            ),
            hnsw_config=HNSW_CONFIG,
        )

        logger.info("Collection created successfully")

    # ... (rest of implementation)
```

**Key Design Decisions:**
1. **In-memory mode**: For fast testing without disk I/O
2. **Cosine distance**: Matches L2-normalized embeddings from M2
3. **HNSW index**: Best performance for approximate search
4. **Lazy collection creation**: Only creates if doesn't exist

### Task 5: Unit Tests

**File:** `tests/test_vector_database.py`

Comprehensive tests including:
- Collection initialization
- Vector insertion (single and batch)
- Similarity search accuracy
- Metadata filtering
- Deletion operations
- In-memory vs persistent mode
- Error handling

### Task 6: Interactive Verification

**File:** `scripts/test_vector_db.py`

Interactive script that:
1. Creates in-memory Qdrant instance
2. Inserts sample embeddings from M2
3. Performs similarity searches
4. Displays results with scores
5. Tests metadata filtering
6. Verifies persistence (if not in-memory mode)

## Integration with M2 (Embeddings)

```python
from src.embeddings import EmbeddingService
from src.database import QdrantDatabase

# Initialize services
embedding_service = EmbeddingService()
vector_db = QdrantDatabase()

# Example: Index a document
text = "Machine learning is a subset of artificial intelligence"
embedding = embedding_service.generate_embedding(text)

vector_db.insert_vectors(
    ids=["doc_001"],
    vectors=[embedding],
    metadata=[{
        "text": text,
        "source": "example.pdf",
        "page": 1
    }]
)

# Example: Search
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

**File:** `.env`

Add:
```env
# Vector Database (Qdrant)
QDRANT_COLLECTION_NAME=docvault_documents
QDRANT_STORAGE_PATH=data/qdrant_storage
QDRANT_IN_MEMORY=False
```

**File:** `config/settings.py`

Add Qdrant settings to Pydantic model:
```python
# Qdrant settings
qdrant_collection_name: str = "docvault_documents"
qdrant_storage_path: Path = Path("data/qdrant_storage")
qdrant_in_memory: bool = False
```

## Verification Criteria

**M3 is complete when:**
- [ ] Qdrant client installed and working
- [ ] QdrantDatabase class implements VectorDatabase interface
- [ ] Collection creation works (in-memory and persistent)
- [ ] Vector insertion works (single and batch)
- [ ] Similarity search returns accurate results
- [ ] Metadata filtering works correctly
- [ ] All unit tests pass (pytest)
- [ ] Interactive verification script runs successfully
- [ ] Documentation updated (README.md, AGENTS.md)

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
- `tests/test_vector_database.py` - Unit tests
- `scripts/test_vector_db.py` - Interactive verification
