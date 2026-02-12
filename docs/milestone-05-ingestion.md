# Milestone 5: Document Ingestion Pipeline

**Status:** ⏸️ Pending
**Dependencies:** M2 (Embeddings), M3 (Vector DB), M4 (Parsers)
**Goal:** Build end-to-end pipeline to ingest, chunk, embed, and index documents

---

## Overview

This milestone integrates all previous components into a complete document ingestion pipeline. It scans directories, parses documents, chunks text, generates embeddings, and stores them in the vector database.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      IngestionPipeline                          │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
      ┌────────────────────────────────────────┐
      │  1. DISCOVERY                          │
      │  - Scan document directories           │
      │  - Filter by extension                 │
      │  - Skip already indexed files          │
      └────────────────────────────────────────┘
                            │
                            ▼
      ┌────────────────────────────────────────┐
      │  2. PARSING                            │
      │  - Select appropriate parser (M4)      │
      │  - Extract text + metadata             │
      │  - Validate content                    │
      └────────────────────────────────────────┘
                            │
                            ▼
      ┌────────────────────────────────────────┐
      │  3. CHUNKING                           │
      │  - Split text into segments            │
      │  - ~500 tokens per chunk               │
      │  - 50 token overlap                    │
      │  - Preserve context                    │
      └────────────────────────────────────────┘
                            │
                            ▼
      ┌────────────────────────────────────────┐
      │  4. EMBEDDING                          │
      │  - Generate vectors (M2)               │
      │  - Batch processing                    │
      │  - Progress tracking                   │
      └────────────────────────────────────────┘
                            │
                            ▼
      ┌────────────────────────────────────────┐
      │  5. INDEXING                           │
      │  - Store in Qdrant (M3)                │
      │  - Include metadata                    │
      │  - Update index state                  │
      └────────────────────────────────────────┘
```

## Why Chunking?

### The Problem
- Documents are too long for single embeddings
- LLM context windows have limits
- Need focused, relevant chunks for RAG

### The Solution
- **Chunk size: ~500 tokens** (~375 words, ~2000 chars)
  - Small enough: Focused topics, fits in context
  - Large enough: Preserves meaning, good embeddings

- **Overlap: 50 tokens** (~40 words, ~200 chars)
  - Prevents information loss at chunk boundaries
  - Maintains context across splits

### Example
```
Document: 2000 tokens

Chunks:
[0-500]    "Introduction to machine learning..."
[450-950]  "...algorithms used in ML include..."  ← 50 token overlap
[900-1400] "...neural networks are a subset..."
[1350-1850] "...applications of deep learning..."
[1800-2000] "...conclusion and future work"
```

## Implementation Plan

### Task 1: Text Chunking Strategy

**File:** `src/ingestion/chunker.py`

```python
"""
Text chunking utilities for document ingestion.

Splits long documents into semantic chunks suitable for embedding.
"""

from __future__ import annotations

import re
from typing import List

from .config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


class TextChunker:
    """
    Splits text into overlapping chunks for embedding.

    Uses token-aware chunking to preserve semantic meaning.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE
    ) -> None:
        """
        Initialize chunker.

        Args:
            chunk_size: Target size in tokens (~500)
            chunk_overlap: Overlap between chunks in tokens (~50)
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Simple token approximation: ~4 chars per token
        # More accurate: use tiktoken, but adds dependency
        chars_per_token = 4
        chunk_chars = self.chunk_size * chars_per_token
        overlap_chars = self.chunk_overlap * chars_per_token

        # Split by paragraphs first (preserve natural breaks)
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for paragraph in paragraphs:
            para_length = len(paragraph)

            # If single paragraph exceeds chunk size, force split
            if para_length > chunk_chars * 2:
                # Add current chunk if any
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                temp_chunk = []
                temp_length = 0

                for sentence in sentences:
                    sent_length = len(sentence)

                    if temp_length + sent_length > chunk_chars:
                        if temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                        temp_chunk = [sentence]
                        temp_length = sent_length
                    else:
                        temp_chunk.append(sentence)
                        temp_length += sent_length

                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))

            # Normal case: add paragraph to current chunk
            elif current_length + para_length > chunk_chars:
                # Save current chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))

                # Start new chunk with overlap
                # Keep last paragraph for context
                if current_chunk and len(current_chunk[-1]) < overlap_chars * 2:
                    current_chunk = [current_chunk[-1], paragraph]
                    current_length = len(current_chunk[-1]) + para_length
                else:
                    current_chunk = [paragraph]
                    current_length = para_length
            else:
                current_chunk.append(paragraph)
                current_length += para_length

        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        # Filter out chunks that are too small
        min_chars = self.min_chunk_size * chars_per_token
        chunks = [c for c in chunks if len(c) >= min_chars]

        return chunks
```

**Key Design:**
- Paragraph-aware chunking (preserves natural breaks)
- Sentence splitting for large paragraphs
- Overlap maintains context
- Character-based approximation (4 chars ≈ 1 token)

### Task 2: Ingestion Configuration

**File:** `src/ingestion/config.py`

```python
"""
Configuration for document ingestion pipeline.
"""

from __future__ import annotations

# Chunking parameters
CHUNK_SIZE = 500          # Tokens per chunk
CHUNK_OVERLAP = 50        # Overlap between chunks
MIN_CHUNK_SIZE = 100      # Minimum chunk size

# Ingestion settings
BATCH_SIZE = 32           # Embeddings per batch
MAX_WORKERS = 4           # Parallel file processing
SHOW_PROGRESS = True      # Show progress bars

# File filtering
SUPPORTED_EXTENSIONS = [".pdf", ".html", ".htm", ".md", ".markdown"]
SKIP_HIDDEN_FILES = True  # Skip files starting with .
SKIP_PATTERNS = [         # Skip files matching these patterns
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
]

# State tracking
INDEX_STATE_FILE = "data/index_state.json"  # Track indexed files
```

### Task 3: Ingestion State Manager

**File:** `src/ingestion/state_manager.py`

```python
"""
Manages ingestion state to track which files have been indexed.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class IngestionStateManager:
    """
    Tracks which files have been indexed and their metadata.

    Prevents re-indexing unchanged files.
    """

    def __init__(self, state_file: str | Path) -> None:
        """
        Initialize state manager.

        Args:
            state_file: Path to JSON state file
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict[str, dict] = self._load_state()

    def _load_state(self) -> Dict[str, dict]:
        """Load state from file."""
        if not self.state_file.exists():
            return {}

        with open(self.state_file, 'r') as f:
            return json.load(f)

    def _save_state(self) -> None:
        """Save state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def is_indexed(self, file_path: str | Path) -> bool:
        """
        Check if file has been indexed.

        Also checks if file has been modified since last indexing.

        Args:
            file_path: Path to check

        Returns:
            True if file is already indexed and unchanged
        """
        path = Path(file_path)
        key = str(path.absolute())

        if key not in self.state:
            return False

        # Check if file modified since last index
        last_modified = path.stat().st_mtime
        indexed_mtime = self.state[key].get('mtime')

        return indexed_mtime == last_modified

    def mark_indexed(
        self,
        file_path: str | Path,
        chunk_count: int,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Mark file as indexed.

        Args:
            file_path: Path to file
            chunk_count: Number of chunks created
            metadata: Optional additional metadata
        """
        path = Path(file_path)
        key = str(path.absolute())

        self.state[key] = {
            'indexed_at': datetime.utcnow().isoformat(),
            'mtime': path.stat().st_mtime,
            'chunk_count': chunk_count,
            'metadata': metadata or {}
        }

        self._save_state()

    def remove_indexed(self, file_path: str | Path) -> None:
        """Remove file from indexed state."""
        key = str(Path(file_path).absolute())
        if key in self.state:
            del self.state[key]
            self._save_state()

    def get_stats(self) -> dict:
        """Get indexing statistics."""
        return {
            'total_files': len(self.state),
            'total_chunks': sum(s['chunk_count'] for s in self.state.values())
        }
```

### Task 4: Main Ingestion Pipeline

**File:** `src/ingestion/pipeline.py`

```python
"""
Main document ingestion pipeline.

Orchestrates parsing, chunking, embedding, and indexing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from src.database import QdrantDatabase
from src.embeddings import EmbeddingService
from src.parsers import ParserFactory

from .chunker import TextChunker
from .config import (
    BATCH_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    INDEX_STATE_FILE,
    SHOW_PROGRESS,
    SKIP_HIDDEN_FILES,
    SKIP_PATTERNS,
    SUPPORTED_EXTENSIONS,
)
from .state_manager import IngestionStateManager


logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Processes documents from parsing through indexing.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_db: Optional[QdrantDatabase] = None,
    ) -> None:
        """
        Initialize ingestion pipeline.

        Args:
            embedding_service: Service for generating embeddings
            vector_db: Vector database for storage
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_db = vector_db or QdrantDatabase()
        self.parser_factory = ParserFactory()
        self.chunker = TextChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.state_manager = IngestionStateManager(INDEX_STATE_FILE)

        logger.info("Ingestion pipeline initialized")

    def ingest_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        force_reindex: bool = False
    ) -> dict:
        """
        Ingest all supported documents from a directory.

        Args:
            directory: Path to directory
            recursive: Search subdirectories
            force_reindex: Re-index even if file already indexed

        Returns:
            Statistics about ingestion
        """
        dir_path = Path(directory)

        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Invalid directory: {directory}")

        logger.info(f"Starting ingestion from: {dir_path}")

        # Discover files
        files = self._discover_files(dir_path, recursive)
        logger.info(f"Found {len(files)} files to process")

        # Filter already indexed
        if not force_reindex:
            files = [f for f in files if not self.state_manager.is_indexed(f)]
            logger.info(f"{len(files)} files need indexing")

        # Process each file
        stats = {
            'processed': 0,
            'failed': 0,
            'total_chunks': 0,
            'skipped': 0
        }

        for file_path in files:
            try:
                result = self.ingest_file(file_path)
                stats['processed'] += 1
                stats['total_chunks'] += result['chunks']
                logger.info(f"✓ {file_path.name}: {result['chunks']} chunks")

            except Exception as e:
                stats['failed'] += 1
                logger.error(f"✗ {file_path.name}: {e}")

        logger.info(f"Ingestion complete: {stats}")
        return stats

    def ingest_file(self, file_path: str | Path) -> dict:
        """
        Ingest a single file.

        Args:
            file_path: Path to file

        Returns:
            Statistics for this file
        """
        path = Path(file_path)

        # 1. Parse document
        parsed_doc = self.parser_factory.parse(path)
        logger.debug(f"Parsed {path.name}: {parsed_doc.word_count} words")

        # 2. Chunk text
        chunks = self.chunker.chunk_text(parsed_doc.text)
        logger.debug(f"Created {len(chunks)} chunks")

        if not chunks:
            raise ValueError("No chunks created from document")

        # 3. Generate embeddings (batch)
        embeddings = self.embedding_service.generate_batch_embeddings(
            chunks,
            batch_size=BATCH_SIZE,
            show_progress=SHOW_PROGRESS
        )

        # 4. Prepare metadata for each chunk
        ids = [str(uuid4()) for _ in chunks]
        metadata_list = []

        for i, chunk_text in enumerate(chunks):
            metadata_list.append({
                'source_file': str(path),
                'chunk_index': i,
                'chunk_text': chunk_text,
                'document_title': parsed_doc.title,
                'document_format': parsed_doc.format,
                'total_chunks': len(chunks),
            })

        # 5. Insert into vector database
        self.vector_db.insert_vectors(
            ids=ids,
            vectors=embeddings,
            metadata=metadata_list
        )

        # 6. Update state
        self.state_manager.mark_indexed(
            path,
            chunk_count=len(chunks),
            metadata={
                'title': parsed_doc.title,
                'format': parsed_doc.format
            }
        )

        return {
            'chunks': len(chunks),
            'embeddings': len(embeddings)
        }

    def _discover_files(
        self,
        directory: Path,
        recursive: bool
    ) -> List[Path]:
        """Discover all supported files in directory."""
        pattern = "**/*" if recursive else "*"
        all_files = directory.glob(pattern)

        supported_files = []

        for file in all_files:
            # Skip if not a file
            if not file.is_file():
                continue

            # Skip hidden files
            if SKIP_HIDDEN_FILES and file.name.startswith('.'):
                continue

            # Skip by pattern
            if any(pattern in str(file) for pattern in SKIP_PATTERNS):
                continue

            # Check extension
            if file.suffix.lower() in SUPPORTED_EXTENSIONS:
                supported_files.append(file)

        return supported_files
```

### Task 5: CLI Interface

**File:** `src/ingestion/cli.py`

```python
"""
Command-line interface for document ingestion.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import IngestionPipeline


def main() -> None:
    """Run ingestion CLI."""
    parser = argparse.ArgumentParser(
        description="DocVault - Ingest documents into vector database"
    )

    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing documents to ingest'
    )

    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively search subdirectories'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-indexing of already indexed files'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Validate directory
    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)

    # Run ingestion
    print(f"\n📚 DocVault - Document Ingestion")
    print(f"Directory: {directory}")
    print(f"Recursive: {args.recursive}")
    print(f"Force: {args.force}\n")

    try:
        pipeline = IngestionPipeline()
        stats = pipeline.ingest_directory(
            directory,
            recursive=args.recursive,
            force_reindex=args.force
        )

        print(f"\n✅ Ingestion Complete!")
        print(f"   Processed: {stats['processed']} files")
        print(f"   Failed: {stats['failed']} files")
        print(f"   Total chunks: {stats['total_chunks']}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
```

### Task 6: Unit Tests

**File:** `tests/test_ingestion.py`

Test coverage:
- Text chunking (overlap, size validation)
- State management (tracking, modification detection)
- Pipeline integration (end-to-end)
- Error handling

### Task 7: Interactive Verification

**File:** `scripts/test_ingestion.py`

Verification script:
1. Create sample documents
2. Run ingestion pipeline
3. Verify chunks in database
4. Test search functionality
5. Display statistics

## Usage Examples

### Python API
```python
from src.ingestion import IngestionPipeline

# Initialize pipeline
pipeline = IngestionPipeline()

# Ingest a directory
stats = pipeline.ingest_directory(
    "data/documents",
    recursive=True,
    force_reindex=False
)

print(f"Indexed {stats['total_chunks']} chunks from {stats['processed']} files")
```

### Command Line
```bash
# Ingest documents
python -m src.ingestion.cli data/documents --recursive

# Force re-index
python -m src.ingestion.cli data/documents --force --verbose
```

## Performance Optimization

### Batch Processing
- Embeddings generated in batches of 32
- ~31x faster than sequential

### Parallel Processing
- Future: Multi-threaded file processing
- Process multiple files simultaneously

### Incremental Indexing
- State manager tracks indexed files
- Only re-index if file modified
- Saves time on large document sets

## Monitoring and Logging

### Progress Tracking
```
📚 DocVault - Document Ingestion
Directory: data/documents
Found 47 files to process
✓ getting-started.md: 12 chunks
✓ api-reference.pdf: 156 chunks
✓ architecture.html: 34 chunks
...
✅ Ingestion Complete!
   Processed: 47 files
   Total chunks: 1,247
```

### Logging Levels
- **INFO**: High-level progress
- **DEBUG**: Detailed chunking/embedding info
- **ERROR**: Failures with stack traces

## Next Steps (M6)

With ingestion complete, M6 will implement the flexible LLM layer:
- Strategy Pattern for provider abstraction
- Ollama (local), OpenAI, Anthropic implementations
- Prompt templates for RAG queries

---

**Related Files:**
- `src/ingestion/config.py` - Ingestion configuration
- `src/ingestion/chunker.py` - Text chunking
- `src/ingestion/state_manager.py` - State tracking
- `src/ingestion/pipeline.py` - Main pipeline
- `src/ingestion/cli.py` - Command-line interface
- `tests/test_ingestion.py` - Unit tests
- `scripts/test_ingestion.py` - Interactive verification
