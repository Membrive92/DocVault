"""
Document ingestion module.

Provides the complete pipeline for discovering, parsing, chunking,
embedding, and indexing documents into the vector database.
"""

from __future__ import annotations

from .chunker import TextChunker
from .models import ChunkMetadata, IngestionResult, IngestionSummary
from .pipeline import IngestionPipeline
from .state_manager import IngestionStateManager

__all__ = [
    "ChunkMetadata",
    "IngestionPipeline",
    "IngestionResult",
    "IngestionSummary",
    "IngestionStateManager",
    "TextChunker",
]
