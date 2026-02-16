"""
Data models for the ingestion pipeline.

Defines structured results for chunking, file ingestion, and directory processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkMetadata:
    """
    Metadata stored per chunk in the Qdrant vector database.

    Attached to each vector point and used during RAG retrieval (M7)
    to reconstruct context without needing the original file.
    """

    chunk_text: str
    source_file: str
    chunk_index: int
    total_chunks: int
    document_title: Optional[str]
    document_format: str
    chunked_at: str


@dataclass
class IngestionResult:
    """
    Result of ingesting a single file.

    Attributes:
        file_path: Path to the processed file.
        chunks_created: Number of chunks created from this file.
        status: One of "success", "skipped", or "failed".
        error: Error message if status is "failed".
    """

    file_path: str
    chunks_created: int
    status: str
    error: Optional[str] = None


@dataclass
class IngestionSummary:
    """
    Summary result of ingesting a directory of documents.

    Attributes:
        processed: Number of files successfully ingested.
        skipped: Number of files skipped (already indexed).
        failed: Number of files that failed to process.
        total_chunks: Total number of chunks created across all files.
        results: Per-file ingestion results.
    """

    processed: int = 0
    skipped: int = 0
    failed: int = 0
    total_chunks: int = 0
    results: list[IngestionResult] = field(default_factory=list)
