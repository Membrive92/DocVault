"""
Main document ingestion pipeline.

Orchestrates the end-to-end flow: file discovery, parsing (M4),
chunking, embedding (M2), vector storage (M3), and state tracking.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import NAMESPACE_URL, uuid5

from src.database import QdrantDatabase
from src.embeddings import EmbeddingService
from src.parsers import ParserFactory

from .chunker import TextChunker
from .config import (
    BATCH_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SKIP_PATTERNS,
    SUPPORTED_EXTENSIONS,
)
from .models import ChunkMetadata, IngestionResult, IngestionSummary
from .state_manager import IngestionStateManager

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Coordinates all M2-M4 components to process documents from raw files
    into indexed, searchable vector embeddings in Qdrant.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_db: Optional[QdrantDatabase] = None,
    ) -> None:
        """
        Initialize the ingestion pipeline.

        Components can be injected for testing or created with defaults.

        Args:
            embedding_service: EmbeddingService instance. Creates default if None.
            vector_db: QdrantDatabase instance. Creates default if None.
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_db = vector_db or QdrantDatabase()
        self.parser_factory = ParserFactory()
        self.chunker = TextChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.state_manager = IngestionStateManager()

        logger.info("Ingestion pipeline initialized")

    def ingest_file(self, file_path: str | Path) -> IngestionResult:
        """
        Ingest a single file through the full pipeline.

        Steps:
        1. Parse document (M4 ParserFactory)
        2. Chunk text (TextChunker)
        3. Generate embeddings (M2 EmbeddingService)
        4. Store in vector DB (M3 QdrantDatabase)
        5. Update state (IngestionStateManager)

        Args:
            file_path: Path to the document file.

        Returns:
            IngestionResult with status and chunk count.

        Raises:
            ValueError: If file format is unsupported or produces no chunks.
            FileNotFoundError: If file doesn't exist.
            RuntimeError: If any pipeline stage fails.
        """
        path = Path(file_path)
        logger.info("Ingesting file: %s", path.name)

        # 1. Parse
        parsed_doc = self.parser_factory.parse(path)
        logger.debug("Parsed %s: %d words", path.name, parsed_doc.word_count)

        # 2. Chunk
        chunks = self.chunker.chunk_text(parsed_doc.text)
        if not chunks:
            raise ValueError(f"No valid chunks created from {path.name}")

        logger.debug("Created %d chunks from %s", len(chunks), path.name)

        # 3. Embed
        embeddings = self.embedding_service.generate_batch_embeddings(
            chunks, batch_size=BATCH_SIZE
        )

        # 4. Prepare IDs and metadata
        now = datetime.now(timezone.utc).isoformat()
        ids: list[str] = []
        metadata_list: list[dict] = []

        for i, chunk_text in enumerate(chunks):
            chunk_id = str(
                uuid5(NAMESPACE_URL, f"{path.resolve()}::chunk::{i}")
            )
            ids.append(chunk_id)

            chunk_meta = ChunkMetadata(
                chunk_text=chunk_text,
                source_file=str(path),
                chunk_index=i,
                total_chunks=len(chunks),
                document_title=parsed_doc.title,
                document_format=parsed_doc.format,
                chunked_at=now,
            )

            metadata_list.append({
                "chunk_text": chunk_meta.chunk_text,
                "source_file": chunk_meta.source_file,
                "chunk_index": chunk_meta.chunk_index,
                "total_chunks": chunk_meta.total_chunks,
                "document_title": chunk_meta.document_title,
                "document_format": chunk_meta.document_format,
                "chunked_at": chunk_meta.chunked_at,
            })

        # 5. Insert into vector DB
        self.vector_db.insert_vectors(
            ids=ids,
            vectors=embeddings,
            metadata=metadata_list,
        )

        # 6. Update state
        self.state_manager.mark_indexed(
            path,
            chunk_count=len(chunks),
            metadata={
                "title": parsed_doc.title,
                "format": parsed_doc.format,
            },
        )

        logger.info("Ingested %s: %d chunks", path.name, len(chunks))

        return IngestionResult(
            file_path=str(path),
            chunks_created=len(chunks),
            status="success",
        )

    def ingest_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        force_reindex: bool = False,
    ) -> IngestionSummary:
        """
        Ingest all supported documents from a directory.

        Args:
            directory: Path to the directory to scan.
            recursive: If True, search subdirectories recursively.
            force_reindex: If True, re-index files even if already indexed.

        Returns:
            IngestionSummary with per-file results and totals.

        Raises:
            ValueError: If directory doesn't exist or is not a directory.
        """
        dir_path = Path(directory)

        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Invalid directory: {directory}")

        logger.info("Starting directory ingestion: %s", dir_path)

        files = self._discover_files(dir_path, recursive)
        logger.info("Discovered %d supported files", len(files))

        summary = IngestionSummary()

        for file_path in files:
            if not force_reindex and self.state_manager.is_indexed(file_path):
                result = IngestionResult(
                    file_path=str(file_path),
                    chunks_created=0,
                    status="skipped",
                )
                summary.skipped += 1
                summary.results.append(result)
                logger.debug("Skipped (already indexed): %s", file_path.name)
                continue

            try:
                result = self.ingest_file(file_path)
                summary.processed += 1
                summary.total_chunks += result.chunks_created
                summary.results.append(result)

            except Exception as e:
                result = IngestionResult(
                    file_path=str(file_path),
                    chunks_created=0,
                    status="failed",
                    error=str(e),
                )
                summary.failed += 1
                summary.results.append(result)
                logger.error("Failed to ingest %s: %s", file_path.name, e)

        logger.info(
            "Ingestion complete: %d processed, %d skipped, %d failed, %d total chunks",
            summary.processed,
            summary.skipped,
            summary.failed,
            summary.total_chunks,
        )

        return summary

    def _discover_files(
        self, directory: Path, recursive: bool
    ) -> list[Path]:
        """
        Discover all supported files in a directory.

        Filters by supported extensions and skips hidden files and
        directories matching SKIP_PATTERNS.

        Args:
            directory: Root directory to scan.
            recursive: Whether to search subdirectories.

        Returns:
            Sorted list of file paths with supported extensions.
        """
        pattern = "**/*" if recursive else "*"
        supported_files = []

        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue

            if file_path.name.startswith("."):
                continue

            if any(skip in str(file_path) for skip in SKIP_PATTERNS):
                continue

            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                supported_files.append(file_path)

        supported_files.sort()
        return supported_files
