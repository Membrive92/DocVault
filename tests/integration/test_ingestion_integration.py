"""
Integration tests for the ingestion pipeline (M2 + M3 + M4 + M5).

These tests use real services: sentence-transformers model, in-memory Qdrant,
and real document parsers. They verify the full end-to-end pipeline.

Run with: pytest tests/integration/test_ingestion_integration.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.database import QdrantDatabase
from src.embeddings import EmbeddingService
from src.ingestion import (
    IngestionPipeline,
    IngestionStateManager,
    IngestionSummary,
)


class TestIngestionIntegration:
    """Full integration tests: parse -> chunk -> embed -> store -> search."""

    @pytest.fixture(scope="class")
    def embedding_service(self) -> EmbeddingService:
        """Create real EmbeddingService (loads ML model once for class)."""
        return EmbeddingService()

    @pytest.fixture()
    def vector_db(self) -> QdrantDatabase:
        """Create real in-memory Qdrant database per test."""
        return QdrantDatabase(
            collection_name="ingestion_integration_test",
            in_memory=True,
        )

    @pytest.fixture()
    def pipeline(
        self,
        embedding_service: EmbeddingService,
        vector_db: QdrantDatabase,
        tmp_path: Path,
    ) -> IngestionPipeline:
        """Create pipeline with real services and temp state file."""
        pipe = IngestionPipeline(
            embedding_service=embedding_service,
            vector_db=vector_db,
        )
        pipe.state_manager = IngestionStateManager(
            state_file=tmp_path / "test_state.json"
        )
        return pipe

    def test_ingest_markdown_and_search(
        self,
        pipeline: IngestionPipeline,
        vector_db: QdrantDatabase,
        embedding_service: EmbeddingService,
        tmp_path: Path,
    ) -> None:
        """Ingest a markdown file and verify chunks are searchable."""
        md_content = """# Machine Learning Guide

Machine learning is a subset of artificial intelligence that focuses on
building systems that learn from data. These systems improve their performance
over time without being explicitly programmed for every scenario.

## Supervised Learning

Supervised learning is a type of machine learning where the model is trained
on labeled data. The algorithm learns a mapping function from input to output.
Common algorithms include linear regression, decision trees, and neural networks.

## Unsupervised Learning

Unsupervised learning deals with unlabeled data. The algorithm tries to find
hidden patterns or groupings. Clustering and dimensionality reduction are
common unsupervised techniques used in data analysis.

## Deep Learning

Deep learning uses neural networks with many layers to learn complex patterns.
Convolutional neural networks are used for image recognition while recurrent
neural networks handle sequential data like text and time series.
"""
        md_file = tmp_path / "ml_guide.md"
        md_file.write_text(md_content, encoding="utf-8")

        result = pipeline.ingest_file(md_file)

        assert result.status == "success"
        assert result.chunks_created > 0

        query = embedding_service.generate_embedding(
            "What is supervised learning?"
        )
        results = vector_db.search_similar(query_vector=query, limit=3)

        assert len(results) > 0
        top_text = results[0]["metadata"]["chunk_text"]
        assert "learning" in top_text.lower()

    def test_ingest_html_and_search(
        self,
        pipeline: IngestionPipeline,
        vector_db: QdrantDatabase,
        embedding_service: EmbeddingService,
        tmp_path: Path,
    ) -> None:
        """Ingest an HTML file and verify chunks are searchable."""
        html_content = """<!DOCTYPE html>
<html><head><title>Docker Guide</title></head>
<body>
<main>
<h1>Docker Containerization</h1>
<p>Docker is a platform for developing, shipping, and running applications
in containers. Containers package applications with their dependencies
ensuring consistent behavior across environments. Docker simplifies
deployment by creating lightweight, portable containers that include
everything needed to run software applications reliably.</p>
<p>Docker images are lightweight, standalone packages that include everything
needed to run a piece of software. They are built from Dockerfiles which
specify the base image, dependencies, and configuration steps needed to
create a reproducible application environment.</p>
</main>
</body></html>"""

        html_file = tmp_path / "docker.html"
        html_file.write_text(html_content, encoding="utf-8")

        result = pipeline.ingest_file(html_file)
        assert result.status == "success"

        query = embedding_service.generate_embedding(
            "How do Docker containers work?"
        )
        results = vector_db.search_similar(query_vector=query, limit=2)
        assert len(results) > 0

    def test_incremental_indexing_skips_unchanged(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """Second ingest of the same file is skipped (incremental)."""
        md_file = tmp_path / "incremental.md"
        md_file.write_text(
            "# Guide\n\n" + "Content paragraph with words. " * 100,
            encoding="utf-8",
        )

        result1 = pipeline.ingest_file(md_file)
        assert result1.status == "success"

        summary = pipeline.ingest_directory(tmp_path, force_reindex=False)
        assert summary.skipped >= 1

    def test_force_reindex_processes_again(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """force_reindex=True re-processes already-indexed files."""
        md_file = tmp_path / "reindex.md"
        md_file.write_text(
            "# Reindex\n\n" + "Content paragraph with words. " * 100,
            encoding="utf-8",
        )

        pipeline.ingest_file(md_file)

        summary = pipeline.ingest_directory(tmp_path, force_reindex=True)
        assert summary.processed >= 1
        assert summary.skipped == 0

    def test_metadata_stored_correctly(
        self,
        pipeline: IngestionPipeline,
        vector_db: QdrantDatabase,
        embedding_service: EmbeddingService,
        tmp_path: Path,
    ) -> None:
        """Verify that chunk metadata is correctly stored in Qdrant."""
        md_file = tmp_path / "metadata_test.md"
        md_file.write_text(
            "---\ntitle: Test Doc\n---\n\n"
            + "Paragraph content about testing metadata storage. " * 50,
            encoding="utf-8",
        )

        pipeline.ingest_file(md_file)

        query = embedding_service.generate_embedding("paragraph content")
        results = vector_db.search_similar(query_vector=query, limit=1)

        assert len(results) == 1
        meta = results[0]["metadata"]
        assert "chunk_text" in meta
        assert "source_file" in meta
        assert "chunk_index" in meta
        assert "total_chunks" in meta
        assert "document_format" in meta
        assert meta["document_format"] == "markdown"

    def test_directory_ingestion_summary(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """Ingest multiple files and verify summary counts."""
        (tmp_path / "file1.md").write_text(
            "# File One\n\n" + "First document content. " * 100,
            encoding="utf-8",
        )
        (tmp_path / "file2.md").write_text(
            "# File Two\n\n" + "Second document content. " * 100,
            encoding="utf-8",
        )
        (tmp_path / "ignored.txt").write_text(
            "This should be ignored", encoding="utf-8"
        )

        summary = pipeline.ingest_directory(tmp_path)

        assert isinstance(summary, IngestionSummary)
        assert summary.processed == 2
        assert summary.total_chunks > 0
        assert len(summary.results) == 2
