"""
Unit tests for the ingestion pipeline module.

Tests chunking, state management, data models, and pipeline orchestration.
Uses mocks for EmbeddingService and QdrantDatabase to avoid loading ML models.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion import (
    ChunkMetadata,
    IngestionPipeline,
    IngestionResult,
    IngestionSummary,
    IngestionStateManager,
    TextChunker,
)


# ==========================================
# TestTextChunker
# ==========================================


class TestTextChunker:
    """Tests for the TextChunker class."""

    @pytest.fixture()
    def chunker(self) -> TextChunker:
        """Create a TextChunker with default settings."""
        return TextChunker()

    def test_empty_text_returns_empty_list(self, chunker: TextChunker) -> None:
        """Empty or whitespace-only text produces no chunks."""
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []
        assert chunker.chunk_text("\n\n\n") == []

    def test_short_text_below_min_size_returns_empty(
        self, chunker: TextChunker
    ) -> None:
        """Text shorter than min_chunk_size is discarded."""
        short_text = "Hello world."
        result = chunker.chunk_text(short_text)
        assert result == []

    def test_single_chunk_for_medium_text(self, chunker: TextChunker) -> None:
        """Text that fits in one chunk produces exactly one chunk."""
        text = "This is a test paragraph with enough words. " * 12
        result = chunker.chunk_text(text)
        assert len(result) == 1

    def test_multiple_chunks_for_long_text(self, chunker: TextChunker) -> None:
        """Long text produces multiple chunks."""
        paragraphs = [
            f"Paragraph {i}. " + "word " * 120 for i in range(10)
        ]
        text = "\n\n".join(paragraphs)
        result = chunker.chunk_text(text)
        assert len(result) > 1

    def test_chunks_respect_maximum_size(self, chunker: TextChunker) -> None:
        """No chunk should be dramatically larger than chunk_size."""
        paragraphs = ["Sentence number one. " * 25 for _ in range(20)]
        text = "\n\n".join(paragraphs)
        result = chunker.chunk_text(text)
        max_chars = chunker._chunk_chars * 3
        for chunk in result:
            assert len(chunk) <= max_chars

    def test_overlap_carries_context(self) -> None:
        """Consecutive chunks should share some content via overlap."""
        chunker = TextChunker(
            chunk_size=100, chunk_overlap=30, min_chunk_size=10
        )
        # Small paragraphs that fit within overlap threshold
        paragraphs = [
            f"Paragraph {i} here." for i in range(30)
        ]
        text = "\n\n".join(paragraphs)
        result = chunker.chunk_text(text)

        assert len(result) >= 2
        # At least one pair of consecutive chunks should share content
        found_overlap = False
        for i in range(len(result) - 1):
            parts_current = result[i].split("\n\n")
            last_part = parts_current[-1]
            if last_part in result[i + 1]:
                found_overlap = True
                break
        assert found_overlap

    def test_paragraph_boundaries_preserved(
        self, chunker: TextChunker
    ) -> None:
        """Chunks should split at paragraph boundaries."""
        para1 = (
            "First paragraph with enough content to be meaningful. " * 10
        )
        para2 = (
            "Second paragraph with different content entirely. " * 10
        )
        text = para1.strip() + "\n\n" + para2.strip()
        result = chunker.chunk_text(text)
        for chunk in result:
            assert not chunk.startswith(" ")

    def test_oversized_paragraph_splits_by_sentences(self) -> None:
        """A single paragraph exceeding 2x chunk size is split by sentences."""
        chunker = TextChunker(
            chunk_size=50, chunk_overlap=10, min_chunk_size=5
        )
        giant_para = (
            ". ".join(
                [
                    f"Sentence number {i} with some extra words"
                    for i in range(50)
                ]
            )
            + "."
        )
        result = chunker.chunk_text(giant_para)
        assert len(result) > 1

    def test_custom_parameters(self) -> None:
        """Custom chunk_size, overlap, and min_chunk_size work correctly."""
        chunker = TextChunker(
            chunk_size=200, chunk_overlap=20, min_chunk_size=50
        )
        assert chunker.chunk_size == 200
        assert chunker.chunk_overlap == 20
        assert chunker.min_chunk_size == 50

    def test_invalid_overlap_raises_error(self) -> None:
        """Overlap >= chunk_size raises ValueError."""
        with pytest.raises(
            ValueError, match="chunk_overlap must be less than chunk_size"
        ):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_negative_min_chunk_raises_error(self) -> None:
        """Negative min_chunk_size raises ValueError."""
        with pytest.raises(
            ValueError, match="min_chunk_size cannot be negative"
        ):
            TextChunker(min_chunk_size=-1)


# ==========================================
# TestIngestionStateManager
# ==========================================


class TestIngestionStateManager:
    """Tests for the IngestionStateManager class."""

    @pytest.fixture()
    def state_file(self, tmp_path: Path) -> Path:
        """Create a temporary state file path."""
        return tmp_path / "test_state.json"

    @pytest.fixture()
    def manager(self, state_file: Path) -> IngestionStateManager:
        """Create a state manager with temp file."""
        return IngestionStateManager(state_file=state_file)

    def test_new_file_is_not_indexed(
        self, manager: IngestionStateManager, tmp_path: Path
    ) -> None:
        """A file that was never indexed returns False."""
        test_file = tmp_path / "new_file.md"
        test_file.write_text("content", encoding="utf-8")
        assert manager.is_indexed(test_file) is False

    def test_mark_and_check_indexed(
        self, manager: IngestionStateManager, tmp_path: Path
    ) -> None:
        """After mark_indexed, is_indexed returns True."""
        test_file = tmp_path / "indexed.md"
        test_file.write_text("content", encoding="utf-8")
        manager.mark_indexed(test_file, chunk_count=5)
        assert manager.is_indexed(test_file) is True

    def test_modified_file_detected(
        self, manager: IngestionStateManager, tmp_path: Path
    ) -> None:
        """Modifying a file after indexing makes is_indexed return False."""
        test_file = tmp_path / "modified.md"
        test_file.write_text("original", encoding="utf-8")
        manager.mark_indexed(test_file, chunk_count=3)
        assert manager.is_indexed(test_file) is True

        time.sleep(0.1)
        test_file.write_text("modified content", encoding="utf-8")
        assert manager.is_indexed(test_file) is False

    def test_remove_indexed(
        self, manager: IngestionStateManager, tmp_path: Path
    ) -> None:
        """remove_indexed makes is_indexed return False."""
        test_file = tmp_path / "removable.md"
        test_file.write_text("content", encoding="utf-8")
        manager.mark_indexed(test_file, chunk_count=2)
        assert manager.is_indexed(test_file) is True
        manager.remove_indexed(test_file)
        assert manager.is_indexed(test_file) is False

    def test_get_stats(
        self, manager: IngestionStateManager, tmp_path: Path
    ) -> None:
        """get_stats returns correct totals."""
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("a", encoding="utf-8")
        f2.write_text("b", encoding="utf-8")
        manager.mark_indexed(f1, chunk_count=3)
        manager.mark_indexed(f2, chunk_count=7)
        stats = manager.get_stats()
        assert stats["total_files"] == 2
        assert stats["total_chunks"] == 10

    def test_state_persists_to_file(
        self, state_file: Path, tmp_path: Path
    ) -> None:
        """State is written to disk and survives manager re-creation."""
        test_file = tmp_path / "persist.md"
        test_file.write_text("content", encoding="utf-8")

        manager1 = IngestionStateManager(state_file=state_file)
        manager1.mark_indexed(test_file, chunk_count=4)

        manager2 = IngestionStateManager(state_file=state_file)
        assert manager2.is_indexed(test_file) is True
        assert manager2.get_stats()["total_chunks"] == 4

    def test_nonexistent_file_is_not_indexed(
        self, manager: IngestionStateManager
    ) -> None:
        """is_indexed returns False for files that don't exist on disk."""
        assert manager.is_indexed("/nonexistent/file.md") is False


# ==========================================
# TestModels
# ==========================================


class TestModels:
    """Tests for ingestion data models."""

    def test_chunk_metadata_creation(self) -> None:
        """ChunkMetadata stores all fields correctly."""
        meta = ChunkMetadata(
            chunk_text="Some text content here",
            source_file="/path/to/doc.pdf",
            chunk_index=0,
            total_chunks=5,
            document_title="My Document",
            document_format="pdf",
            chunked_at="2026-01-01T00:00:00",
        )
        assert meta.chunk_text == "Some text content here"
        assert meta.chunk_index == 0
        assert meta.total_chunks == 5
        assert meta.document_format == "pdf"

    def test_chunk_metadata_nullable_title(self) -> None:
        """ChunkMetadata accepts None for document_title."""
        meta = ChunkMetadata(
            chunk_text="text",
            source_file="/doc.md",
            chunk_index=0,
            total_chunks=1,
            document_title=None,
            document_format="markdown",
            chunked_at="2026-01-01T00:00:00",
        )
        assert meta.document_title is None

    def test_ingestion_result_success(self) -> None:
        """IngestionResult captures successful ingestion."""
        result = IngestionResult(
            file_path="/path/to/file.pdf",
            chunks_created=10,
            status="success",
        )
        assert result.status == "success"
        assert result.chunks_created == 10
        assert result.error is None

    def test_ingestion_result_failure(self) -> None:
        """IngestionResult captures failure with error message."""
        result = IngestionResult(
            file_path="/path/to/file.pdf",
            chunks_created=0,
            status="failed",
            error="Unsupported format",
        )
        assert result.status == "failed"
        assert result.error == "Unsupported format"

    def test_ingestion_summary_defaults(self) -> None:
        """IngestionSummary has sensible defaults."""
        summary = IngestionSummary()
        assert summary.processed == 0
        assert summary.skipped == 0
        assert summary.failed == 0
        assert summary.total_chunks == 0
        assert summary.results == []

    def test_ingestion_summary_accumulation(self) -> None:
        """IngestionSummary accepts custom values."""
        summary = IngestionSummary(
            processed=3, skipped=1, failed=1, total_chunks=25
        )
        assert summary.processed == 3
        assert summary.total_chunks == 25


# ==========================================
# TestIngestionPipeline
# ==========================================


class TestIngestionPipeline:
    """Tests for IngestionPipeline with mocked dependencies."""

    @pytest.fixture()
    def mock_embedding_service(self) -> MagicMock:
        """Create a mock EmbeddingService."""
        mock = MagicMock()
        mock.generate_batch_embeddings.return_value = [
            [0.1] * 384 for _ in range(20)
        ]
        return mock

    @pytest.fixture()
    def mock_vector_db(self) -> MagicMock:
        """Create a mock QdrantDatabase."""
        return MagicMock()

    @pytest.fixture()
    def pipeline(
        self,
        mock_embedding_service: MagicMock,
        mock_vector_db: MagicMock,
        tmp_path: Path,
    ) -> IngestionPipeline:
        """Create pipeline with mocked services."""
        with patch(
            "src.ingestion.pipeline.IngestionStateManager"
        ) as MockState:
            mock_state = MagicMock()
            mock_state.is_indexed.return_value = False
            MockState.return_value = mock_state

            pipe = IngestionPipeline(
                embedding_service=mock_embedding_service,
                vector_db=mock_vector_db,
            )
            pipe.state_manager = mock_state
            return pipe

    def test_ingest_file_calls_all_stages(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """ingest_file calls parse, chunk, embed, insert, and mark_indexed."""
        md_file = tmp_path / "test.md"
        md_file.write_text(
            "# Title\n\n" + "Content paragraph with words. " * 100,
            encoding="utf-8",
        )

        result = pipeline.ingest_file(md_file)

        assert result.status == "success"
        assert result.chunks_created > 0
        pipeline.embedding_service.generate_batch_embeddings.assert_called_once()
        pipeline.vector_db.insert_vectors.assert_called_once()
        pipeline.state_manager.mark_indexed.assert_called_once()

    def test_ingest_directory_processes_supported_files(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """ingest_directory discovers and processes supported files."""
        (tmp_path / "doc.md").write_text(
            "# Doc\n\n" + "Paragraph content. " * 100, encoding="utf-8"
        )
        (tmp_path / "notes.txt").write_text(
            "Plain text", encoding="utf-8"
        )

        summary = pipeline.ingest_directory(tmp_path)

        assert summary.processed >= 1
        assert isinstance(summary, IngestionSummary)

    def test_ingest_directory_skips_indexed_files(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """Files already indexed are skipped."""
        (tmp_path / "already.md").write_text(
            "# Already indexed\n\n" + "text " * 200, encoding="utf-8"
        )
        pipeline.state_manager.is_indexed.return_value = True

        summary = pipeline.ingest_directory(tmp_path, force_reindex=False)

        assert summary.skipped >= 1
        assert summary.processed == 0

    def test_ingest_directory_force_reindex(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """force_reindex=True processes already indexed files."""
        (tmp_path / "force.md").write_text(
            "# Force\n\n" + "content " * 200, encoding="utf-8"
        )
        pipeline.state_manager.is_indexed.return_value = True

        summary = pipeline.ingest_directory(tmp_path, force_reindex=True)

        assert summary.processed >= 1
        assert summary.skipped == 0

    def test_ingest_directory_invalid_path_raises(
        self, pipeline: IngestionPipeline
    ) -> None:
        """Passing a nonexistent directory raises ValueError."""
        with pytest.raises(ValueError, match="Invalid directory"):
            pipeline.ingest_directory("/nonexistent/directory")

    def test_ingest_directory_captures_failure(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """A file that fails parsing is captured in summary.failed."""
        bad_file = tmp_path / "corrupt.pdf"
        bad_file.write_bytes(b"not a real pdf")

        summary = pipeline.ingest_directory(tmp_path)

        assert summary.failed >= 1
        failed_results = [
            r for r in summary.results if r.status == "failed"
        ]
        assert len(failed_results) >= 1
        assert failed_results[0].error is not None
