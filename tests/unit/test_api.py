"""
Unit tests for the FastAPI REST API.

Tests all endpoints using FastAPI TestClient with mocked RAG pipeline.
The lifespan creates the pipeline, so we patch the RAGPipeline constructor.

Tests cover:
- M7 endpoints: health, query, query/stream, sources
- M8 endpoints: documents/upload, documents, documents/{filename}, ingest, ingest/status, config
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.ingestion.models import IngestionResult, IngestionSummary
from src.rag import RAGResponse, Source
from src.rag.config import NO_RESULTS_MESSAGE


# ==========================================
# Fixtures
# ==========================================


@pytest.fixture
def mock_rag_response():
    """Standard RAG response for testing."""
    return RAGResponse(
        answer="Docker is a containerization platform.",
        sources=[
            Source(
                chunk_text="Docker helps with containers.",
                source_file="docker.md",
                chunk_index=0,
                similarity_score=0.92,
                document_title="Docker Guide",
                document_format="markdown",
            ),
        ],
        query="What is Docker?",
        model_used="test-model",
        retrieval_count=1,
        retrieval_time_ms=50.0,
        generation_time_ms=200.0,
    )


@pytest.fixture
def mock_pipeline(mock_rag_response):
    """Mock RAG pipeline."""
    pipeline = MagicMock()
    pipeline.query.return_value = mock_rag_response
    pipeline.get_indexed_sources.return_value = {
        "collection_name": "test_collection",
        "vectors_count": 100,
        "vector_size": 384,
        "distance_metric": "Cosine",
        "status": "green",
    }
    return pipeline


@pytest.fixture
def mock_ingestion_pipeline():
    """Mock ingestion pipeline."""
    pipeline = MagicMock()
    return pipeline


@pytest.fixture
def temp_documents_dir():
    """Create a temporary directory for document tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def client_with_pipeline(mock_pipeline, mock_ingestion_pipeline, temp_documents_dir):
    """TestClient with initialized mock pipelines via patched constructors."""
    import src.api.server as server_module

    with (
        patch("src.api.server.RAGPipeline", return_value=mock_pipeline),
        patch("src.api.server.IngestionPipeline", return_value=mock_ingestion_pipeline),
        patch("src.api.server.settings") as mock_settings,
    ):
        mock_settings.documents_dir = temp_documents_dir
        mock_settings.get_full_path.return_value = temp_documents_dir
        mock_settings.llm_provider = "ollama_local"
        mock_settings.llm_model = "llama3.2:3b"
        mock_settings.rag_top_k = 5
        mock_settings.rag_min_similarity = 0.3
        mock_settings.api_host = "0.0.0.0"
        mock_settings.api_port = 8000

        # Reset global state between tests
        server_module.last_ingestion_summary = None

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_pipeline, mock_ingestion_pipeline, temp_documents_dir


@pytest.fixture
def client_no_pipeline():
    """TestClient with no pipeline initialized (constructor raises)."""
    with (
        patch(
            "src.api.server.RAGPipeline",
            side_effect=RuntimeError("No pipeline for testing"),
        ),
        patch(
            "src.api.server.IngestionPipeline",
            side_effect=RuntimeError("No ingestion for testing"),
        ),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


# ==========================================
# TestHealthEndpoint
# ==========================================


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_check(self, client_with_pipeline) -> None:
        """Health endpoint returns healthy status."""
        client, _, _, _ = client_with_pipeline
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["pipeline_initialized"] is True

    def test_health_no_pipeline(self, client_no_pipeline) -> None:
        """Health endpoint reports pipeline not initialized."""
        response = client_no_pipeline.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["pipeline_initialized"] is False


# ==========================================
# TestQueryEndpoint
# ==========================================


class TestQueryEndpoint:
    """Tests for POST /query."""

    def test_query_success(self, client_with_pipeline) -> None:
        """Successful query returns answer and sources."""
        client, mock_pipe, _, _ = client_with_pipeline
        response = client.post(
            "/query",
            json={"query": "What is Docker?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Docker is a containerization platform."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source_file"] == "docker.md"
        assert data["model_used"] == "test-model"
        assert data["retrieval_count"] == 1

    def test_query_pipeline_not_initialized(self, client_no_pipeline) -> None:
        """Query returns 503 when pipeline is not initialized."""
        response = client_no_pipeline.post(
            "/query",
            json={"query": "What is Docker?"},
        )

        assert response.status_code == 503

    def test_query_error(self, client_with_pipeline) -> None:
        """Query returns 500 on pipeline error."""
        client, mock_pipe, _, _ = client_with_pipeline
        mock_pipe.query.side_effect = RuntimeError("LLM connection failed")

        response = client.post(
            "/query",
            json={"query": "What is Docker?"},
        )

        assert response.status_code == 500


# ==========================================
# TestStreamEndpoint
# ==========================================


class TestStreamEndpoint:
    """Tests for POST /query/stream."""

    def test_stream_success(self, client_with_pipeline) -> None:
        """Streaming query returns chunked response."""
        client, mock_pipe, _, _ = client_with_pipeline
        mock_pipe.query.return_value = iter(["Hello ", "world!"])

        response = client.post(
            "/query/stream",
            json={"query": "test query"},
        )

        assert response.status_code == 200
        assert response.text == "Hello world!"

    def test_stream_pipeline_not_initialized(self, client_no_pipeline) -> None:
        """Streaming returns 503 when pipeline is not initialized."""
        response = client_no_pipeline.post(
            "/query/stream",
            json={"query": "test query"},
        )

        assert response.status_code == 503


# ==========================================
# TestSourcesEndpoint
# ==========================================


class TestSourcesEndpoint:
    """Tests for GET /sources."""

    def test_sources_success(self, client_with_pipeline) -> None:
        """Sources endpoint returns collection info."""
        client, _, _, _ = client_with_pipeline
        response = client.get("/sources")

        assert response.status_code == 200
        data = response.json()
        assert data["sources"]["collection_name"] == "test_collection"
        assert data["sources"]["vectors_count"] == 100

    def test_sources_pipeline_not_initialized(self, client_no_pipeline) -> None:
        """Sources returns 503 when pipeline is not initialized."""
        response = client_no_pipeline.get("/sources")

        assert response.status_code == 503


# ==========================================
# TestUploadEndpoint
# ==========================================


class TestUploadEndpoint:
    """Tests for POST /documents/upload."""

    def test_upload_valid_markdown(self, client_with_pipeline) -> None:
        """Upload a valid markdown file succeeds."""
        client, _, _, docs_dir = client_with_pipeline
        response = client.post(
            "/documents/upload",
            files={"file": ("test.md", b"# Hello World\nThis is a test.", "text/markdown")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.md"
        assert data["size_bytes"] > 0
        assert "uploaded successfully" in data["message"]
        assert (docs_dir / "test.md").exists()

    def test_upload_valid_pdf(self, client_with_pipeline) -> None:
        """Upload a PDF file succeeds."""
        client, _, _, docs_dir = client_with_pipeline
        response = client.post(
            "/documents/upload",
            files={"file": ("document.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "document.pdf"

    def test_upload_unsupported_format(self, client_with_pipeline) -> None:
        """Upload an unsupported file format returns 400."""
        client, _, _, _ = client_with_pipeline
        response = client.post(
            "/documents/upload",
            files={"file": ("malware.exe", b"bad content", "application/octet-stream")},
        )

        assert response.status_code == 400
        assert "Unsupported format" in response.json()["detail"]

    def test_upload_txt_unsupported(self, client_with_pipeline) -> None:
        """Upload a .txt file returns 400 (not in supported extensions)."""
        client, _, _, _ = client_with_pipeline
        response = client.post(
            "/documents/upload",
            files={"file": ("notes.txt", b"some text", "text/plain")},
        )

        assert response.status_code == 400


# ==========================================
# TestDocumentsEndpoint
# ==========================================


class TestDocumentsEndpoint:
    """Tests for GET /documents."""

    def test_list_empty_directory(self, client_with_pipeline) -> None:
        """Listing an empty documents directory returns empty list."""
        client, _, _, _ = client_with_pipeline
        response = client.get("/documents")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_with_files(self, client_with_pipeline) -> None:
        """Listing directory with files returns metadata."""
        client, _, _, docs_dir = client_with_pipeline

        # Create some test files
        (docs_dir / "guide.md").write_text("# Guide\nContent here")
        (docs_dir / "manual.html").write_text("<html><body>Manual</body></html>")
        (docs_dir / "notes.txt").write_text("ignored")  # Not a supported format

        response = client.get("/documents")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2  # .txt should be filtered out

        filenames = [d["filename"] for d in data]
        assert "guide.md" in filenames
        assert "manual.html" in filenames
        assert "notes.txt" not in filenames

        # Verify metadata structure
        for doc in data:
            assert "filename" in doc
            assert "format" in doc
            assert "size_bytes" in doc
            assert "modified_at" in doc
            assert doc["size_bytes"] > 0


# ==========================================
# TestDeleteEndpoint
# ==========================================


class TestDeleteEndpoint:
    """Tests for DELETE /documents/{filename}."""

    def test_delete_existing_file(self, client_with_pipeline) -> None:
        """Deleting an existing file removes it."""
        client, _, _, docs_dir = client_with_pipeline

        # Create a test file
        test_file = docs_dir / "to_delete.md"
        test_file.write_text("# Delete me")
        assert test_file.exists()

        response = client.delete("/documents/to_delete.md")

        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]
        assert not test_file.exists()

    def test_delete_nonexistent_file(self, client_with_pipeline) -> None:
        """Deleting a non-existent file returns 404."""
        client, _, _, _ = client_with_pipeline
        response = client.delete("/documents/nonexistent.md")

        assert response.status_code == 404


# ==========================================
# TestIngestEndpoint
# ==========================================


class TestIngestEndpoint:
    """Tests for POST /ingest."""

    def test_ingest_all_directory(self, client_with_pipeline) -> None:
        """Ingesting all documents calls ingest_directory."""
        client, _, mock_ingest, docs_dir = client_with_pipeline

        mock_ingest.ingest_directory.return_value = IngestionSummary(
            processed=2, skipped=1, failed=0, total_chunks=15, results=[]
        )

        response = client.post("/ingest", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["processed"] == 2
        assert data["skipped"] == 1
        assert data["failed"] == 0
        assert data["total_chunks"] == 15
        mock_ingest.ingest_directory.assert_called_once()

    def test_ingest_specific_file(self, client_with_pipeline) -> None:
        """Ingesting a specific file calls ingest_file."""
        client, _, mock_ingest, docs_dir = client_with_pipeline

        # Create the file so it passes the existence check
        (docs_dir / "readme.md").write_text("# README")

        mock_ingest.ingest_file.return_value = IngestionResult(
            file_path=str(docs_dir / "readme.md"),
            chunks_created=5,
            status="success",
        )

        response = client.post("/ingest", json={"file_path": "readme.md"})

        assert response.status_code == 200
        data = response.json()
        assert data["processed"] == 1
        assert data["total_chunks"] == 5
        mock_ingest.ingest_file.assert_called_once()

    def test_ingest_file_not_found(self, client_with_pipeline) -> None:
        """Ingesting a non-existent file returns 404."""
        client, _, _, _ = client_with_pipeline

        response = client.post("/ingest", json={"file_path": "missing.md"})

        assert response.status_code == 404

    def test_ingest_pipeline_not_initialized(self, client_no_pipeline) -> None:
        """Ingestion returns 503 when pipeline is not initialized."""
        response = client_no_pipeline.post("/ingest", json={})

        assert response.status_code == 503


# ==========================================
# TestIngestStatusEndpoint
# ==========================================


class TestIngestStatusEndpoint:
    """Tests for GET /ingest/status."""

    def test_status_no_prior_ingestion(self, client_with_pipeline) -> None:
        """Status with no prior ingestion returns info message."""
        client, _, _, _ = client_with_pipeline
        response = client.get("/ingest/status")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_status_after_ingestion(self, client_with_pipeline) -> None:
        """Status after ingestion returns last summary."""
        client, _, mock_ingest, docs_dir = client_with_pipeline

        mock_ingest.ingest_directory.return_value = IngestionSummary(
            processed=3, skipped=0, failed=0, total_chunks=20, results=[]
        )

        # Trigger an ingestion first
        client.post("/ingest", json={})

        # Now check status
        response = client.get("/ingest/status")

        assert response.status_code == 200
        data = response.json()
        assert data["processed"] == 3
        assert data["total_chunks"] == 20


# ==========================================
# TestConfigEndpoint
# ==========================================


class TestConfigEndpoint:
    """Tests for GET /config."""

    def test_config_returns_public_settings(self, client_with_pipeline) -> None:
        """Config endpoint returns public settings without API keys."""
        client, _, _, _ = client_with_pipeline
        response = client.get("/config")

        assert response.status_code == 200
        data = response.json()
        assert data["llm_provider"] == "ollama_local"
        assert data["llm_model"] == "llama3.2:3b"
        assert data["rag_top_k"] == 5
        assert data["rag_min_similarity"] == 0.3
        assert data["api_port"] == 8000

        # Ensure no API keys are exposed
        assert "openai_api_key" not in data
        assert "anthropic_api_key" not in data
