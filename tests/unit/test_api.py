"""
Unit tests for the FastAPI REST API.

Tests all endpoints using FastAPI TestClient with mocked RAG pipeline.
The lifespan creates the pipeline, so we patch the RAGPipeline constructor.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
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
def client_with_pipeline(mock_pipeline):
    """TestClient with an initialized mock pipeline via patched constructor."""
    with patch("src.api.server.RAGPipeline", return_value=mock_pipeline):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_pipeline


@pytest.fixture
def client_no_pipeline():
    """TestClient with no pipeline initialized (constructor raises)."""
    with patch(
        "src.api.server.RAGPipeline",
        side_effect=RuntimeError("No pipeline for testing"),
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
        client, _ = client_with_pipeline
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
        client, mock_pipe = client_with_pipeline
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
        client, mock_pipe = client_with_pipeline
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
        client, mock_pipe = client_with_pipeline
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
        client, _ = client_with_pipeline
        response = client.get("/sources")

        assert response.status_code == 200
        data = response.json()
        assert data["sources"]["collection_name"] == "test_collection"
        assert data["sources"]["vectors_count"] == 100

    def test_sources_pipeline_not_initialized(self, client_no_pipeline) -> None:
        """Sources returns 503 when pipeline is not initialized."""
        response = client_no_pipeline.get("/sources")

        assert response.status_code == 503
