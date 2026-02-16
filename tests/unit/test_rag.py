"""
Unit tests for the RAG pipeline module.

Tests configuration, data models, and the complete RAG pipeline
using mocked services (embedding, vector DB, LLM).
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import pytest

from src.rag import RAGConfig, RAGPipeline, RAGResponse, Source
from src.rag.config import DEFAULT_MIN_SIMILARITY, DEFAULT_TOP_K, NO_RESULTS_MESSAGE


# ==========================================
# Fixtures
# ==========================================


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service that returns a fixed vector."""
    service = MagicMock()
    service.generate_embedding.return_value = [0.1] * 384
    return service


@pytest.fixture
def mock_search_results():
    """Standard search results returned by the mock vector DB."""
    return [
        {
            "id": "chunk_001",
            "score": 0.92,
            "metadata": {
                "chunk_text": "Docker is a containerization platform.",
                "source_file": "docker_guide.md",
                "chunk_index": 0,
                "document_title": "Docker Guide",
                "document_format": "markdown",
            },
        },
        {
            "id": "chunk_002",
            "score": 0.85,
            "metadata": {
                "chunk_text": "To install Docker, download from docker.com.",
                "source_file": "docker_guide.md",
                "chunk_index": 1,
                "document_title": "Docker Guide",
                "document_format": "markdown",
            },
        },
    ]


@pytest.fixture
def mock_vector_db(mock_search_results):
    """Mock vector database that returns predefined search results."""
    db = MagicMock()
    db.search_similar.return_value = mock_search_results
    db.get_collection_info.return_value = {
        "collection_name": "test_collection",
        "vectors_count": 100,
        "vector_size": 384,
        "distance_metric": "Cosine",
        "status": "green",
    }
    return db


@pytest.fixture
def mock_llm():
    """Mock LLM provider that returns a fixed answer."""
    llm = MagicMock()
    llm.generate.return_value = "Docker is a platform for containerization."
    llm.generate_stream.return_value = iter(["Docker ", "is ", "great."])
    llm.get_model_info.return_value = {"provider": "mock", "model": "test-model"}
    return llm


@pytest.fixture
def pipeline(mock_embedding_service, mock_vector_db, mock_llm):
    """RAGPipeline with all mocked dependencies."""
    return RAGPipeline(
        embedding_service=mock_embedding_service,
        vector_db=mock_vector_db,
        llm_provider=mock_llm,
    )


# ==========================================
# TestRAGConfig
# ==========================================


class TestRAGConfig:
    """Tests for RAG configuration defaults."""

    def test_defaults(self) -> None:
        """RAGConfig has sensible defaults."""
        config = RAGConfig()
        assert config.top_k == DEFAULT_TOP_K
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.min_similarity == DEFAULT_MIN_SIMILARITY

    def test_custom_values(self) -> None:
        """RAGConfig accepts custom values."""
        config = RAGConfig(top_k=10, temperature=0.5, max_tokens=2048, min_similarity=0.5)
        assert config.top_k == 10
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.min_similarity == 0.5


# ==========================================
# TestRAGModels
# ==========================================


class TestRAGModels:
    """Tests for RAG data models."""

    def test_source_creation(self) -> None:
        """Source can be created with required fields."""
        source = Source(
            chunk_text="Hello world",
            source_file="test.md",
            chunk_index=0,
            similarity_score=0.95,
        )
        assert source.chunk_text == "Hello world"
        assert source.source_file == "test.md"
        assert source.chunk_index == 0
        assert source.similarity_score == 0.95
        assert source.document_title is None
        assert source.document_format is None

    def test_rag_response_with_sources(self) -> None:
        """RAGResponse correctly stores answer and sources."""
        sources = [
            Source(
                chunk_text="text",
                source_file="file.md",
                chunk_index=0,
                similarity_score=0.9,
            )
        ]
        response = RAGResponse(
            answer="The answer",
            sources=sources,
            query="test query",
            model_used="test-model",
            retrieval_count=1,
            retrieval_time_ms=50.0,
            generation_time_ms=200.0,
        )
        assert response.answer == "The answer"
        assert len(response.sources) == 1
        assert response.query == "test query"
        assert response.retrieval_time_ms == 50.0
        assert response.generation_time_ms == 200.0

    def test_rag_response_empty_sources(self) -> None:
        """RAGResponse works with no sources (no results case)."""
        response = RAGResponse(
            answer=NO_RESULTS_MESSAGE,
            sources=[],
            query="unknown topic",
            model_used="test-model",
            retrieval_count=0,
        )
        assert response.sources == []
        assert response.retrieval_count == 0
        assert response.retrieval_time_ms is None
        assert response.generation_time_ms is None


# ==========================================
# TestRAGPipeline
# ==========================================


class TestRAGPipeline:
    """Tests for the RAG pipeline."""

    def test_query_returns_rag_response(self, pipeline) -> None:
        """query() returns a RAGResponse with correct fields."""
        response = pipeline.query("How do I install Docker?")

        assert isinstance(response, RAGResponse)
        assert response.answer == "Docker is a platform for containerization."
        assert response.query == "How do I install Docker?"
        assert response.model_used == "test-model"
        assert response.retrieval_count == 2
        assert response.retrieval_time_ms is not None
        assert response.generation_time_ms is not None

    def test_query_calls_embedding_service(
        self, pipeline, mock_embedding_service
    ) -> None:
        """query() generates an embedding for the query text."""
        pipeline.query("test question")

        mock_embedding_service.generate_embedding.assert_called_once_with(
            "test question"
        )

    def test_query_calls_vector_db(self, pipeline, mock_vector_db) -> None:
        """query() searches the vector database with the embedding."""
        pipeline.query("test question")

        mock_vector_db.search_similar.assert_called_once()
        call_kwargs = mock_vector_db.search_similar.call_args
        assert call_kwargs.kwargs["query_vector"] == [0.1] * 384
        assert call_kwargs.kwargs["limit"] == 5

    def test_query_calls_llm_with_context(self, pipeline, mock_llm) -> None:
        """query() passes the built context to the LLM."""
        pipeline.query("test question")

        mock_llm.generate.assert_called_once()
        call_kwargs = mock_llm.generate.call_args
        assert call_kwargs.kwargs["prompt"] == "test question"
        assert "docker_guide.md" in call_kwargs.kwargs["context"]
        assert "Docker is a containerization platform." in call_kwargs.kwargs["context"]

    def test_query_empty_text_raises(self, pipeline) -> None:
        """query() raises ValueError for empty input."""
        with pytest.raises(ValueError, match="cannot be empty"):
            pipeline.query("")

        with pytest.raises(ValueError, match="cannot be empty"):
            pipeline.query("   ")

    def test_query_no_results(
        self, mock_embedding_service, mock_llm
    ) -> None:
        """query() returns NO_RESULTS_MESSAGE when no documents match."""
        empty_db = MagicMock()
        empty_db.search_similar.return_value = []

        pipe = RAGPipeline(
            embedding_service=mock_embedding_service,
            vector_db=empty_db,
            llm_provider=mock_llm,
        )

        response = pipe.query("obscure topic")

        assert response.answer == NO_RESULTS_MESSAGE
        assert response.sources == []
        assert response.retrieval_count == 0
        mock_llm.generate.assert_not_called()

    def test_query_custom_parameters(self, pipeline, mock_vector_db, mock_llm) -> None:
        """query() passes custom top_k, temperature, max_tokens."""
        pipeline.query("test", top_k=3, temperature=0.2, max_tokens=512)

        call_kwargs = mock_vector_db.search_similar.call_args
        assert call_kwargs.kwargs["limit"] == 3

        call_kwargs = mock_llm.generate.call_args
        assert call_kwargs.kwargs["temperature"] == 0.2
        assert call_kwargs.kwargs["max_tokens"] == 512

    def test_query_streaming(self, pipeline) -> None:
        """query(streaming=True) returns an iterator of chunks."""
        result = pipeline.query("test", streaming=True)

        chunks = list(result)
        assert chunks == ["Docker ", "is ", "great."]

    def test_build_context_format(self, pipeline) -> None:
        """_build_context() formats sources with headers and scores."""
        sources = [
            Source(
                chunk_text="First chunk.",
                source_file="file1.md",
                chunk_index=0,
                similarity_score=0.95,
            ),
            Source(
                chunk_text="Second chunk.",
                source_file="file2.txt",
                chunk_index=1,
                similarity_score=0.80,
            ),
        ]

        context = pipeline._build_context(sources)

        assert "[Source 1] file1.md (similarity: 0.95)" in context
        assert "First chunk." in context
        assert "[Source 2] file2.txt (similarity: 0.80)" in context
        assert "Second chunk." in context

    def test_get_indexed_sources(self, pipeline, mock_vector_db) -> None:
        """get_indexed_sources() delegates to vector database."""
        info = pipeline.get_indexed_sources()

        mock_vector_db.get_collection_info.assert_called_once()
        assert info["collection_name"] == "test_collection"
        assert info["vectors_count"] == 100
