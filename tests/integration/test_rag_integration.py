"""
Integration tests for the RAG pipeline.

Uses real EmbeddingService and in-memory QdrantDatabase with
a mocked LLM provider to test the full retrieval flow.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from src.database import QdrantDatabase
from src.embeddings import EmbeddingService
from src.rag import RAGConfig, RAGPipeline, RAGResponse
from src.rag.config import NO_RESULTS_MESSAGE


# ==========================================
# Fixtures
# ==========================================


@pytest.fixture(scope="class")
def embedding_service():
    """Real embedding service (loaded once per test class)."""
    return EmbeddingService()


@pytest.fixture
def vector_db():
    """In-memory Qdrant database for each test."""
    return QdrantDatabase(in_memory=True)


@pytest.fixture
def mock_llm():
    """Mock LLM that echoes context info."""
    llm = MagicMock()
    llm.generate.side_effect = lambda prompt, context=None, **kwargs: (
        f"Based on the context, here is the answer about: {prompt}"
    )
    llm.generate_stream.side_effect = lambda prompt, context=None, **kwargs: iter(
        ["Streaming ", "answer ", f"about {prompt}"]
    )
    llm.get_model_info.return_value = {"provider": "mock", "model": "integration-test"}
    return llm


@pytest.fixture
def populated_db(embedding_service, vector_db):
    """Vector DB with pre-ingested test documents."""
    texts = [
        "Python is a high-level programming language known for its readability.",
        "Docker containers package applications with their dependencies.",
        "Machine learning uses algorithms to learn patterns from data.",
        "FastAPI is a modern web framework for building APIs with Python.",
    ]

    files = ["python_guide.md", "docker_guide.md", "ml_intro.md", "fastapi_docs.md"]

    embeddings = embedding_service.generate_batch_embeddings(texts)

    ids = [str(uuid.uuid4()) for _ in range(len(texts))]
    metadata = [
        {
            "chunk_text": text,
            "source_file": file,
            "chunk_index": 0,
            "document_title": file.replace("_", " ").replace(".md", "").title(),
            "document_format": "markdown",
        }
        for text, file in zip(texts, files)
    ]

    vector_db.insert_vectors(ids=ids, vectors=embeddings, metadata=metadata)

    return vector_db


# ==========================================
# Integration Tests
# ==========================================


class TestRAGIntegration:
    """Integration tests with real embeddings and vector search."""

    def test_query_returns_relevant_sources(
        self, embedding_service, populated_db, mock_llm
    ) -> None:
        """Query retrieves semantically relevant documents."""
        pipeline = RAGPipeline(
            embedding_service=embedding_service,
            vector_db=populated_db,
            llm_provider=mock_llm,
        )

        response = pipeline.query("How do I use Docker containers?")

        assert isinstance(response, RAGResponse)
        assert response.retrieval_count > 0
        source_files = [s.source_file for s in response.sources]
        assert "docker_guide.md" in source_files

    def test_query_empty_database(
        self, embedding_service, vector_db, mock_llm
    ) -> None:
        """Query on empty database returns no-results message."""
        pipeline = RAGPipeline(
            embedding_service=embedding_service,
            vector_db=vector_db,
            llm_provider=mock_llm,
        )

        response = pipeline.query("What is quantum computing?")

        assert response.answer == NO_RESULTS_MESSAGE
        assert response.sources == []
        assert response.retrieval_count == 0

    def test_context_contains_chunk_text(
        self, embedding_service, populated_db, mock_llm
    ) -> None:
        """The context passed to the LLM contains the retrieved chunk text."""
        pipeline = RAGPipeline(
            embedding_service=embedding_service,
            vector_db=populated_db,
            llm_provider=mock_llm,
        )

        pipeline.query("Tell me about Python programming")

        mock_llm.generate.assert_called_once()
        context = mock_llm.generate.call_args.kwargs["context"]
        assert "Python" in context
        assert "[Source" in context

    def test_streaming_with_real_retrieval(
        self, embedding_service, populated_db, mock_llm
    ) -> None:
        """Streaming query works with real retrieval + mock LLM."""
        pipeline = RAGPipeline(
            embedding_service=embedding_service,
            vector_db=populated_db,
            llm_provider=mock_llm,
        )

        result = pipeline.query("What is FastAPI?", streaming=True)
        chunks = list(result)

        assert len(chunks) > 0
        assert "Streaming " in chunks
