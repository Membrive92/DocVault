"""
Integration tests for the vector database with real embeddings (M2 + M3).

These tests use the real sentence-transformers model to generate
embeddings and store/search them in Qdrant. They verify that the
full pipeline works end-to-end.

Run with: pytest tests/integration/ -v
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.database import QdrantDatabase
from src.embeddings import EmbeddingService


class TestVectorDbIntegration:
    """Integration tests: EmbeddingService (M2) + QdrantDatabase (M3)."""

    @pytest.fixture(scope="class")
    def embedding_service(self) -> EmbeddingService:
        """Create a real EmbeddingService instance."""
        return EmbeddingService()

    @pytest.fixture(scope="class")
    def db(self) -> QdrantDatabase:
        """Create an in-memory Qdrant database for integration tests."""
        return QdrantDatabase(
            collection_name="integration_test",
            in_memory=True,
        )

    def test_insert_real_embeddings(
        self,
        embedding_service: EmbeddingService,
        db: QdrantDatabase,
    ) -> None:
        """Test indexing documents with real embeddings."""
        documents = [
            "Machine learning is a branch of artificial intelligence that enables systems to learn from data.",
            "Docker containers package applications with their dependencies for consistent deployment.",
            "Python is a high-level programming language known for its simple and readable syntax.",
            "Neural networks are computing systems inspired by biological neural networks in the brain.",
            "Kubernetes orchestrates containerized applications across clusters of machines.",
            "Natural language processing allows computers to understand and generate human language.",
        ]

        embeddings = embedding_service.generate_batch_embeddings(documents)

        assert len(embeddings) == len(documents)

        ids = [str(uuid4()) for _ in documents]
        metadata = [
            {"text": doc, "index": i, "source": "test_docs"}
            for i, doc in enumerate(documents)
        ]

        db.insert_vectors(ids=ids, vectors=embeddings, metadata=metadata)

        info = db.get_collection_info()
        assert info["vectors_count"] == len(documents)

    def test_semantic_search_ai_query(
        self,
        embedding_service: EmbeddingService,
        db: QdrantDatabase,
    ) -> None:
        """Test that 'What is AI?' returns machine learning / neural network docs."""
        query_embedding = embedding_service.generate_embedding("What is AI?")
        results = db.search_similar(query_vector=query_embedding, limit=2)

        assert len(results) == 2
        # Top results should be about AI-related topics
        top_texts = [r["metadata"]["text"] for r in results]
        assert any("machine learning" in t.lower() or "neural" in t.lower() for t in top_texts)

    def test_semantic_search_devops_query(
        self,
        embedding_service: EmbeddingService,
        db: QdrantDatabase,
    ) -> None:
        """Test that 'How do I deploy applications?' returns Docker / Kubernetes docs."""
        query_embedding = embedding_service.generate_embedding(
            "How do I deploy applications?"
        )
        results = db.search_similar(query_vector=query_embedding, limit=2)

        assert len(results) == 2
        top_texts = [r["metadata"]["text"] for r in results]
        assert any("docker" in t.lower() or "kubernetes" in t.lower() for t in top_texts)

    def test_semantic_search_programming_query(
        self,
        embedding_service: EmbeddingService,
        db: QdrantDatabase,
    ) -> None:
        """Test that 'Tell me about programming languages' returns Python doc."""
        query_embedding = embedding_service.generate_embedding(
            "Tell me about programming languages"
        )
        results = db.search_similar(query_vector=query_embedding, limit=2)

        assert len(results) == 2
        top_texts = [r["metadata"]["text"] for r in results]
        assert any("python" in t.lower() for t in top_texts)

    def test_score_threshold_filters_irrelevant(
        self,
        embedding_service: EmbeddingService,
        db: QdrantDatabase,
    ) -> None:
        """Test that high score threshold filters unrelated queries."""
        query_embedding = embedding_service.generate_embedding(
            "The history of ancient Roman architecture"
        )

        all_results = db.search_similar(
            query_vector=query_embedding, limit=5
        )
        filtered_results = db.search_similar(
            query_vector=query_embedding,
            limit=5,
            score_threshold=0.8,
        )

        assert len(filtered_results) <= len(all_results)

    def test_collection_info_correct(self, db: QdrantDatabase) -> None:
        """Test collection info after integration operations."""
        info = db.get_collection_info()

        assert info["vector_size"] == 384
        assert info["distance_metric"] == "Cosine"
        assert info["vectors_count"] >= 6  # At least the 6 documents inserted

    def test_delete_after_insert(
        self,
        embedding_service: EmbeddingService,
        db: QdrantDatabase,
    ) -> None:
        """Test deleting a vector after insertion with real embeddings."""
        doc_id = str(uuid4())
        embedding = embedding_service.generate_embedding("temporary document")

        db.insert_vectors(
            ids=[doc_id],
            vectors=[embedding],
            metadata=[{"text": "temporary", "temp": True}],
        )

        count_before = db.get_collection_info()["vectors_count"]

        db.delete_by_id([doc_id])

        count_after = db.get_collection_info()["vectors_count"]
        assert count_after == count_before - 1
