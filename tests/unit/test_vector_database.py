"""
Unit tests for vector database module.

Tests the QdrantDatabase class functionality including:
- Collection initialization
- Vector insertion (single and batch)
- Similarity search
- Metadata handling
- Deletion
- Error handling
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.database import QdrantDatabase


def _uuid() -> str:
    """Generate a random UUID string (Qdrant requires valid UUIDs for string IDs)."""
    return str(uuid4())


class TestQdrantDatabase:
    """Test suite for QdrantDatabase class."""

    @pytest.fixture(scope="class")
    def db(self) -> QdrantDatabase:
        """
        Fixture to create an in-memory QdrantDatabase instance.

        Scope is 'class' to reuse across tests.
        In-memory mode avoids disk I/O and cleanup.
        """
        return QdrantDatabase(
            collection_name="test_collection",
            in_memory=True,
        )

    @pytest.fixture()
    def fresh_db(self) -> QdrantDatabase:
        """
        Fixture to create a fresh in-memory database for each test.

        Use this when tests need an empty database.
        """
        return QdrantDatabase(
            collection_name="fresh_test_collection",
            in_memory=True,
        )

    # -- Initialization tests --

    def test_initialization_creates_collection(self, db: QdrantDatabase) -> None:
        """Test that initialization creates the collection."""
        assert db.client is not None
        info = db.get_collection_info()
        assert info["collection_name"] == "test_collection"

    def test_initialization_with_correct_vector_size(self, db: QdrantDatabase) -> None:
        """Test that collection is created with correct vector dimensions."""
        assert db.vector_size == 384
        info = db.get_collection_info()
        assert info["vector_size"] == 384

    def test_in_memory_mode(self, db: QdrantDatabase) -> None:
        """Test that in-memory mode is set correctly."""
        assert db.in_memory is True

    def test_initialize_existing_collection_is_idempotent(
        self, db: QdrantDatabase
    ) -> None:
        """Test that calling initialize on existing collection doesn't fail."""
        db.initialize_collection()  # Should not raise
        info = db.get_collection_info()
        assert info["collection_name"] == "test_collection"

    # -- Insert tests --

    def test_insert_single_vector(self, fresh_db: QdrantDatabase) -> None:
        """Test inserting a single vector with metadata."""
        vector = [0.1] * 384
        fresh_db.insert_vectors(
            ids=[_uuid()],
            vectors=[vector],
            metadata=[{"text": "hello world", "source": "test.pdf"}],
        )

        info = fresh_db.get_collection_info()
        assert info["vectors_count"] == 1

    def test_insert_multiple_vectors(self, fresh_db: QdrantDatabase) -> None:
        """Test inserting multiple vectors at once."""
        vectors = [[float(i) * 0.01] * 384 for i in range(5)]
        ids = [_uuid() for _ in range(5)]
        metadata = [{"index": i, "text": f"document {i}"} for i in range(5)]

        fresh_db.insert_vectors(ids=ids, vectors=vectors, metadata=metadata)

        info = fresh_db.get_collection_info()
        assert info["vectors_count"] == 5

    def test_insert_validates_matching_lengths(
        self, fresh_db: QdrantDatabase
    ) -> None:
        """Test that mismatched input lengths raise ValueError."""
        with pytest.raises(ValueError, match="Input lengths must match"):
            fresh_db.insert_vectors(
                ids=[_uuid(), _uuid()],
                vectors=[[0.1] * 384],  # Only 1 vector, but 2 ids
                metadata=[{"text": "hello"}],
            )

    def test_insert_validates_vector_dimensions(
        self, fresh_db: QdrantDatabase
    ) -> None:
        """Test that wrong vector dimensions raise ValueError."""
        with pytest.raises(ValueError, match="dimensions"):
            fresh_db.insert_vectors(
                ids=[_uuid()],
                vectors=[[0.1] * 100],  # Wrong dimension (100 instead of 384)
                metadata=[{"text": "hello"}],
            )

    def test_insert_empty_list_raises_error(
        self, fresh_db: QdrantDatabase
    ) -> None:
        """Test that inserting empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot insert empty list"):
            fresh_db.insert_vectors(ids=[], vectors=[], metadata=[])

    # -- Search tests --

    def test_search_returns_results(self, fresh_db: QdrantDatabase) -> None:
        """Test that search returns inserted vectors."""
        vector = [0.5] * 384
        doc_id = _uuid()

        fresh_db.insert_vectors(
            ids=[doc_id],
            vectors=[vector],
            metadata=[{"text": "machine learning"}],
        )

        results = fresh_db.search_similar(query_vector=vector, limit=1)

        assert len(results) == 1
        assert results[0]["id"] == doc_id
        assert results[0]["score"] > 0.99  # Same vector = very high similarity
        assert results[0]["metadata"]["text"] == "machine learning"

    def test_search_returns_most_similar_first(
        self, fresh_db: QdrantDatabase
    ) -> None:
        """Test that search results are ordered by similarity (descending)."""
        # Vectors must point in DIFFERENT directions (not just different magnitudes)
        # because cosine similarity measures angle, not magnitude.
        similar_vector = [0.5] * 384
        different_vector = [-0.5] * 384
        medium_vector = [0.5] * 192 + [-0.5] * 192  # Mixed direction

        fresh_db.insert_vectors(
            ids=[_uuid(), _uuid(), _uuid()],
            vectors=[similar_vector, different_vector, medium_vector],
            metadata=[
                {"label": "similar"},
                {"label": "different"},
                {"label": "medium"},
            ],
        )

        query = [0.5] * 384
        results = fresh_db.search_similar(query_vector=query, limit=3)

        assert len(results) == 3
        # Most similar should be first
        assert results[0]["metadata"]["label"] == "similar"
        # Scores should be descending
        assert results[0]["score"] >= results[1]["score"]
        assert results[1]["score"] >= results[2]["score"]

    def test_search_respects_limit(self, fresh_db: QdrantDatabase) -> None:
        """Test that search respects the limit parameter."""
        vectors = [[float(i) * 0.1] * 384 for i in range(10)]
        ids = [_uuid() for _ in range(10)]
        metadata = [{"index": i} for i in range(10)]

        fresh_db.insert_vectors(ids=ids, vectors=vectors, metadata=metadata)

        results = fresh_db.search_similar(
            query_vector=[0.5] * 384, limit=3
        )
        assert len(results) <= 3

    def test_search_with_score_threshold(
        self, fresh_db: QdrantDatabase
    ) -> None:
        """Test that score_threshold filters low-similarity results."""
        similar = [0.5] * 384
        different = [-0.5] * 384

        fresh_db.insert_vectors(
            ids=[_uuid(), _uuid()],
            vectors=[similar, different],
            metadata=[{"label": "high"}, {"label": "low"}],
        )

        # Search with high threshold - should only return the similar one
        results = fresh_db.search_similar(
            query_vector=[0.5] * 384,
            limit=10,
            score_threshold=0.9,
        )

        # All returned results should have score >= threshold
        assert all(r["score"] >= 0.9 for r in results)

    def test_search_validates_query_dimensions(
        self, fresh_db: QdrantDatabase
    ) -> None:
        """Test that wrong query dimensions raise ValueError."""
        with pytest.raises(ValueError, match="dimensions"):
            fresh_db.search_similar(query_vector=[0.1] * 100)

    def test_search_empty_collection_returns_empty(
        self, fresh_db: QdrantDatabase
    ) -> None:
        """Test that searching an empty collection returns empty list."""
        results = fresh_db.search_similar(
            query_vector=[0.5] * 384, limit=5
        )
        assert results == []

    # -- Metadata tests --

    def test_metadata_is_preserved(self, fresh_db: QdrantDatabase) -> None:
        """Test that all metadata fields survive insert and search."""
        metadata = {
            "text": "Machine learning is great",
            "source_file": "docs/ml.pdf",
            "page": 42,
            "language": "en",
            "tags": ["ml", "ai"],
        }

        fresh_db.insert_vectors(
            ids=[_uuid()],
            vectors=[[0.5] * 384],
            metadata=[metadata],
        )

        results = fresh_db.search_similar(
            query_vector=[0.5] * 384, limit=1
        )

        assert results[0]["metadata"]["text"] == "Machine learning is great"
        assert results[0]["metadata"]["source_file"] == "docs/ml.pdf"
        assert results[0]["metadata"]["page"] == 42
        assert results[0]["metadata"]["language"] == "en"
        assert results[0]["metadata"]["tags"] == ["ml", "ai"]

    # -- Delete tests --

    def test_delete_removes_vector(self, fresh_db: QdrantDatabase) -> None:
        """Test that delete removes vectors from collection."""
        id_1 = _uuid()
        id_2 = _uuid()

        fresh_db.insert_vectors(
            ids=[id_1, id_2],
            vectors=[[0.1] * 384, [0.2] * 384],
            metadata=[{"text": "first"}, {"text": "second"}],
        )

        info_before = fresh_db.get_collection_info()
        assert info_before["vectors_count"] == 2

        fresh_db.delete_by_id([id_1])

        info_after = fresh_db.get_collection_info()
        assert info_after["vectors_count"] == 1

    def test_delete_empty_list_raises_error(
        self, fresh_db: QdrantDatabase
    ) -> None:
        """Test that deleting empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot delete empty list"):
            fresh_db.delete_by_id([])

    # -- Collection info tests --

    def test_get_collection_info_returns_expected_fields(
        self, db: QdrantDatabase
    ) -> None:
        """Test that collection info contains expected fields."""
        info = db.get_collection_info()

        assert "collection_name" in info
        assert "vectors_count" in info
        assert "vector_size" in info
        assert "distance_metric" in info
        assert "status" in info

        assert info["distance_metric"] == "Cosine"
        assert info["vector_size"] == 384
