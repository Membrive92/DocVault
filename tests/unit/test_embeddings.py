"""
Unit tests for embeddings module.

Tests the EmbeddingService class functionality including:
- Embedding generation
- Batch processing
- Cosine similarity
- Error handling
"""

from __future__ import annotations

import math

import pytest

from src.embeddings import EmbeddingService


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity value between -1 and 1
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


class TestEmbeddingService:
    """Test suite for EmbeddingService class."""

    @pytest.fixture(scope="class")
    def service(self) -> EmbeddingService:
        """
        Fixture to create an EmbeddingService instance.

        Scope is 'class' to avoid reloading the model for every test.
        """
        return EmbeddingService()

    def test_service_initialization(self, service: EmbeddingService) -> None:
        """Test that the service initializes correctly."""
        assert service is not None
        assert service.model is not None
        assert service.embedding_dimension == 384
        assert "multilingual" in service.model_name.lower()

    def test_generate_embedding_returns_correct_dimensions(
        self,
        service: EmbeddingService
    ) -> None:
        """Test that embeddings have the correct dimension."""
        embedding = service.generate_embedding("hello world")

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_generate_embedding_with_different_texts(
        self,
        service: EmbeddingService
    ) -> None:
        """Test that different texts produce different embeddings."""
        emb1 = service.generate_embedding("machine learning")
        emb2 = service.generate_embedding("artificial intelligence")
        emb3 = service.generate_embedding("cooking recipes")

        # Similar topics should be more similar than unrelated topics
        sim_ml_ai = cosine_similarity(emb1, emb2)
        sim_ml_cooking = cosine_similarity(emb1, emb3)

        assert sim_ml_ai > sim_ml_cooking
        assert sim_ml_ai > 0.5  # Should have some similarity
        assert sim_ml_cooking < 0.7  # Should be less similar

    def test_similar_texts_have_high_cosine_similarity(
        self,
        service: EmbeddingService
    ) -> None:
        """Test that semantically similar texts have high cosine similarity."""
        emb1 = service.generate_embedding("the cat sits on the mat")
        emb2 = service.generate_embedding("a cat is sitting on a rug")

        similarity = cosine_similarity(emb1, emb2)

        # Similar sentences should have high cosine similarity
        assert similarity > 0.7, f"Expected similarity > 0.7, got {similarity}"

    def test_different_texts_have_low_cosine_similarity(
        self,
        service: EmbeddingService
    ) -> None:
        """Test that semantically different texts have low cosine similarity."""
        emb1 = service.generate_embedding("the cat sits on the mat")
        emb2 = service.generate_embedding("quantum physics equations")

        similarity = cosine_similarity(emb1, emb2)

        # Unrelated sentences should have low cosine similarity
        assert similarity < 0.3, f"Expected similarity < 0.3, got {similarity}"

    def test_multilingual_support(self, service: EmbeddingService) -> None:
        """Test that the model handles both English and Spanish."""
        # English text
        emb_en = service.generate_embedding("Hello, how are you?")

        # Spanish text (same meaning)
        emb_es = service.generate_embedding("Hola, ¿cómo estás?")

        # Both should generate valid embeddings
        assert len(emb_en) == 384
        assert len(emb_es) == 384

        # Similar meaning across languages should have some similarity
        similarity = cosine_similarity(emb_en, emb_es)
        assert similarity > 0.5, f"Cross-language similarity too low: {similarity}"

    def test_generate_batch_embeddings_returns_correct_shape(
        self,
        service: EmbeddingService
    ) -> None:
        """Test that batch embeddings return correct number and dimensions."""
        texts = ["hello", "world", "machine learning"]
        embeddings = service.generate_batch_embeddings(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
        assert all(
            all(isinstance(x, float) for x in emb)
            for emb in embeddings
        )

    def test_batch_embeddings_match_individual_embeddings(
        self,
        service: EmbeddingService
    ) -> None:
        """Test that batch processing produces same results as individual."""
        text = "test sentence for embedding"

        # Generate individual embedding
        individual_emb = service.generate_embedding(text)

        # Generate batch embedding with single text
        batch_emb = service.generate_batch_embeddings([text])[0]

        # They should be very similar (may have minor floating point differences)
        similarity = cosine_similarity(individual_emb, batch_emb)
        assert similarity > 0.99, f"Batch and individual embeddings differ: {similarity}"

    def test_generate_embedding_raises_error_on_empty_string(
        self,
        service: EmbeddingService
    ) -> None:
        """Test that empty strings raise ValueError."""
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            service.generate_embedding("")

        with pytest.raises(ValueError, match="Input text cannot be empty"):
            service.generate_embedding("   ")  # Only whitespace

    def test_generate_batch_embeddings_raises_error_on_empty_list(
        self,
        service: EmbeddingService
    ) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="Input text list cannot be empty"):
            service.generate_batch_embeddings([])

    def test_generate_batch_embeddings_raises_error_on_empty_strings(
        self,
        service: EmbeddingService
    ) -> None:
        """Test that list with empty strings raises ValueError."""
        with pytest.raises(ValueError, match="cannot contain empty strings"):
            service.generate_batch_embeddings(["hello", "", "world"])

    def test_get_model_info_returns_correct_information(
        self,
        service: EmbeddingService
    ) -> None:
        """Test that model info contains expected fields."""
        info = service.get_model_info()

        assert "model_name" in info
        assert "embedding_dimension" in info
        assert "max_seq_length" in info

        assert info["embedding_dimension"] == 384
        assert isinstance(info["max_seq_length"], int)
        assert info["max_seq_length"] > 0

    def test_embeddings_are_normalized(self, service: EmbeddingService) -> None:
        """Test that embeddings are L2 normalized."""
        embedding = service.generate_embedding("test normalization")

        # Calculate L2 norm (magnitude)
        magnitude = math.sqrt(sum(x * x for x in embedding))

        # L2 normalized vectors should have magnitude ≈ 1.0
        assert abs(magnitude - 1.0) < 0.01, f"Embedding not normalized: {magnitude}"
