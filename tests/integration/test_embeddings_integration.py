"""
Integration tests for the embedding service (M2).

These tests use the real sentence-transformers model to verify
end-to-end functionality. They are slower than unit tests because
they load the ML model and generate real embeddings.

Run with: pytest tests/integration/ -v
"""

from __future__ import annotations

import math

import pytest

from src.embeddings import EmbeddingService


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


class TestEmbeddingsIntegration:
    """Integration tests for the embedding service with real model."""

    @pytest.fixture(scope="class")
    def service(self) -> EmbeddingService:
        """Create a real EmbeddingService instance (loads the ML model)."""
        return EmbeddingService()

    def test_single_embedding_generation(self, service: EmbeddingService) -> None:
        """Test generating a single embedding with real model."""
        embedding = service.generate_embedding("Hello world")

        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_semantic_similarity_similar_texts(
        self, service: EmbeddingService
    ) -> None:
        """Test that similar texts have high cosine similarity."""
        emb1 = service.generate_embedding("the cat sits on the mat")
        emb2 = service.generate_embedding("a cat is sitting on a rug")

        similarity = cosine_similarity(emb1, emb2)

        assert similarity > 0.7, f"Expected similarity > 0.7, got {similarity:.4f}"

    def test_semantic_similarity_different_texts(
        self, service: EmbeddingService
    ) -> None:
        """Test that unrelated texts have low cosine similarity."""
        emb1 = service.generate_embedding("the cat sits on the mat")
        emb2 = service.generate_embedding("quantum physics equations")

        similarity = cosine_similarity(emb1, emb2)

        assert similarity < 0.3, f"Expected similarity < 0.3, got {similarity:.4f}"

    def test_batch_embedding_generation(self, service: EmbeddingService) -> None:
        """Test batch embedding generation with real model."""
        texts = ["machine learning", "artificial intelligence", "data science"]
        embeddings = service.generate_batch_embeddings(texts)

        assert len(embeddings) == len(texts)
        assert all(len(emb) == 384 for emb in embeddings)

    def test_multilingual_english_spanish(self, service: EmbeddingService) -> None:
        """Test cross-language similarity between English and Spanish."""
        emb_en = service.generate_embedding("Hello, how are you?")
        emb_es = service.generate_embedding("Hola, ¿cómo estás?")

        assert len(emb_en) == 384
        assert len(emb_es) == 384

        similarity = cosine_similarity(emb_en, emb_es)
        assert similarity > 0.5, f"Cross-language similarity too low: {similarity:.4f}"

    def test_model_info(self, service: EmbeddingService) -> None:
        """Test model information retrieval."""
        info = service.get_model_info()

        assert info["embedding_dimension"] == 384
        assert isinstance(info["max_seq_length"], int)
        assert info["max_seq_length"] > 0
