"""
Embedding service for DocVault using sentence-transformers.

This service provides local embedding generation without external API dependencies.
"""

from __future__ import annotations

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from .config import DEFAULT_MODEL, MODEL_DIMENSIONS


logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using sentence-transformers.

    This service loads a pre-trained multilingual model and provides methods
    to convert text into vector representations (embeddings) that can be used
    for semantic similarity search.

    Attributes:
        model_name: Name of the sentence-transformer model to use
        model: Loaded SentenceTransformer model instance
        embedding_dimension: Dimension of the generated embeddings

    Example:
        >>> service = EmbeddingService()
        >>> embedding = service.generate_embedding("Hello world")
        >>> print(len(embedding))  # Should print 384
        384
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initialize the embedding service with a sentence-transformer model.

        Args:
            model_name: Name of the sentence-transformer model to use.
                       Defaults to the configured DEFAULT_MODEL if not provided.

        Raises:
            RuntimeError: If the model fails to load
        """
        self.model_name = model_name or DEFAULT_MODEL

        logger.info(f"Loading embedding model: {self.model_name}")

        try:
            # Load the model (will download on first use, then cache locally)
            self.model = SentenceTransformer(self.model_name)

            # Get embedding dimension from config
            self.embedding_dimension = MODEL_DIMENSIONS.get(
                self.model_name,
                384  # Default fallback
            )

            logger.info(
                f"Embedding model loaded successfully. "
                f"Dimension: {self.embedding_dimension}"
            )

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(
                f"Could not initialize embedding model '{self.model_name}': {e}"
            ) from e

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding vector for a single text.

        Args:
            text: The input text to generate an embedding for

        Returns:
            A list of floats representing the embedding vector

        Raises:
            ValueError: If the input text is empty
            RuntimeError: If embedding generation fails

        Example:
            >>> service = EmbeddingService()
            >>> embedding = service.generate_embedding("machine learning")
            >>> isinstance(embedding, list)
            True
            >>> len(embedding)
            384
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        try:
            # Generate embedding using the model
            # convert_to_numpy=True returns a numpy array, then we convert to list
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization for cosine similarity
            )

            # Convert numpy array to Python list
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def generate_batch_embeddings(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        This method is more efficient than calling generate_embedding() multiple
        times because it processes texts in batches using the GPU (if available).

        Args:
            texts: List of input texts to generate embeddings for
            batch_size: Number of texts to process at once. Larger batches are
                       faster but use more memory. Default is 32.
            show_progress: Whether to show a progress bar during processing

        Returns:
            A list of embedding vectors, one for each input text

        Raises:
            ValueError: If the input list is empty or contains empty strings
            RuntimeError: If embedding generation fails

        Example:
            >>> service = EmbeddingService()
            >>> texts = ["hello", "world", "machine learning"]
            >>> embeddings = service.generate_batch_embeddings(texts)
            >>> len(embeddings)
            3
            >>> len(embeddings[0])
            384
        """
        if not texts:
            raise ValueError("Input text list cannot be empty")

        # Check for empty strings
        if any(not text or not text.strip() for text in texts):
            raise ValueError("Input texts cannot contain empty strings")

        try:
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization
            )

            # Convert numpy array to list of lists
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}") from e

    def get_model_info(self) -> dict[str, str | int]:
        """
        Get information about the loaded embedding model.

        Returns:
            Dictionary containing model information:
                - model_name: Name of the loaded model
                - embedding_dimension: Dimension of the embeddings
                - max_seq_length: Maximum sequence length the model can handle

        Example:
            >>> service = EmbeddingService()
            >>> info = service.get_model_info()
            >>> info['embedding_dimension']
            384
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_seq_length": self.model.max_seq_length,
        }
