"""
Abstract interface for vector database operations.

This allows swapping Qdrant for alternative vector databases
without changing downstream code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class VectorDatabase(ABC):
    """
    Abstract base class for vector database implementations.

    Defines the contract that any vector database must implement
    to work with DocVault. Equivalent to a Java interface.

    Methods:
        initialize_collection: Create or connect to the vector collection
        insert_vectors: Store vectors with metadata
        search_similar: Find vectors closest to a query
        delete_by_id: Remove vectors by ID
        get_collection_info: Get collection metadata
    """

    @abstractmethod
    def initialize_collection(self) -> None:
        """Create or connect to the vector collection."""
        pass

    @abstractmethod
    def insert_vectors(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None:
        """
        Insert vectors with metadata into the collection.

        Args:
            ids: Unique identifiers for each vector
            vectors: List of embedding vectors (each 384 floats)
            metadata: List of metadata dicts (one per vector)

        Raises:
            ValueError: If input lengths don't match
            RuntimeError: If insertion fails
        """
        pass

    @abstractmethod
    def search_similar(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for vectors most similar to the query.

        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of dicts with keys: id, score, metadata

        Raises:
            RuntimeError: If search fails
        """
        pass

    @abstractmethod
    def delete_by_id(self, ids: list[str]) -> None:
        """
        Delete vectors by their IDs.

        Args:
            ids: List of vector IDs to delete

        Raises:
            RuntimeError: If deletion fails
        """
        pass

    @abstractmethod
    def get_collection_info(self) -> dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dict with collection metadata (name, vector count, etc.)
        """
        pass
