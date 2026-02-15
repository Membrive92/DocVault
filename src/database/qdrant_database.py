"""
Qdrant implementation of the vector database interface.

Provides vector storage and similarity search using Qdrant's
in-process client (no external server required).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    VectorParams,
)

from .config import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_STORAGE_PATH,
    DISTANCE_METRIC,
    HNSW_EF_CONSTRUCT,
    HNSW_M,
    VECTOR_SIZE,
)
from .vector_database import VectorDatabase


logger = logging.getLogger(__name__)


class QdrantDatabase(VectorDatabase):
    """
    Qdrant implementation of vector database.

    Supports both in-memory mode (for testing) and persistent
    local storage (for production). No external Qdrant server needed.

    Attributes:
        collection_name: Name of the vector collection
        vector_size: Dimension of stored vectors (384)
        in_memory: Whether to use in-memory storage
        client: Qdrant client instance

    Example:
        >>> db = QdrantDatabase(in_memory=True)
        >>> db.insert_vectors(
        ...     ids=["doc1"],
        ...     vectors=[[0.1, 0.2, ...]],  # 384 floats
        ...     metadata=[{"text": "hello"}]
        ... )
        >>> results = db.search_similar([0.1, 0.2, ...], limit=5)
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = VECTOR_SIZE,
        in_memory: bool = False,
        storage_path: str = DEFAULT_STORAGE_PATH,
    ) -> None:
        """
        Initialize Qdrant database connection.

        Args:
            collection_name: Name of the vector collection
            vector_size: Dimension of vectors (must match embeddings)
            in_memory: If True, use in-memory storage (no persistence)
            storage_path: Path for persistent storage (ignored if in_memory)

        Raises:
            RuntimeError: If client initialization fails
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.in_memory = in_memory

        logger.info(f"Initializing Qdrant database: {collection_name}")

        try:
            if in_memory:
                logger.info("Using in-memory mode")
                self.client = QdrantClient(":memory:")
            else:
                path = Path(storage_path)
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using persistent storage: {path}")
                self.client = QdrantClient(path=str(path))

            self.initialize_collection()

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise RuntimeError(
                f"Could not initialize Qdrant database: {e}"
            ) from e

    def initialize_collection(self) -> None:
        """
        Create the collection if it doesn't already exist.

        Uses cosine distance and HNSW index with configured parameters.
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists:
            logger.info(f"Collection '{self.collection_name}' already exists")
            return

        logger.info(f"Creating collection '{self.collection_name}'")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(
                m=HNSW_M,
                ef_construct=HNSW_EF_CONSTRUCT,
            ),
        )

        logger.info("Collection created successfully")

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
            ValueError: If input lengths don't match or vectors have wrong dimension
            RuntimeError: If insertion fails
        """
        if len(ids) != len(vectors) or len(ids) != len(metadata):
            raise ValueError(
                f"Input lengths must match: ids={len(ids)}, "
                f"vectors={len(vectors)}, metadata={len(metadata)}"
            )

        if not ids:
            raise ValueError("Cannot insert empty list")

        # Validate vector dimensions
        for i, vector in enumerate(vectors):
            if len(vector) != self.vector_size:
                raise ValueError(
                    f"Vector at index {i} has {len(vector)} dimensions, "
                    f"expected {self.vector_size}"
                )

        try:
            points = [
                PointStruct(
                    id=id_,
                    vector=vector,
                    payload=meta,
                )
                for id_, vector, meta in zip(ids, vectors, metadata)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.info(f"Inserted {len(points)} vectors")

        except Exception as e:
            logger.error(f"Failed to insert vectors: {e}")
            raise RuntimeError(f"Vector insertion failed: {e}") from e

    def search_similar(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for vectors most similar to the query.

        Args:
            query_vector: The query embedding vector (384 floats)
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1). If None,
                returns all results up to limit.

        Returns:
            List of dicts with keys: id, score, metadata.
            Sorted by score descending (most similar first).

        Raises:
            ValueError: If query vector has wrong dimension
            RuntimeError: If search fails
        """
        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"Query vector has {len(query_vector)} dimensions, "
                f"expected {self.vector_size}"
            )

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
            ).points

            return [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "metadata": point.payload or {},
                }
                for point in results
            ]

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise RuntimeError(f"Vector search failed: {e}") from e

    def delete_by_id(self, ids: list[str]) -> None:
        """
        Delete vectors by their IDs.

        Args:
            ids: List of vector IDs to delete

        Raises:
            ValueError: If ids list is empty
            RuntimeError: If deletion fails
        """
        if not ids:
            raise ValueError("Cannot delete empty list of IDs")

        try:
            from qdrant_client.models import PointIdsList

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids),
            )

            logger.info(f"Deleted {len(ids)} vectors")

        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise RuntimeError(f"Vector deletion failed: {e}") from e

    def get_collection_info(self) -> dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dict with: collection_name, vectors_count, vector_size,
            distance_metric, status
        """
        try:
            info = self.client.get_collection(self.collection_name)

            return {
                "collection_name": self.collection_name,
                "vectors_count": info.points_count,
                "vector_size": self.vector_size,
                "distance_metric": DISTANCE_METRIC,
                "status": str(info.status),
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise RuntimeError(f"Could not get collection info: {e}") from e
