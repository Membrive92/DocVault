"""
Vector database configuration for DocVault.

This module defines Qdrant connection settings, collection parameters,
and HNSW index configuration.
"""

from __future__ import annotations


# Collection configuration
DEFAULT_COLLECTION_NAME = "docvault_documents"
VECTOR_SIZE = 384  # Must match embedding dimension from M2
DISTANCE_METRIC = "Cosine"  # Compatible with L2-normalized embeddings

# HNSW index parameters
# m: Number of edges per node. Higher = better recall, slower search.
# ef_construct: Build-time search depth. Higher = better graph, slower indexing.
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100

# Storage configuration
DEFAULT_STORAGE_PATH = "data/qdrant_storage"
