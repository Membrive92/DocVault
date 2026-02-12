"""
Embeddings module for DocVault.

This module provides local embedding generation using sentence-transformers.
"""

from __future__ import annotations

from .embedding_service import EmbeddingService

__all__ = [
    "EmbeddingService",
]
