"""
RAG (Retrieval-Augmented Generation) pipeline module.

Provides a complete pipeline that integrates embedding generation,
vector search, and LLM generation with source citations.
"""

from __future__ import annotations

from .models import RAGConfig, RAGResponse, Source
from .pipeline import RAGPipeline

__all__ = [
    "RAGConfig",
    "RAGPipeline",
    "RAGResponse",
    "Source",
]
