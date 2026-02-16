"""
Data models for the RAG pipeline.

Defines the structured types used throughout the RAG pipeline
for sources, responses, and configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import DEFAULT_MIN_SIMILARITY, DEFAULT_TOP_K


@dataclass
class Source:
    """A retrieved document chunk with its metadata."""

    chunk_text: str
    source_file: str
    chunk_index: int
    similarity_score: float
    document_title: Optional[str] = None
    document_format: Optional[str] = None


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""

    answer: str
    sources: list[Source]
    query: str
    model_used: str
    retrieval_count: int
    retrieval_time_ms: Optional[float] = None
    generation_time_ms: Optional[float] = None


@dataclass
class RAGConfig:
    """Configuration for a RAG pipeline instance."""

    top_k: int = DEFAULT_TOP_K
    temperature: float = 0.7
    max_tokens: int = 1024
    min_similarity: float = DEFAULT_MIN_SIMILARITY
