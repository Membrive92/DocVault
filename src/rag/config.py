"""
RAG pipeline configuration constants.

Centralized defaults for the RAG pipeline. These can be overridden
via RAGConfig or environment variables in config/settings.py.
"""

from __future__ import annotations

# Retrieval defaults
DEFAULT_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.3

# Context assembly
MAX_CONTEXT_CHUNKS = 10
CONTEXT_SEPARATOR = "\n\n"

# Response when no relevant documents are found
NO_RESULTS_MESSAGE = (
    "I couldn't find any relevant information in the indexed documents "
    "to answer your question. Please try rephrasing your query or ensure "
    "that relevant documents have been ingested."
)
