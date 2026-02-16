"""
Configuration for document ingestion pipeline.

Module-level constants for chunking, batch processing, and file filtering.
"""

from __future__ import annotations

# ==========================================
# Chunking Parameters
# ==========================================
CHUNK_SIZE = 500  # Target chunk size in tokens
CHUNK_OVERLAP = 50  # Overlap between consecutive chunks in tokens
MIN_CHUNK_SIZE = 100  # Minimum chunk size in tokens
CHARS_PER_TOKEN = 4  # Approximate characters per token

# ==========================================
# Embedding Batch Processing
# ==========================================
BATCH_SIZE = 32  # Number of texts to embed per batch

# ==========================================
# File Discovery
# ==========================================
SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown"}
SKIP_PATTERNS = [
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
]

# ==========================================
# State Tracking
# ==========================================
INDEX_STATE_FILE = "data/index_state.json"
