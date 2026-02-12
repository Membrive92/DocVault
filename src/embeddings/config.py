"""
Embedding models configuration for DocVault.

This module defines available embedding models and their specifications.
"""

from __future__ import annotations

from typing import Literal


# Embedding model configuration
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Model specifications
MODEL_DIMENSIONS = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
}

# Supported languages for the default model
SUPPORTED_LANGUAGES = ["en", "es"]  # English and Spanish

# Model type alias
EmbeddingModelType = Literal["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
