"""
Configuration for LLM providers.

Module-level constants for provider types, default models, and generation parameters.
"""

from __future__ import annotations

from enum import Enum


class LLMProviderType(str, Enum):
    """Available LLM provider types."""

    OLLAMA_LOCAL = "ollama_local"
    OLLAMA_SERVER = "ollama_server"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# ==========================================
# Default Models
# ==========================================
DEFAULT_MODELS = {
    LLMProviderType.OLLAMA_LOCAL: "llama3.2:3b",
    LLMProviderType.OLLAMA_SERVER: "llama3.2:3b",
    LLMProviderType.OPENAI: "gpt-4",
    LLMProviderType.ANTHROPIC: "claude-3-5-sonnet-20241022",
}

# ==========================================
# Generation Parameters
# ==========================================
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024

# ==========================================
# Timeouts
# ==========================================
REQUEST_TIMEOUT = 60  # seconds
