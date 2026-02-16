"""
Flexible LLM layer module.

Provides a provider-agnostic LLM abstraction using the Strategy Pattern.
Switch between Ollama, OpenAI, and Anthropic via configuration.
"""

from __future__ import annotations

from .anthropic_provider import AnthropicProvider
from .base_provider import LLMProvider
from .config import LLMProviderType
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .provider_factory import LLMProviderFactory

__all__ = [
    "AnthropicProvider",
    "LLMProvider",
    "LLMProviderFactory",
    "LLMProviderType",
    "OllamaProvider",
    "OpenAIProvider",
]
