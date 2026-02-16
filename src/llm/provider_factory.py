"""
Factory for creating LLM providers based on configuration.

Reads defaults from config/settings.py and creates the appropriate
provider instance. Supports all LLMProviderType values.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base_provider import LLMProvider
from .config import LLMProviderType

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.

    Reads from config/settings.py or accepts explicit configuration.
    Same pattern as ParserFactory in M4.
    """

    @staticmethod
    def create_provider(
        provider_type: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: str,
    ) -> LLMProvider:
        """
        Create an LLM provider based on configuration.

        Args:
            provider_type: Provider type string. Defaults to settings.llm_provider.
            model: Model name. Defaults to settings.llm_model.
            **kwargs: Additional provider-specific arguments:
                - server_url: For ollama_server provider.
                - api_key: For openai/anthropic providers.

        Returns:
            Configured LLMProvider instance.

        Raises:
            ValueError: If provider type is unsupported.
        """
        from config.settings import settings

        provider_type = provider_type or settings.llm_provider
        model = model or settings.llm_model

        provider_enum = LLMProviderType(provider_type)

        logger.info("Creating LLM provider: %s", provider_enum.value)

        if provider_enum == LLMProviderType.OLLAMA_LOCAL:
            from .ollama_provider import OllamaProvider

            return OllamaProvider(model=model, server_url=None)

        elif provider_enum == LLMProviderType.OLLAMA_SERVER:
            from .ollama_provider import OllamaProvider

            server_url = kwargs.get("server_url") or settings.llm_server_url
            return OllamaProvider(model=model, server_url=server_url)

        elif provider_enum == LLMProviderType.OPENAI:
            from .openai_provider import OpenAIProvider

            api_key = kwargs.get("api_key") or settings.openai_api_key
            return OpenAIProvider(model=model, api_key=api_key)

        elif provider_enum == LLMProviderType.ANTHROPIC:
            from .anthropic_provider import AnthropicProvider

            api_key = kwargs.get("api_key") or settings.anthropic_api_key
            return AnthropicProvider(model=model, api_key=api_key)

        else:
            raise ValueError(f"Unsupported provider: {provider_type}")

    @staticmethod
    def get_available_providers() -> list[str]:
        """
        Get list of available provider type names.

        Returns:
            List of provider type string values.
        """
        return [p.value for p in LLMProviderType]
