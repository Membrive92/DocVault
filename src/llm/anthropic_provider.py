"""
Anthropic LLM provider implementation.

Supports Claude models via the official Anthropic Python SDK.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from anthropic import Anthropic

from .base_provider import LLMProvider
from .config import DEFAULT_MODELS, LLMProviderType

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """
    Anthropic LLM provider.

    Uses the Anthropic Messages API for text generation with Claude models.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize Anthropic provider.

        Args:
            model: Claude model name (e.g., "claude-3-5-sonnet-20241022").
                   Defaults to config default if None.
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var if None.
        """
        super().__init__(model)

        self.model = model or DEFAULT_MODELS[LLMProviderType.ANTHROPIC]
        self.client = Anthropic(api_key=api_key)

        logger.info("Anthropic client initialized: %s", self.model)

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate response using Anthropic Messages API."""
        full_prompt = self.format_prompt_with_context(prompt, context)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": full_prompt}],
            )

            return response.content[0].text

        except Exception as e:
            logger.error("Anthropic generation failed: %s", e)
            raise RuntimeError(f"Failed to generate response: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Generate streaming response using Anthropic."""
        full_prompt = self.format_prompt_with_context(prompt, context)

        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": full_prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error("Anthropic streaming failed: %s", e)
            raise RuntimeError(f"Failed to stream response: {e}") from e

    def get_model_info(self) -> dict[str, str]:
        """Get model information."""
        return {
            "provider": "anthropic",
            "model": self.model,
        }
