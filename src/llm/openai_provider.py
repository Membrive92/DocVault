"""
OpenAI LLM provider implementation.

Supports GPT-4, GPT-3.5-turbo, and other OpenAI chat models.
Uses the official openai Python SDK.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from openai import OpenAI

from .base_provider import LLMProvider
from .config import DEFAULT_MODELS, LLMProviderType

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider.

    Uses the OpenAI Chat Completions API for text generation.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize OpenAI provider.

        Args:
            model: OpenAI model name (e.g., "gpt-4").
                   Defaults to config default if None.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var if None.
        """
        super().__init__(model)

        self.model = model or DEFAULT_MODELS[LLMProviderType.OPENAI]
        self.client = OpenAI(api_key=api_key)

        logger.info("OpenAI client initialized: %s", self.model)

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate response using OpenAI Chat Completions."""
        full_prompt = self.format_prompt_with_context(prompt, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error("OpenAI generation failed: %s", e)
            raise RuntimeError(f"Failed to generate response: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Generate streaming response using OpenAI."""
        full_prompt = self.format_prompt_with_context(prompt, context)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("OpenAI streaming failed: %s", e)
            raise RuntimeError(f"Failed to stream response: {e}") from e

    def get_model_info(self) -> dict[str, str]:
        """Get model information."""
        return {
            "provider": "openai",
            "model": self.model,
        }
