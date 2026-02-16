"""
Ollama LLM provider implementation.

Supports both local (localhost:11434) and custom server URLs.
Uses the official ollama Python SDK.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import ollama

from .base_provider import LLMProvider
from .config import DEFAULT_MODELS, LLMProviderType

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider.

    Supports both local Ollama installations (localhost:11434) and
    remote Ollama servers via custom URL.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        server_url: Optional[str] = None,
    ) -> None:
        """
        Initialize Ollama provider.

        Args:
            model: Ollama model name (e.g., "llama3.2:3b").
                   Defaults to config default if None.
            server_url: Custom server URL. None uses localhost:11434.
        """
        super().__init__(model)

        self.model = model or DEFAULT_MODELS[LLMProviderType.OLLAMA_LOCAL]
        self.server_url = server_url

        if server_url:
            self.client = ollama.Client(host=server_url)
            logger.info("Ollama client initialized: %s", server_url)
        else:
            self.client = ollama.Client()
            logger.info("Ollama client initialized: localhost")

        logger.info("Using model: %s", self.model)

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate response using Ollama."""
        full_prompt = self.format_prompt_with_context(prompt, context)

        try:
            response = self.client.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )

            return response["response"]

        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            raise RuntimeError(f"Failed to generate response: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Generate streaming response using Ollama."""
        full_prompt = self.format_prompt_with_context(prompt, context)

        try:
            stream = self.client.generate(
                model=self.model,
                prompt=full_prompt,
                stream=True,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )

            for chunk in stream:
                yield chunk["response"]

        except Exception as e:
            logger.error("Ollama streaming failed: %s", e)
            raise RuntimeError(f"Failed to stream response: {e}") from e

    def get_model_info(self) -> dict[str, str]:
        """Get model information."""
        return {
            "provider": "ollama",
            "model": self.model,
            "server_url": self.server_url or "localhost:11434",
        }
