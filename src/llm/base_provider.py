"""
Abstract base class for LLM providers.

Defines the interface that all LLM providers must implement,
following the Strategy Pattern used throughout the project.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional


class LLMProvider(ABC):
    """
    Abstract interface for LLM providers.

    All LLM providers must implement this interface to work with DocVault.
    Same pattern as DocumentParser (M4) and VectorDatabase (M3).
    """

    def __init__(self, model: Optional[str] = None) -> None:
        """
        Initialize provider.

        Args:
            model: Model identifier (provider-specific).
        """
        self.model = model

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: User query or instruction.
            context: Optional context from RAG retrieval.
            temperature: Sampling temperature (0=deterministic, 1=creative).
            max_tokens: Maximum tokens in response.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If generation fails.
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """
        Generate a streaming response from the LLM.

        Args:
            prompt: User query or instruction.
            context: Optional context from RAG retrieval.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Yields:
            Chunks of generated text.

        Raises:
            RuntimeError: If generation fails.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, str]:
        """
        Get information about the current model and provider.

        Returns:
            Dictionary with provider and model metadata.
        """
        pass

    def format_prompt_with_context(
        self,
        prompt: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Format a prompt with RAG context.

        Provides a default RAG prompt template. Providers can override
        this method if they need a different format.

        Args:
            prompt: User query.
            context: Retrieved document chunks.

        Returns:
            Formatted prompt string.
        """
        if not context:
            return prompt

        return (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {prompt}\n\n"
            "Answer based on the context above. "
            "If the context doesn't contain enough information, say so."
        )
