# Milestone 6: Flexible LLM Layer

**Status:** ⏸️ Pending
**Dependencies:** M1 (Foundation)
**Goal:** Implement provider-agnostic LLM abstraction using Strategy Pattern

---

## Overview

This milestone creates a flexible LLM layer that allows switching between different AI providers without code changes. The Strategy Pattern provides a clean abstraction that prevents vendor lock-in and enables easy testing and scaling.

## The Problem

Different LLM providers have different:
- API interfaces
- Authentication methods
- Request/response formats
- Pricing models
- Performance characteristics

**Without abstraction:** Changing providers requires rewriting all LLM-dependent code.

**With Strategy Pattern:** Change a single environment variable.

## Architecture - Strategy Pattern

```
┌──────────────────────────────────────────────────────┐
│              LLMProvider (ABC)                       │
│                                                       │
│  + generate(prompt, context) -> str                  │
│  + generate_stream(prompt, context) -> Iterator      │
│  + get_model_info() -> dict                          │
└──────────────────────────────────────────────────────┘
                        ▲
                        │ implements
        ┌───────────────┼──────────────┬───────────────┐
        │               │              │               │
┌───────────────┐ ┌────────────┐ ┌──────────────┐ ┌─────────────┐
│ OllamaLocal   │ │OllamaServer│ │ OpenAI       │ │ Anthropic   │
│ Provider      │ │ Provider   │ │ Provider     │ │ Provider    │
│               │ │            │ │              │ │             │
│ localhost     │ │ custom URL │ │ API key      │ │ API key     │
│ :11434        │ │            │ │ gpt-4        │ │ claude-3    │
└───────────────┘ └────────────┘ └──────────────┘ └─────────────┘
```

### Provider Factory

```
┌──────────────────────────────────────────────────────┐
│              LLMProviderFactory                      │
│                                                       │
│  + create_provider(provider_name) -> LLMProvider     │
│  + get_available_providers() -> list[str]            │
└──────────────────────────────────────────────────────┘
                        │
                        │ creates based on config
                        ▼
                  LLMProvider instance
```

## Why Strategy Pattern?

### Benefits
1. **Vendor Independence**: Switch providers without code changes
2. **Testing**: Mock providers for unit tests
3. **Cost Optimization**: Start local, scale to commercial when needed
4. **Reliability**: Fallback to different providers if one fails
5. **Flexibility**: Use different models for different tasks

### Real-World Scenario
```python
# Development: Free local model
LLM_PROVIDER=ollama_local
LLM_MODEL=llama3.2:3b

# Production: High-quality commercial model
LLM_PROVIDER=openai
LLM_MODEL=gpt-4

# No code changes required!
```

## Implementation Plan

### Task 1: Install Provider SDKs

```bash
pip install openai anthropic ollama
```

**Why official SDKs?**
- Type hints included
- Maintained by providers
- Handle auth, retries, errors

### Task 2: LLM Configuration

**File:** `src/llm/config.py`

```python
"""
Configuration for LLM providers.
"""

from __future__ import annotations

from enum import Enum


class LLMProviderType(str, Enum):
    """Available LLM provider types."""
    OLLAMA_LOCAL = "ollama_local"
    OLLAMA_SERVER = "ollama_server"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Default models for each provider
DEFAULT_MODELS = {
    LLMProviderType.OLLAMA_LOCAL: "llama3.2:3b",
    LLMProviderType.OLLAMA_SERVER: "llama3.2:3b",
    LLMProviderType.OPENAI: "gpt-4",
    LLMProviderType.ANTHROPIC: "claude-3-5-sonnet-20241022",
}

# Generation parameters
DEFAULT_TEMPERATURE = 0.7  # Creativity (0=deterministic, 1=creative)
DEFAULT_MAX_TOKENS = 1024  # Max response length
DEFAULT_TOP_P = 0.9        # Nucleus sampling

# Timeouts
REQUEST_TIMEOUT = 60  # seconds
STREAM_TIMEOUT = 120  # seconds for streaming
```

### Task 3: Abstract LLM Interface

**File:** `src/llm/base_provider.py`

```python
"""
Abstract base class for LLM providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional


class LLMProvider(ABC):
    """
    Abstract interface for LLM providers.

    All LLM providers must implement this interface to work with DocVault.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        """
        Initialize provider.

        Args:
            model: Model identifier (provider-specific)
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
            prompt: User query/instruction
            context: Optional context (RAG retrieved documents)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response

        Raises:
            RuntimeError: If generation fails
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
            prompt: User query/instruction
            context: Optional context
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Yields:
            Chunks of generated text

        Raises:
            RuntimeError: If generation fails
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, str]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model metadata
        """
        pass

    def format_prompt_with_context(
        self,
        prompt: str,
        context: Optional[str] = None
    ) -> str:
        """
        Format prompt with context for RAG.

        Args:
            prompt: User query
            context: Retrieved document chunks

        Returns:
            Formatted prompt
        """
        if not context:
            return prompt

        return f"""Use the following context to answer the question.

Context:
{context}

Question: {prompt}

Answer based on the context above. If the context doesn't contain enough information, say so."""
```

### Task 4: Ollama Provider (Local)

**File:** `src/llm/ollama_provider.py`

```python
"""
Ollama provider implementation (local and server).
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import ollama

from .base_provider import LLMProvider
from .config import DEFAULT_MODELS, LLMProviderType, REQUEST_TIMEOUT


logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider.

    Supports both local (localhost:11434) and custom server URLs.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        server_url: Optional[str] = None,
    ) -> None:
        """
        Initialize Ollama provider.

        Args:
            model: Ollama model name (e.g., "llama3.2:3b")
            server_url: Custom server URL (None = localhost:11434)
        """
        super().__init__(model)

        self.model = model or DEFAULT_MODELS[LLMProviderType.OLLAMA_LOCAL]
        self.server_url = server_url

        # Create client
        if server_url:
            self.client = ollama.Client(host=server_url)
            logger.info(f"Ollama client initialized: {server_url}")
        else:
            self.client = ollama.Client()
            logger.info("Ollama client initialized: localhost")

        logger.info(f"Using model: {self.model}")

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
                }
            )

            return response['response']

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
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
                }
            )

            for chunk in stream:
                yield chunk['response']

        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise RuntimeError(f"Failed to stream response: {e}") from e

    def get_model_info(self) -> dict[str, str]:
        """Get model information."""
        return {
            "provider": "ollama",
            "model": self.model,
            "server_url": self.server_url or "localhost:11434",
        }
```

### Task 5: OpenAI Provider

**File:** `src/llm/openai_provider.py`

```python
"""
OpenAI provider implementation.
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

    Supports GPT-4, GPT-3.5, and other OpenAI models.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize OpenAI provider.

        Args:
            model: OpenAI model name (e.g., "gpt-4")
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        """
        super().__init__(model)

        self.model = model or DEFAULT_MODELS[LLMProviderType.OPENAI]

        # Create client (API key from env if not provided)
        self.client = OpenAI(api_key=api_key)

        logger.info(f"OpenAI client initialized: {self.model}")

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate response using OpenAI."""
        full_prompt = self.format_prompt_with_context(prompt, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
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
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise RuntimeError(f"Failed to stream response: {e}") from e

    def get_model_info(self) -> dict[str, str]:
        """Get model information."""
        return {
            "provider": "openai",
            "model": self.model,
        }
```

### Task 6: Anthropic Provider

**File:** `src/llm/anthropic_provider.py`

```python
"""
Anthropic provider implementation.
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

    Supports Claude models.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize Anthropic provider.

        Args:
            model: Claude model name
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
        """
        super().__init__(model)

        self.model = model or DEFAULT_MODELS[LLMProviderType.ANTHROPIC]

        # Create client
        self.client = Anthropic(api_key=api_key)

        logger.info(f"Anthropic client initialized: {self.model}")

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate response using Anthropic."""
        full_prompt = self.format_prompt_with_context(prompt, context)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
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
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise RuntimeError(f"Failed to stream response: {e}") from e

    def get_model_info(self) -> dict[str, str]:
        """Get model information."""
        return {
            "provider": "anthropic",
            "model": self.model,
        }
```

### Task 7: Provider Factory

**File:** `src/llm/provider_factory.py`

```python
"""
Factory for creating LLM providers based on configuration.
"""

from __future__ import annotations

import os
from typing import Optional

from .anthropic_provider import AnthropicProvider
from .base_provider import LLMProvider
from .config import LLMProviderType
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.

    Reads from environment variables or accepts explicit configuration.
    """

    @staticmethod
    def create_provider(
        provider_type: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMProvider:
        """
        Create LLM provider based on configuration.

        Args:
            provider_type: Provider type (or from LLM_PROVIDER env var)
            model: Model name (or from LLM_MODEL env var)
            **kwargs: Additional provider-specific arguments

        Returns:
            Configured LLM provider instance

        Raises:
            ValueError: If provider type is unsupported
        """
        # Get from env vars if not provided
        provider_type = provider_type or os.getenv("LLM_PROVIDER", "ollama_local")
        model = model or os.getenv("LLM_MODEL")

        provider_enum = LLMProviderType(provider_type)

        if provider_enum == LLMProviderType.OLLAMA_LOCAL:
            return OllamaProvider(model=model, server_url=None)

        elif provider_enum == LLMProviderType.OLLAMA_SERVER:
            server_url = kwargs.get("server_url") or os.getenv("LLM_SERVER_URL")
            return OllamaProvider(model=model, server_url=server_url)

        elif provider_enum == LLMProviderType.OPENAI:
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            return OpenAIProvider(model=model, api_key=api_key)

        elif provider_enum == LLMProviderType.ANTHROPIC:
            api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            return AnthropicProvider(model=model, api_key=api_key)

        else:
            raise ValueError(f"Unsupported provider: {provider_type}")

    @staticmethod
    def get_available_providers() -> list[str]:
        """Get list of available provider types."""
        return [p.value for p in LLMProviderType]
```

### Task 8: Update Configuration

**File:** `.env`

Add:
```env
# LLM Configuration
LLM_PROVIDER=ollama_local  # ollama_local, ollama_server, openai, anthropic
LLM_MODEL=llama3.2:3b      # Provider-specific model name

# Provider-specific settings
LLM_SERVER_URL=http://localhost:11434  # For ollama_server
OPENAI_API_KEY=sk-...     # For OpenAI
ANTHROPIC_API_KEY=sk-ant-...  # For Anthropic
```

**File:** `config/settings.py`

Add:
```python
# LLM settings
llm_provider: str = "ollama_local"
llm_model: Optional[str] = None
llm_server_url: Optional[str] = None
```

### Task 9: Unit Tests

**File:** `tests/test_llm_providers.py`

Test each provider:
- Mock API responses
- Test generation (sync and stream)
- Test error handling
- Test factory creation

### Task 10: Interactive Verification

**File:** `scripts/test_llm.py`

Interactive script:
1. Create provider via factory
2. Test simple generation
3. Test with RAG context
4. Test streaming
5. Display model info

## Usage Examples

### Basic Usage
```python
from src.llm import LLMProviderFactory

# Create provider from environment
provider = LLMProviderFactory.create_provider()

# Generate response
response = provider.generate(
    prompt="What is machine learning?",
    temperature=0.7
)

print(response)
```

### RAG Integration
```python
from src.embeddings import EmbeddingService
from src.database import QdrantDatabase
from src.llm import LLMProviderFactory

# Initialize services
embeddings = EmbeddingService()
vector_db = QdrantDatabase()
llm = LLMProviderFactory.create_provider()

# User query
query = "How do I install Docker?"

# 1. Generate query embedding
query_embedding = embeddings.generate_embedding(query)

# 2. Search vector database
results = vector_db.search_similar(query_embedding, limit=3)

# 3. Format context from results
context = "\n\n".join([r['chunk_text'] for r in results])

# 4. Generate answer with LLM
answer = llm.generate(
    prompt=query,
    context=context
)

print(answer)
```

### Switching Providers
```python
# Development: Free local model
os.environ["LLM_PROVIDER"] = "ollama_local"
os.environ["LLM_MODEL"] = "llama3.2:3b"
provider = LLMProviderFactory.create_provider()

# Production: High-quality commercial model
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_MODEL"] = "gpt-4"
provider = LLMProviderFactory.create_provider()

# Same code, different providers!
```

## Cost Comparison

| Provider | Cost (per 1M tokens) | Quality | Latency | Privacy |
|----------|---------------------|---------|---------|---------|
| Ollama Local | $0 | Good | Fast | 100% Private |
| OpenAI GPT-4 | ~$30 | Excellent | Medium | Shared with OpenAI |
| Anthropic Claude | ~$15 | Excellent | Medium | Shared with Anthropic |

## Next Steps (M7)

M7 integrates everything into the complete RAG pipeline:
- Combine vector search + LLM generation
- FastAPI endpoints
- Interactive CLI
- End-to-end testing

---

**Related Files:**
- `src/llm/config.py` - LLM configuration
- `src/llm/base_provider.py` - Abstract interface
- `src/llm/ollama_provider.py` - Ollama implementation
- `src/llm/openai_provider.py` - OpenAI implementation
- `src/llm/anthropic_provider.py` - Anthropic implementation
- `src/llm/provider_factory.py` - Provider factory
- `tests/test_llm_providers.py` - Unit tests
- `scripts/test_llm.py` - Interactive verification
