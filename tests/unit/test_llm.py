"""
Unit tests for the flexible LLM layer module.

Tests configuration, base provider, all three provider implementations,
and the provider factory. Uses mocks for all SDK clients to avoid
needing real API keys or running services.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.llm import (
    AnthropicProvider,
    LLMProvider,
    LLMProviderFactory,
    LLMProviderType,
    OllamaProvider,
    OpenAIProvider,
)
from src.llm.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODELS,
    DEFAULT_TEMPERATURE,
)


# ==========================================
# TestLLMConfig
# ==========================================


class TestLLMConfig:
    """Tests for LLM configuration constants."""

    def test_provider_type_values(self) -> None:
        """All expected provider types are defined."""
        assert LLMProviderType.OLLAMA_LOCAL == "ollama_local"
        assert LLMProviderType.OLLAMA_SERVER == "ollama_server"
        assert LLMProviderType.OPENAI == "openai"
        assert LLMProviderType.ANTHROPIC == "anthropic"

    def test_default_models_for_all_providers(self) -> None:
        """Every provider type has a default model."""
        for provider_type in LLMProviderType:
            assert provider_type in DEFAULT_MODELS
            assert isinstance(DEFAULT_MODELS[provider_type], str)
            assert len(DEFAULT_MODELS[provider_type]) > 0

    def test_generation_defaults(self) -> None:
        """Default generation parameters have sensible values."""
        assert DEFAULT_TEMPERATURE == 0.7
        assert DEFAULT_MAX_TOKENS == 1024


# ==========================================
# TestBaseProvider
# ==========================================


class TestBaseProvider:
    """Tests for the LLMProvider base class."""

    def _create_concrete_provider(self) -> LLMProvider:
        """Create a minimal concrete provider for testing base methods."""

        class ConcreteProvider(LLMProvider):
            def generate(self, prompt, context=None, temperature=0.7, max_tokens=1024):
                return "response"

            def generate_stream(self, prompt, context=None, temperature=0.7, max_tokens=1024):
                yield "chunk"

            def get_model_info(self):
                return {"provider": "test", "model": self.model or "test-model"}

        return ConcreteProvider()

    def test_format_prompt_without_context(self) -> None:
        """Without context, returns prompt unchanged."""
        provider = self._create_concrete_provider()
        result = provider.format_prompt_with_context("What is Python?")
        assert result == "What is Python?"

    def test_format_prompt_with_context(self) -> None:
        """With context, returns formatted RAG prompt."""
        provider = self._create_concrete_provider()
        result = provider.format_prompt_with_context(
            "What is Python?", context="Python is a programming language."
        )
        assert "Context:" in result
        assert "Python is a programming language." in result
        assert "Question: What is Python?" in result
        assert "Answer based on the context above" in result

    def test_format_prompt_with_empty_context(self) -> None:
        """Empty string context returns prompt unchanged."""
        provider = self._create_concrete_provider()
        result = provider.format_prompt_with_context("What is Python?", context="")
        assert result == "What is Python?"


# ==========================================
# TestOllamaProvider
# ==========================================


class TestOllamaProvider:
    """Tests for the OllamaProvider with mocked SDK."""

    @patch("src.llm.ollama_provider.ollama.Client")
    def test_initialization_default(self, mock_client_cls: MagicMock) -> None:
        """Default init uses default model and localhost."""
        provider = OllamaProvider()
        assert provider.model == DEFAULT_MODELS[LLMProviderType.OLLAMA_LOCAL]
        assert provider.server_url is None
        mock_client_cls.assert_called_once_with()

    @patch("src.llm.ollama_provider.ollama.Client")
    def test_initialization_custom_url(self, mock_client_cls: MagicMock) -> None:
        """Custom server URL is passed to the client."""
        provider = OllamaProvider(
            model="mistral:7b", server_url="http://remote:11434"
        )
        assert provider.model == "mistral:7b"
        assert provider.server_url == "http://remote:11434"
        mock_client_cls.assert_called_once_with(host="http://remote:11434")

    @patch("src.llm.ollama_provider.ollama.Client")
    def test_generate_success(self, mock_client_cls: MagicMock) -> None:
        """generate() returns response text."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "Hello, world!"}
        mock_client_cls.return_value = mock_client

        provider = OllamaProvider()
        result = provider.generate("Say hello")

        assert result == "Hello, world!"
        mock_client.generate.assert_called_once()

    @patch("src.llm.ollama_provider.ollama.Client")
    def test_generate_stream_success(self, mock_client_cls: MagicMock) -> None:
        """generate_stream() yields response chunks."""
        mock_client = MagicMock()
        mock_client.generate.return_value = iter([
            {"response": "Hello"},
            {"response": ", "},
            {"response": "world!"},
        ])
        mock_client_cls.return_value = mock_client

        provider = OllamaProvider()
        chunks = list(provider.generate_stream("Say hello"))

        assert chunks == ["Hello", ", ", "world!"]

    @patch("src.llm.ollama_provider.ollama.Client")
    def test_generate_failure_raises_runtime_error(
        self, mock_client_cls: MagicMock
    ) -> None:
        """generate() wraps SDK errors in RuntimeError."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = ConnectionError("Connection refused")
        mock_client_cls.return_value = mock_client

        provider = OllamaProvider()
        with pytest.raises(RuntimeError, match="Failed to generate response"):
            provider.generate("Say hello")

    @patch("src.llm.ollama_provider.ollama.Client")
    def test_get_model_info(self, mock_client_cls: MagicMock) -> None:
        """get_model_info() returns correct metadata."""
        provider = OllamaProvider(model="llama3.2:3b")
        info = provider.get_model_info()

        assert info["provider"] == "ollama"
        assert info["model"] == "llama3.2:3b"
        assert info["server_url"] == "localhost:11434"


# ==========================================
# TestOpenAIProvider
# ==========================================


class TestOpenAIProvider:
    """Tests for the OpenAIProvider with mocked SDK."""

    @patch("src.llm.openai_provider.OpenAI")
    def test_initialization(self, mock_openai_cls: MagicMock) -> None:
        """Init creates client with API key."""
        provider = OpenAIProvider(api_key="sk-test-key")
        assert provider.model == DEFAULT_MODELS[LLMProviderType.OPENAI]
        mock_openai_cls.assert_called_once_with(api_key="sk-test-key")

    @patch("src.llm.openai_provider.OpenAI")
    def test_generate_success(self, mock_openai_cls: MagicMock) -> None:
        """generate() returns message content."""
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "The answer is 42."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test")
        result = provider.generate("What is the meaning of life?")

        assert result == "The answer is 42."
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.llm.openai_provider.OpenAI")
    def test_generate_stream_success(self, mock_openai_cls: MagicMock) -> None:
        """generate_stream() yields delta content."""
        mock_client = MagicMock()

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = None

        mock_client.chat.completions.create.return_value = iter(
            [chunk1, chunk2, chunk3]
        )
        mock_openai_cls.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test")
        chunks = list(provider.generate_stream("Say hello"))

        assert chunks == ["Hello", " world"]

    @patch("src.llm.openai_provider.OpenAI")
    def test_generate_failure_raises_runtime_error(
        self, mock_openai_cls: MagicMock
    ) -> None:
        """generate() wraps API errors in RuntimeError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai_cls.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test")
        with pytest.raises(RuntimeError, match="Failed to generate response"):
            provider.generate("test")

    @patch("src.llm.openai_provider.OpenAI")
    def test_get_model_info(self, mock_openai_cls: MagicMock) -> None:
        """get_model_info() returns correct metadata."""
        provider = OpenAIProvider(model="gpt-4o", api_key="sk-test")
        info = provider.get_model_info()

        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4o"


# ==========================================
# TestAnthropicProvider
# ==========================================


class TestAnthropicProvider:
    """Tests for the AnthropicProvider with mocked SDK."""

    @patch("src.llm.anthropic_provider.Anthropic")
    def test_initialization(self, mock_anthropic_cls: MagicMock) -> None:
        """Init creates client with API key."""
        provider = AnthropicProvider(api_key="sk-ant-test")
        assert provider.model == DEFAULT_MODELS[LLMProviderType.ANTHROPIC]
        mock_anthropic_cls.assert_called_once_with(api_key="sk-ant-test")

    @patch("src.llm.anthropic_provider.Anthropic")
    def test_generate_success(self, mock_anthropic_cls: MagicMock) -> None:
        """generate() returns content text."""
        mock_client = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = "Claude responds here."
        mock_response = MagicMock()
        mock_response.content = [mock_content_block]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        provider = AnthropicProvider(api_key="sk-ant-test")
        result = provider.generate("Hello Claude")

        assert result == "Claude responds here."
        mock_client.messages.create.assert_called_once()

    @patch("src.llm.anthropic_provider.Anthropic")
    def test_generate_stream_success(self, mock_anthropic_cls: MagicMock) -> None:
        """generate_stream() yields text from stream."""
        mock_client = MagicMock()
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_stream_ctx.text_stream = iter(["Hello", " from", " Claude"])
        mock_client.messages.stream.return_value = mock_stream_ctx
        mock_anthropic_cls.return_value = mock_client

        provider = AnthropicProvider(api_key="sk-ant-test")
        chunks = list(provider.generate_stream("Say hello"))

        assert chunks == ["Hello", " from", " Claude"]

    @patch("src.llm.anthropic_provider.Anthropic")
    def test_generate_failure_raises_runtime_error(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        """generate() wraps API errors in RuntimeError."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("Auth failed")
        mock_anthropic_cls.return_value = mock_client

        provider = AnthropicProvider(api_key="sk-ant-test")
        with pytest.raises(RuntimeError, match="Failed to generate response"):
            provider.generate("test")

    @patch("src.llm.anthropic_provider.Anthropic")
    def test_get_model_info(self, mock_anthropic_cls: MagicMock) -> None:
        """get_model_info() returns correct metadata."""
        provider = AnthropicProvider(
            model="claude-3-opus-20240229", api_key="sk-ant-test"
        )
        info = provider.get_model_info()

        assert info["provider"] == "anthropic"
        assert info["model"] == "claude-3-opus-20240229"


# ==========================================
# TestLLMProviderFactory
# ==========================================


class TestLLMProviderFactory:
    """Tests for the LLMProviderFactory."""

    @patch("src.llm.ollama_provider.ollama.Client")
    @patch("config.settings.settings")
    def test_create_ollama_local(
        self, mock_settings: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        """Factory creates OllamaProvider for ollama_local."""
        mock_settings.llm_provider = "ollama_local"
        mock_settings.llm_model = None

        provider = LLMProviderFactory.create_provider(provider_type="ollama_local")

        assert isinstance(provider, OllamaProvider)
        assert provider.server_url is None

    @patch("src.llm.ollama_provider.ollama.Client")
    @patch("config.settings.settings")
    def test_create_ollama_server(
        self, mock_settings: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        """Factory creates OllamaProvider with server URL for ollama_server."""
        mock_settings.llm_provider = "ollama_server"
        mock_settings.llm_model = None
        mock_settings.llm_server_url = "http://myserver:11434"

        provider = LLMProviderFactory.create_provider(provider_type="ollama_server")

        assert isinstance(provider, OllamaProvider)
        assert provider.server_url == "http://myserver:11434"

    @patch("src.llm.openai_provider.OpenAI")
    @patch("config.settings.settings")
    def test_create_openai(
        self, mock_settings: MagicMock, mock_openai_cls: MagicMock
    ) -> None:
        """Factory creates OpenAIProvider."""
        mock_settings.llm_provider = "openai"
        mock_settings.llm_model = "gpt-4o"
        mock_settings.openai_api_key = "sk-test"

        provider = LLMProviderFactory.create_provider(provider_type="openai")

        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4o"

    @patch("src.llm.anthropic_provider.Anthropic")
    @patch("config.settings.settings")
    def test_create_anthropic(
        self, mock_settings: MagicMock, mock_anthropic_cls: MagicMock
    ) -> None:
        """Factory creates AnthropicProvider."""
        mock_settings.llm_provider = "anthropic"
        mock_settings.llm_model = None
        mock_settings.anthropic_api_key = "sk-ant-test"

        provider = LLMProviderFactory.create_provider(provider_type="anthropic")

        assert isinstance(provider, AnthropicProvider)

    def test_invalid_provider_raises_error(self) -> None:
        """Invalid provider type raises ValueError."""
        with pytest.raises(ValueError):
            LLMProviderFactory.create_provider(provider_type="nonexistent")

    def test_get_available_providers(self) -> None:
        """get_available_providers() returns all provider type values."""
        providers = LLMProviderFactory.get_available_providers()

        assert "ollama_local" in providers
        assert "ollama_server" in providers
        assert "openai" in providers
        assert "anthropic" in providers
        assert len(providers) == 4
