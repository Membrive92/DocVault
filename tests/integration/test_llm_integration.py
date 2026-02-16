"""
Integration tests for the LLM layer.

Tests Ollama provider with a real running Ollama instance.
These tests are skipped automatically if Ollama is not available.
OpenAI and Anthropic tests are not included as they require paid API keys.
"""

from __future__ import annotations

import pytest

from src.llm import OllamaProvider


def _ollama_available() -> bool:
    """Check if Ollama is running and has a model available."""
    try:
        import ollama

        client = ollama.Client()
        models = client.list()
        return len(models.get("models", [])) > 0
    except Exception:
        return False


def _get_ollama_model() -> str:
    """Get the first available Ollama model name."""
    import ollama

    client = ollama.Client()
    models = client.list()
    return models["models"][0]["name"]


skip_no_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not running or no models available",
)


@skip_no_ollama
class TestOllamaIntegration:
    """Integration tests with a real Ollama instance."""

    @pytest.fixture(scope="class")
    def provider(self) -> OllamaProvider:
        """Create OllamaProvider with the first available model."""
        model = _get_ollama_model()
        return OllamaProvider(model=model)

    def test_ollama_generate(self, provider: OllamaProvider) -> None:
        """Generate a real response from Ollama."""
        response = provider.generate(
            prompt="Reply with exactly one word: hello",
            temperature=0.0,
            max_tokens=10,
        )

        assert isinstance(response, str)
        assert len(response) > 0

    def test_ollama_stream(self, provider: OllamaProvider) -> None:
        """Stream a real response from Ollama."""
        chunks = list(
            provider.generate_stream(
                prompt="Reply with exactly one word: hello",
                temperature=0.0,
                max_tokens=10,
            )
        )

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0

    def test_ollama_model_info(self, provider: OllamaProvider) -> None:
        """Get model info from a real Ollama provider."""
        info = provider.get_model_info()

        assert info["provider"] == "ollama"
        assert len(info["model"]) > 0
        assert info["server_url"] == "localhost:11434"
