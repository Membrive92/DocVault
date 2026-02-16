# Milestone 6: Flexible LLM Layer

**Status:** ✅ Done
**Dependencies:** M1 (Foundation — config/settings.py)
**Goal:** Implement provider-agnostic LLM abstraction using Strategy Pattern

---

## Overview

This milestone creates a flexible LLM layer that allows switching between different AI providers without code changes. The Strategy Pattern provides a clean abstraction that prevents vendor lock-in and enables easy testing and scaling.

**Key result:** Change `LLM_PROVIDER=ollama_local` to `LLM_PROVIDER=openai` in `.env` — no code changes required.

## The Problem

Different LLM providers have different:
- API interfaces (`ollama.Client.generate()` vs `OpenAI.chat.completions.create()` vs `Anthropic.messages.create()`)
- Authentication methods (none for local, API keys for cloud)
- Request/response formats (dict vs objects vs streaming context managers)
- Pricing models ($0 local vs $30/1M tokens cloud)

**Without abstraction:** Changing providers requires rewriting all LLM-dependent code.

**With Strategy Pattern:** Change a single environment variable.

## Architecture — Strategy Pattern

```
┌──────────────────────────────────────────────────────┐
│              LLMProvider (ABC)                        │
│                                                       │
│  + generate(prompt, context, temperature, max_tokens) │
│  + generate_stream(prompt, context, ...) -> Iterator  │
│  + get_model_info() -> dict                           │
│  + format_prompt_with_context(prompt, context) -> str │
└──────────────────────────────────────────────────────┘
                        ▲
                        │ implements
        ┌───────────────┼───────────────┐
        │               │               │
┌───────────────┐ ┌──────────────┐ ┌─────────────┐
│ OllamaProvider│ │ OpenAI       │ │ Anthropic   │
│               │ │ Provider     │ │ Provider    │
│ local or      │ │              │ │             │
│ server URL    │ │ GPT-4, etc.  │ │ Claude, etc.│
└───────────────┘ └──────────────┘ └─────────────┘
```

### Provider Factory

```
┌──────────────────────────────────────────────────────┐
│              LLMProviderFactory                       │
│                                                       │
│  + create_provider(provider_type, model) -> LLMProvider│
│  + get_available_providers() -> list[str]             │
│                                                       │
│  Reads defaults from config/settings.py               │
│  Lazy imports for provider classes and settings        │
└──────────────────────────────────────────────────────┘
```

## Implementation

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `src/llm/config.py` | LLMProviderType enum, default models, generation params | 41 |
| `src/llm/base_provider.py` | LLMProvider ABC with abstract + concrete methods | 119 |
| `src/llm/ollama_provider.py` | OllamaProvider for local and remote Ollama servers | 117 |
| `src/llm/openai_provider.py` | OpenAIProvider for GPT-4/GPT-3.5 models | 105 |
| `src/llm/anthropic_provider.py` | AnthropicProvider for Claude models | 101 |
| `src/llm/provider_factory.py` | LLMProviderFactory with lazy imports | 93 |
| `src/llm/__init__.py` | Module exports | 25 |

### Files Modified

| File | Changes |
|------|---------|
| `config/settings.py` | Added 7 LLM fields (provider, model, server_url, temperature, max_tokens, api keys) |
| `.env.example` | Added LLM configuration variables |
| `requirements.txt` | Added openai, anthropic, ollama SDKs |

### Key Components

#### 1. LLMProviderType Enum (`config.py`)

```python
class LLMProviderType(str, Enum):
    OLLAMA_LOCAL = "ollama_local"
    OLLAMA_SERVER = "ollama_server"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
```

Default models per provider:
- `ollama_local` / `ollama_server`: `llama3.2:3b`
- `openai`: `gpt-4`
- `anthropic`: `claude-3-5-sonnet-20241022`

#### 2. LLMProvider ABC (`base_provider.py`)

Abstract methods all providers implement:
- `generate(prompt, context, temperature, max_tokens) -> str` — sync response
- `generate_stream(prompt, context, temperature, max_tokens) -> Iterator[str]` — streaming
- `get_model_info() -> dict[str, str]` — provider metadata

Concrete method (shared by all):
- `format_prompt_with_context(prompt, context) -> str` — RAG prompt template

#### 3. Provider Implementations

**OllamaProvider:** Uses `ollama.Client` SDK. Supports both local (no URL) and remote (custom URL) servers. Uses `client.generate()` for both sync and streaming (with `stream=True`).

**OpenAIProvider:** Uses `OpenAI` SDK. Uses `client.chat.completions.create()` for sync and streaming (with `stream=True`). Reads `delta.content` from stream chunks.

**AnthropicProvider:** Uses `Anthropic` SDK. Uses `client.messages.create()` for sync and `client.messages.stream()` context manager for streaming. Reads from `stream.text_stream`.

#### 4. LLMProviderFactory (`provider_factory.py`)

Key design decisions:
- **Lazy import of settings:** `from config.settings import settings` inside `create_provider()` method body, not at module level. Prevents import issues in tests.
- **Lazy import of providers:** Each provider class imported inside its `if`-branch. Only loads the SDK you actually use.
- **Falls back to config/settings.py:** If `provider_type` or `model` not passed, reads from `settings.llm_provider` and `settings.llm_model`.

```python
# Usage — from config
provider = LLMProviderFactory.create_provider()

# Usage — explicit
provider = LLMProviderFactory.create_provider(
    provider_type="openai",
    model="gpt-4",
    api_key="sk-..."
)
```

### Configuration (`config/settings.py`)

Added fields:
```python
llm_provider: str = "ollama_local"
llm_model: Optional[str] = None
llm_server_url: Optional[str] = None
llm_temperature: float = 0.7
llm_max_tokens: int = 1024
openai_api_key: Optional[str] = None
anthropic_api_key: Optional[str] = None
```

## Testing

### Unit Tests (28 tests)

| Test Class | Tests | What it tests |
|-----------|-------|---------------|
| `TestLLMConfig` | 3 | Enum values, default models, generation defaults |
| `TestBaseProvider` | 3 | format_prompt without/with/empty context |
| `TestOllamaProvider` | 6 | Init default, init custom URL, generate, stream, error, model_info |
| `TestOpenAIProvider` | 5 | Init, generate, stream, error, model_info |
| `TestAnthropicProvider` | 5 | Init, generate, stream, error, model_info |
| `TestLLMProviderFactory` | 6 | Create each provider type, invalid type, available list |

All provider tests use `MagicMock` + `@patch` to mock SDK clients.

Factory tests patch `config.settings.settings` (the actual settings object) because the factory uses lazy imports.

### Integration Tests (3 tests — Ollama only)

| Test | What it verifies |
|------|-----------------|
| `test_ollama_generate` | Real generation with local Ollama |
| `test_ollama_stream` | Real streaming with local Ollama |
| `test_ollama_model_info` | Model metadata from running Ollama |

Auto-skip with `pytest.mark.skipif` when Ollama is not running.

Only Ollama is tested in integration (free, local). OpenAI/Anthropic require API keys.

### Running Tests

```bash
# Unit tests only (fast, no services needed)
pytest tests/unit/test_llm.py -v

# Integration tests (requires Ollama running)
pytest tests/integration/test_llm_integration.py -v

# All M6 tests
pytest tests/ -k llm -v

# All project tests
pytest tests/ -v
```

**Results:** 28 unit tests passed, 3 integration tests skipped (Ollama not running). 157 total project tests passed.

## Usage Examples

### Basic Usage
```python
from src.llm import LLMProviderFactory

# Create provider from .env configuration
provider = LLMProviderFactory.create_provider()

# Sync generation
response = provider.generate(
    prompt="What is machine learning?",
    temperature=0.7
)
print(response)

# Streaming generation
for chunk in provider.generate_stream(prompt="Explain RAG"):
    print(chunk, end="")
```

### RAG Integration (Preview of M7)
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
context = "\n\n".join([r.metadata["chunk_text"] for r in results])

# 4. Generate answer with LLM (uses built-in RAG template)
answer = llm.generate(prompt=query, context=context)
print(answer)
```

### Switching Providers
```env
# Development: Free local model
LLM_PROVIDER=ollama_local
LLM_MODEL=llama3.2:3b

# Production: High-quality commercial model
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...

# Alternative: Anthropic Claude
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-...
```

No code changes required — just update `.env`.

## Differences from Original Spec

| Aspect | Original doc | Actual implementation |
|--------|-------------|----------------------|
| Factory config | `os.getenv()` directly | `config/settings.py` (project convention) |
| Scripts | `scripts/test_llm.py` | Not created (scripts eliminated since M3) |
| Test location | `tests/test_llm_providers.py` | `tests/unit/test_llm.py` + `tests/integration/test_llm_integration.py` |
| Logging | f-strings | `%s` formatting (best practice) |
| Integration tests | All providers | Only Ollama (free, local) |
| Provider classes | Separate OllamaLocal/OllamaServer | Single OllamaProvider with `server_url` param |

## Cost Comparison

| Provider | Cost (per 1M tokens) | Quality | Latency | Privacy |
|----------|---------------------|---------|---------|---------|
| Ollama Local | $0 | Good | Fast | 100% Private |
| OpenAI GPT-4 | ~$30 | Excellent | Medium | Shared with OpenAI |
| Anthropic Claude | ~$15 | Excellent | Medium | Shared with Anthropic |

## Next Steps (M7)

M7 integrates everything into the complete RAG pipeline:
- Combine vector search (M3) + LLM generation (M6) into RAGPipeline
- FastAPI REST endpoints
- Interactive CLI
- Streaming responses
- Source citations
- End-to-end testing

---

**Related Files:**
- `src/llm/config.py` — LLM configuration and provider types
- `src/llm/base_provider.py` — Abstract LLM interface
- `src/llm/ollama_provider.py` — Ollama implementation
- `src/llm/openai_provider.py` — OpenAI implementation
- `src/llm/anthropic_provider.py` — Anthropic implementation
- `src/llm/provider_factory.py` — Provider factory
- `src/llm/__init__.py` — Module exports
- `tests/unit/test_llm.py` — 28 unit tests
- `tests/integration/test_llm_integration.py` — 3 integration tests
