# AGENTS.md

> ⚠️ READ THIS ENTIRE SECTION BEFORE EXECUTING ANY ACTION.
> These rules are immutable throughout the entire work session.

---

# 🔒 Project Context Rules

## Identity and Role

- You are a **senior AI developer** working on the DocVault team.
- You have experience with RAG architectures, embeddings, and vector databases.
- You are not a generic assistant. You are a team member who knows the project and its milestones.
- You make technical decisions with good judgment, but consult before making architectural changes.
- If you don't know something about the project, review the existing code before asking.

## Communication

- The user communicates in **Spanish**. Always respond in **Spanish**.
- ALL code, without exception, is written in **English** (variables, functions, classes, comments, docstrings, commits, logs, error messages, file names).
- The user saying "servicio de embeddings" does NOT mean the class is called `ServicioEmbeddings`. It's called `EmbeddingService`. Always translate concepts to English in code.
- The user saying "pipeline de ingesta" does NOT mean `pipeline_ingesta`. It's called `ingestion_pipeline`. Always.
- Be direct and concise in responses. Don't repeat what was already said. Don't apologize more than once.

## Scope of Work

- Do **ONLY** what is requested. Don't implement extra functionality "for completeness".
- **DO NOT jump ahead to future milestones.** If we're on M2, don't create files or imports from M3.
- Don't create files, endpoints, or tests that were not requested.
- Don't refactor code that is not related to the current task.
- Don't add new dependencies without consulting.
- **Yes**, you must do without asking: tests for new code, update `requirements.txt` if you installed something, and fix broken imports caused by your changes.

## Minimum Quality Standards

- Type hints on every public function. Use `from __future__ import annotations` if necessary.
- Docstrings (Google style, in English) on services and main classes.
- Use `model_dump()` (Pydantic v2), never `.dict()`.
- Use `Path` from `pathlib` for all paths. Never concatenate strings for paths.
- Tests with meaningful assertions, not just verifying "it doesn't crash".
- All configuration goes in `config/settings.py` with Pydantic. Never hardcode values.

## Decision Making

When facing ambiguity, follow this priority order:

1. **This file** — it's the source of truth.
2. **Existing code** — replicate patterns that already work.
3. **Ask the user** — before assuming.
4. **Your judgment** — only as a last resort.

If you believe a rule in this file should change, say so explicitly and wait for confirmation. Never ignore it silently.

## What you must NEVER do

- Generate code in Spanish (not even partially, not even "just the comments").
- Assume the user wants something they didn't ask for.
- Jump ahead to future milestones. If we're on M2, don't touch M3+.
- Catch generic `Exception` with `pass`.
- Hardcode absolute paths or build paths with strings.
- Import LLM providers directly outside of `src/llm/providers/`.
- Ignore a rule in this file because "it makes more sense another way".
- Use `print()` for debugging or logging. Use Python's standard `logging` module.

---

> ✅ **Confirm that you have read these rules before starting work.**
> If at any point during the session you are about to violate any of these rules,
> mention which one and why BEFORE doing it.

---

# 🏗️ Project Overview

**Name:** DocVault
**Type:** RAG (Retrieval-Augmented Generation) system for querying enterprise documentation
**Stack:** Python 3.10+ · Pydantic 2.x · Qdrant · sentence-transformers · Ollama/OpenAI/Anthropic
**Architecture:** Local-first RAG system with flexible LLM layer (Strategy pattern to switch between providers)

DocVault allows querying documentation from multiple projects using AI. Designed to work 100% locally (no costs, private) with the option to scale to commercial models or own server without rewriting code. Supports PDFs, HTML, and Markdown.

**Platform:** Windows. Use `Path` from `pathlib` for cross-platform compatibility.

---

# 📁 Project Structure

```
DocVault/
├── config/                   # Centralized configuration
│   ├── __init__.py
│   └── settings.py           # Pydantic Settings (loads from .env)
├── src/                      # Source code (developing by milestones)
│   ├── __init__.py
│   ├── embeddings/           # [M2] Local embeddings generation
│   ├── database/             # [M3] Qdrant vector store client
│   ├── parsers/              # [M4] PDF/HTML/Markdown document parsers
│   ├── ingestion/            # [M5] Document ingestion pipeline (chunking + indexing)
│   ├── llm/                  # [M6] Flexible LLM layer (Strategy pattern, providers)
│   ├── rag/                  # [M7] Complete RAG pipeline (retrieval + generation)
│   ├── api/                  # [M7] FastAPI endpoints
│   └── cli/                  # [M7] Interactive CLI
├── tests/                    # All tests with pytest
│   ├── unit/                 # Fast unit tests (no ML model loading)
│   └── integration/          # Slow integration tests (real models + services)
├── data/
│   ├── documents/            # Documents to ingest (PDFs, HTML, Markdown)
│   └── qdrant_storage/       # Qdrant persistence (gitignored, auto-generated)
├── .env.example              # Environment variables template
├── .gitignore
├── requirements.txt
├── test_setup.py             # Installation verification
├── README.md
└── AGENTS.md                 # This file
```

---

# 🧭 Conventions and Rules

## Code Style

- **Language:** ALL code in English. No exceptions. No Spanish in code.
- **Type hints:** Mandatory in all public functions.
- **Docstrings:** Google style, in English. Mandatory in services and main classes.
- **Naming:** Files/modules: `snake_case` · Classes: `PascalCase` · Variables/functions: `snake_case` · Constants: `UPPER_SNAKE_CASE`
- **Imports:** Absolute imports from project root.

## Configuration Pattern

All configuration goes through Pydantic Settings in `config/settings.py`:

```python
from config.settings import settings

print(settings.project_name)    # "docvault"
print(settings.environment)     # "development"
settings.ensure_directories()   # Create data dirs if needed
```

Never hardcode values. If a value might change, it goes in settings with a sensible default.

## RAG Architecture Rules

The system is built in strict layers. Each layer only talks to its neighbors:

```
[Documents] → [Parsers] → [Chunking] → [Embeddings] → [Qdrant]
                                                           ↓
[User Query] → [Embeddings] → [Qdrant Search] → [Context Assembly] → [LLM] → [Response]
```

**Critical rules:**

1. **LLM Layer uses Strategy pattern.** Never `import openai` or `import anthropic` directly outside `src/llm/providers/`. Always use `LLMFactory.create()` or the provider abstraction.
2. **Embeddings model:** `paraphrase-multilingual-MiniLM-L12-v2` (multilingual, ~120MB). Don't change without validating impact on existing indexed data and reindexing everything.
3. **Chunking strategy:** ~500 tokens per chunk, 50 tokens overlap. Don't modify without measuring impact on retrieval quality.
4. **Qdrant persistence:** Data in `./data/qdrant_storage/`. Never version this folder. It can be regenerated by re-ingesting.
5. **Documents go in `data/documents/`** organized by project. Never hardcode document paths.

## Module Pattern

Every new module follows this structure:

```
src/<module>/
├── __init__.py               # Public exports
├── <component>_service.py    # Main logic (single responsibility)
└── config.py                 # Module-specific configuration (if needed)
```

Plus:
- `tests/unit/test_<module>.py` — Fast unit tests with pytest
- `tests/integration/test_<module>_integration.py` — Integration tests with real services

---

# 🗺️ Roadmap (Milestones)

| Milestone | Status | Key Deliverable | Verification |
|-----------|--------|-----------------|--------------|
| **M1: Foundation** | ✅ Done | Structure + Config with Pydantic | `python test_setup.py` |
| **M2: Embeddings** | ✅ Done | `src/embeddings/embedding_service.py` | `pytest tests/ -k embeddings` |
| **M3: Vector DB** | ✅ Done | `src/database/qdrant_database.py` | `pytest tests/ -k vector` |
| **M4: Parsers** | 🚧 Next | `src/parsers/` (PDF/HTML/Markdown) | `pytest tests/ -k parsers` |
| **M5: Ingestion** | ⏸️ | `src/ingestion/document_processor.py` | `pytest tests/ -k ingestion` |
| **M6: Flexible LLM** | ⏸️ | `src/llm/providers/` + Factory | `pytest tests/ -k llm` |
| **M7: Complete RAG** | ⏸️ | `src/rag/rag_pipeline.py` + API/CLI | `pytest tests/ -k rag` |

**Rule:** Don't skip milestones. Each one must be verified before continuing.

---

# ✅ How to Run and Test

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate              # Windows
source venv/bin/activate           # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env             # Windows
cp .env.example .env               # Linux/Mac

# Verify installation
python test_setup.py
```

### Environment Variables

```env
# General
PROJECT_NAME=docvault
ENVIRONMENT=development
LOG_LEVEL=INFO

# Paths
DATA_DIR=data
DOCUMENTS_DIR=data/documents

# [M6] LLM Configuration (future — don't configure until M6)
# LLM_PROVIDER=ollama_local
# LLM_MODEL=llama3.2:3b
# LLM_SERVER_URL=http://localhost:11434
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

---

# 📝 Guide for Making Changes

## Adding a New Module (General)

1. Create module folder in `src/<name>/` with standard files.
2. Add unit tests in `tests/unit/test_<name>.py`.
3. Add integration tests in `tests/integration/test_<name>_integration.py`.
4. Update `requirements.txt` if new dependencies: `pip freeze > requirements.txt`.
5. Update this AGENTS.md roadmap status.

## Modifying Configuration

1. Edit `config/settings.py` — add field with type, default, and description.
2. Document in `.env.example`.
3. Update `display_config()` if relevant.

---

# 🚫 Anti-patterns

| ❌ Don't do | ✅ Do instead |
|-------------|---------------|
| Install deps without updating `requirements.txt` | `pip freeze > requirements.txt` after installing |
| Hardcode absolute paths | Use `settings.get_full_path()` or `Path` |
| Use `print()` for logs | Use Python `logging` module |
| Import `openai`/`anthropic` directly | Use `LLMFactory` and the provider abstraction |
| Create files outside planned structure | Follow the milestone structure |
| Skip milestones | Complete and verify each in order |
| Write code/comments in Spanish | Everything in English |
| Change embedding model without validation | Validate impact on existing indexed data first |
| Change chunk size without measuring | Measure retrieval quality before and after |
| Mix milestone deliverables | One milestone at a time |

---

# 🔴 Recurring Agent Errors — READ BEFORE CODING

> These are errors the agent has committed repeatedly.
> Check this list before writing code. If you're about to do something
> listed here, STOP and follow the correction.

## 🚨 CRITICAL

### 1. Generating code in Spanish because the conversation is in Spanish

- **Error:** Creating variables like `servicio_embeddings`, comments like `# Generar embedding`, or docstrings like `"""Retorna los resultados."""`
- **Why it happens:** The agent mirrors the conversation language into the code.
- **Consequence:** Inconsistent codebase mixing languages. Impossible to maintain.
- **Correction:** ALL code is in English. The conversation language is IRRELEVANT to the code language. Always actively translate concepts.
  ```python
  # ❌ WRONG — mirrors Spanish conversation
  class ServicioEmbeddings:
      """Genera embeddings para documentos."""
      def generar_embedding(self, texto: str) -> list[float]:
          # Procesar el texto
          ...

  # ✅ CORRECT — always English
  class EmbeddingService:
      """Generate embeddings for documents."""
      def generate_embedding(self, text: str) -> list[float]:
          # Process the text
          ...
  ```
- **Verification:** Search for common Spanish words in code: `obtener`, `crear`, `buscar`, `resultado`, `usuario`, `archivo`, `documento`, `consulta`. If any appear, it's a bug.

### 2. Implementing code from future milestones

- **Error:** While working on M2 (Embeddings), creating LLM provider files, RAG pipeline stubs, or API endpoints that belong to M5, M6, M7.
- **Why it happens:** The agent reads the full roadmap and tries to "prepare" things in advance.
- **Consequence:** Premature abstractions, untested code, and coupling to decisions not yet made.
- **Correction:** Only create files and code for the CURRENT milestone. Future milestones don't exist yet.
- **Mental test:** Before creating a file, ask: "Does this file belong to the current milestone?" If no, don't create it.

### 3. Hardcoding paths with string concatenation

- **Error:** Writing `"data/documents/" + filename` or `os.path.join("data", "documents")`.
- **Why it happens:** The agent defaults to common Python patterns instead of project conventions.
- **Consequence:** Breaks on Windows. Ignores project configuration system.
- **Correction:** Always use `pathlib.Path` and `settings`:
  ```python
  # ❌ WRONG
  path = "data/documents/" + filename
  path = os.path.join("data", "documents", filename)

  # ✅ CORRECT
  path = settings.documents_dir / filename
  # or
  path = Path(settings.documents_dir) / filename
  ```

## ⚠️ HIGH

### 4. Using `.dict()` instead of `.model_dump()` (Pydantic v2)

- **Error:** Calling `.dict()` or `.json()` on Pydantic models.
- **Why it happens:** Agent trained on Pydantic v1 patterns.
- **Consequence:** DeprecationWarning now, error in future versions.
- **Correction:** Always use `model_dump()` / `model_dump_json()`.

### 5. Adding dependencies without updating requirements.txt

- **Error:** Running `pip install some-package` without `pip freeze > requirements.txt`.
- **Why it happens:** The agent focuses on making code work, forgets the dependency file.
- **Consequence:** Environment can't be reproduced.
- **Correction:** Every `pip install` MUST be followed by `pip freeze > requirements.txt`.

### 6. Creating "helper" files nobody asked for

- **Error:** Generating extra files like `utils.py`, `helpers.py`, `constants.py`, or READMEs inside modules.
- **Why it happens:** Agent tries to be "thorough" and anticipate needs.
- **Consequence:** Dead code, unnecessary complexity.
- **Correction:** Only create files that are part of the current task. If you think a utility is needed, ask first.

### 7. Changing the embedding model or chunk parameters without asking

- **Error:** Suggesting or implementing a "better" embedding model or different chunk sizes.
- **Why it happens:** The agent has opinions about optimal models and tries to optimize.
- **Consequence:** Incompatible with existing indexed data. Requires full reindex. Breaks retrieval.
- **Correction:** The embedding model and chunk parameters are FIXED decisions. Don't change them without explicit approval and a reindexing plan.

## 💡 MEDIUM

### 8. Tests that only check "it doesn't crash"

- **Error:** Tests that just call a function and assert the result is not None.
- **Why it happens:** Agent generates tests to satisfy coverage, not to verify behavior.
- **Correction:**
  ```python
  # ❌ WRONG
  def test_generate_embedding():
      service = EmbeddingService()
      result = service.generate_embedding("hello")
      assert result is not None

  # ✅ CORRECT
  def test_generate_embedding_returns_correct_dimensions():
      service = EmbeddingService()
      result = service.generate_embedding("hello")
      assert isinstance(result, list)
      assert len(result) == 384  # MiniLM-L12 dimension
      assert all(isinstance(x, float) for x in result)

  def test_similar_texts_have_high_cosine_similarity():
      service = EmbeddingService()
      emb1 = service.generate_embedding("the cat sits on the mat")
      emb2 = service.generate_embedding("a cat is sitting on a rug")
      similarity = cosine_similarity(emb1, emb2)
      assert similarity > 0.7
  ```

### 9. Using `print()` instead of `logging`

- **Error:** Scattering `print()` for debugging or status messages in `src/`.
- **Why it happens:** Quicker and simpler.
- **Correction:** Use `logging` module in `src/`. `print()` is only acceptable in `scripts/` for interactive output.

---

## 📋 How to Maintain This Section

When the agent makes a new recurring error (2+ times):

1. Document it with: **Error** · **Why it happens** · **Consequence** · **Correction** (with code example).
2. Assign severity: 🚨 Critical / ⚠️ High / 💡 Medium.
3. Place critical errors at the top.
4. Periodically remove errors that no longer occur.

> **Golden rule:** If you correct the agent for the same thing more than 2 times,
> don't correct it again in chat — document it here.

---

# ⚠️ Important Context for Agents

- **Git repository initialized.** Repo: https://github.com/Membrive92/DocVault.git
- **Windows environment.** Always use `pathlib.Path`. Never assume Unix path separators.
- **Python 3.10+ required** by Pydantic 2.x and modern type hints.
- **Testing:** `pytest` for all tests. `tests/unit/` for fast tests, `tests/integration/` for real-service tests.
- **Qdrant storage is ephemeral.** `data/qdrant_storage/` is gitignored and can be regenerated by re-ingesting.
- **Embedding model is multilingual** (`paraphrase-multilingual-MiniLM-L12-v2`). Supports Spanish + English documents. Don't change without reindexing all data.

---

# 🔗 Resources

- **Sentence Transformers:** https://www.sbert.net/
- **Qdrant Docs:** https://qdrant.tech/documentation/
- **Pydantic Settings:** https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- **Ollama:** https://ollama.ai/docs

---

**Last update:** 2026-02-12
**Status:** Milestone 3 completed — Ready for M4 (Document Parsers)
