# DocVault

> **Local-first RAG system for enterprise documentation with flexible LLM provider switching**

DocVault is a Retrieval-Augmented Generation (RAG) system designed to query documentation across multiple projects using AI. Built with a local-first approach for privacy and cost-effectiveness, it can seamlessly scale to commercial models or your own server infrastructure without code changes.

## ✨ Key Features

- 🏠 **100% Local Operation** — Run entirely on your machine with no external costs or privacy concerns
- 🔄 **Flexible LLM Switching** — Switch between Ollama (local), OpenAI, Anthropic, or your own server with a single config change
- 📚 **Multi-Project Support** — Query documentation across multiple projects with context preservation
- 📄 **Multi-Format Support** — Ingest PDFs, HTML, and Markdown documents
- 🎯 **Strategy Pattern Architecture** — Clean abstraction layer prevents vendor lock-in
- 🧩 **Incremental Development** — Built milestone by milestone with verification at each step

## 🏗️ Architecture Overview

DocVault uses a layered architecture with a flexible LLM abstraction at its core:

```
[Documents] → [Parsers] → [Chunking] → [Embeddings] → [Qdrant Vector DB]
                                                              ↓
[User Query] → [Embeddings] → [Vector Search] → [Context] → [LLM] → [Response]
```

### Flexible LLM Layer (Strategy Pattern)

The key architectural decision is the **provider-agnostic LLM layer** that allows switching between:

- **Ollama (Local)** — Free, private, runs on localhost
- **Ollama (Server)** — Own infrastructure, enterprise control
- **OpenAI** — GPT-4, GPT-3.5, etc.
- **Anthropic** — Claude models

Switch providers by changing a single environment variable — no code changes required.

## 📦 Current Status

**Milestone 1: Foundation** ✅ **COMPLETED**

- [x] Project structure with modular organization
- [x] Pydantic-based configuration system
- [x] Environment variables management
- [x] Installation verification script
- [x] Comprehensive documentation (README, AGENTS.md)

**Next:** Milestone 2 — Local Embeddings with sentence-transformers

## 🗺️ Roadmap

| Milestone | Status | Focus |
|-----------|--------|-------|
| **M1: Foundation** | ✅ Done | Project structure + Pydantic config |
| **M2: Embeddings** | 🚧 Next | Local sentence-transformers integration |
| **M3: Vector DB** | ⏸️ Pending | Qdrant setup and connection |
| **M4: Parsers** | ⏸️ Pending | PDF, HTML, Markdown document parsers |
| **M5: Ingestion** | ⏸️ Pending | Document chunking and indexing pipeline |
| **M6: Flexible LLM** | ⏸️ Pending | Multi-provider LLM abstraction layer |
| **M7: Complete RAG** | ⏸️ Pending | End-to-end RAG pipeline + API + CLI |

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip and virtualenv

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Membrive92/DocVault.git
cd DocVault
```

2. **Create and activate virtual environment**

```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment**

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Edit `.env` if needed (defaults work for M1).

5. **Verify installation**

```bash
python test_setup.py
```

Expected output:
```
🎉 Everything is configured correctly!
📝 Next step: Milestone 2 - Embeddings
```

## 📁 Project Structure

```
DocVault/
├── config/                   # Centralized configuration
│   ├── __init__.py
│   └── settings.py           # Pydantic Settings (loads from .env)
├── src/                      # Source code (developing by milestones)
│   ├── __init__.py
│   ├── embeddings/           # [M2] Local embeddings generation
│   ├── database/             # [M3] Qdrant vector store client
│   ├── parsers/              # [M4] PDF/HTML/Markdown parsers
│   ├── ingestion/            # [M5] Document ingestion pipeline
│   ├── llm/                  # [M6] Flexible LLM layer (providers)
│   ├── rag/                  # [M7] Complete RAG pipeline
│   ├── api/                  # [M7] FastAPI endpoints
│   └── cli/                  # [M7] Interactive CLI
├── tests/                    # Automated tests with pytest
├── scripts/                  # Interactive verification scripts
├── data/
│   ├── documents/            # Documents to ingest (PDFs, HTML, MD)
│   └── qdrant_storage/       # Vector DB persistence (gitignored)
├── .env.example              # Environment variables template
├── .gitignore
├── requirements.txt          # Python dependencies
├── test_setup.py             # Installation verification
├── README.md                 # This file
└── AGENTS.md                 # Detailed guide for AI agents/developers
```

## 🔧 Configuration

Configuration uses **Pydantic Settings** with three-tier priority:

1. System environment variables (highest priority)
2. `.env` file
3. Default values in `config/settings.py` (lowest priority)

### Configuration Example

```python
from config.settings import settings

# Access configuration
print(settings.project_name)     # "docvault"
print(settings.environment)      # "development"
print(settings.log_level)        # "INFO"

# Create necessary directories
settings.ensure_directories()

# Display current configuration
settings.display_config()
```

### Environment Variables

Edit `.env` to customize:

```env
# General
PROJECT_NAME=docvault
ENVIRONMENT=development
LOG_LEVEL=INFO

# Paths (relative to project root)
DATA_DIR=data
DOCUMENTS_DIR=data/documents

# [M6] LLM Configuration (future milestones)
# LLM_PROVIDER=ollama_local
# LLM_MODEL=llama3.2:3b
# LLM_SERVER_URL=http://localhost:11434
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

## 🛠️ Technology Stack

### Current (M1)
- **Python 3.10+** — Modern type hints and async support
- **Pydantic 2.x** — Type-safe configuration management
- **pathlib** — Cross-platform path handling

### Planned (M2-M7)
- **sentence-transformers** — Local multilingual embeddings (M2)
- **Qdrant** — Vector database for similarity search (M3)
- **pypdf / beautifulsoup4** — Document parsing (M4)
- **Ollama / OpenAI / Anthropic** — LLM providers (M6)
- **FastAPI** — REST API endpoints (M7)

**Note:** We are NOT using LangChain. The project implements custom components for learning and full control.

## 👥 Development

### For AI Agents

This project is designed to be AI-agent-friendly. **Read [`AGENTS.md`](AGENTS.md)** before making any changes. It contains:

- Project context rules and conventions
- Code style requirements (all code in English)
- Architecture patterns and anti-patterns
- Milestone-by-milestone development guide
- Recurring errors to avoid

### For Human Developers

1. Follow the milestone order strictly (don't skip ahead)
2. All code, comments, and docstrings must be in English
3. Use type hints on all public functions
4. Write meaningful tests with pytest
5. Update `requirements.txt` after installing dependencies: `pip freeze > requirements.txt`

### Running Tests

```bash
# Install pytest (will be added in M2+)
pip install pytest

# Run all tests
pytest

# Run with coverage
pytest --cov=src
```

## 📚 Documentation

- **[README.md](README.md)** — This file, project overview and quick start
- **[AGENTS.md](AGENTS.md)** — Comprehensive guide for AI agents and developers
- **[.env.example](.env.example)** — Environment variables template with documentation

## 🤝 Contributing

This is currently an internal/educational project. Contributions follow these principles:

1. **Incremental development** — Complete one milestone before starting the next
2. **Verification required** — Each milestone must pass its verification script
3. **English only** — All code, comments, and documentation in English
4. **Type safety** — Use type hints and Pydantic for validation
5. **No vendor lock-in** — Maintain provider abstraction layers

## 📄 License

Internal project - Enterprise use

## 🔗 Resources

- **Sentence Transformers:** https://www.sbert.net/
- **Qdrant Documentation:** https://qdrant.tech/documentation/
- **Pydantic Settings:** https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- **Ollama:** https://ollama.ai/docs

---

**Status:** Milestone 1 completed ✅ — Ready for Milestone 2 (Embeddings)

**Last Updated:** 2025-02-11
