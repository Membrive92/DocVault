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

## 🗺️ Milestones

| Milestone | Status | Focus |
|-----------|--------|-------|
| **M1: Foundation** | ✅ Done | Project structure + Pydantic config |
| **M2: Embeddings** | ✅ Done | Local sentence-transformers integration |
| **M3: Vector DB** | ✅ Done | Qdrant vector database integration |
| **M4: Parsers** | ✅ Done | PDF, HTML, Markdown document parsers |
| **M5: Ingestion** | ✅ Done | Document chunking and indexing pipeline |
| **M6: Flexible LLM** | ✅ Done | Multi-provider LLM abstraction layer |
| **M7: Complete RAG** | ✅ Done | End-to-end RAG pipeline + API + CLI |
| **M8: Web Frontend** | 🚧 In Progress | React + Vite UI for non-technical users |

**203 tests passed** (unit + integration). See [docs/](docs/) for detailed milestone documentation.

### Milestone 8: Web Frontend (In Progress)

Web UI for non-technical documentation users built with **React + Vite + TypeScript + Tailwind CSS**.

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1: Backend API** | ✅ Done | CORS + 6 new endpoints + 15 tests |
| **Phase 2: Frontend Foundation** | 🚧 Next | Vite project + Tailwind + Router + Layout |
| **Phase 3: Functional Pages** | ⏸️ Pending | QueryPage + DocumentsPage + AdminPage |
| **Phase 4: Polish & Documentation** | ⏸️ Pending | UX refinement + responsive design |

See [docs/milestone-08-frontend.md](docs/milestone-08-frontend.md) for detailed implementation plan.

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

Edit `.env` to configure your LLM provider (see [Environment Variables](#environment-variables)).

5. **Verify installation**

```bash
python test_setup.py
```

## 📖 Usage

### 1. Ingest Documents

Place your documents (PDF, HTML, Markdown) in `data/documents/`, then run the ingestion pipeline:

```python
from pathlib import Path
from src.ingestion import IngestionPipeline

pipeline = IngestionPipeline()
summary = pipeline.ingest_directory(Path("data/documents/"))
print(f"Ingested {summary.total_chunks} chunks from {summary.processed} files")
```

### 2. Query via Python API

```python
from src.rag import RAGPipeline

pipeline = RAGPipeline()

response = pipeline.query("How do I configure logging?")
print(response.answer)
for source in response.sources:
    print(f"  {source.source_file} (score: {source.similarity_score:.2f})")
```

### 3. Query via REST API

```bash
# Start the server
python -m src.api.server

# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I install Docker?", "top_k": 5}'

# Streaming response
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Kubernetes?"}' \
  --no-buffer
```

### 4. Document Management via REST API

```bash
# Upload a document
curl -X POST http://localhost:8000/documents/upload -F "file=@manual.pdf"

# List all documents
curl http://localhost:8000/documents

# Delete a document
curl -X DELETE http://localhost:8000/documents/manual.pdf

# Trigger ingestion (all documents)
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{}'

# Force re-index
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"force_reindex": true}'

# Check ingestion status
curl http://localhost:8000/ingest/status

# View public configuration
curl http://localhost:8000/config
```

### 5. Query via Interactive CLI

```bash
python -m src.cli.interactive

# Inside the REPL:
> How do I configure logging?
> /sources    # Show indexed collection info
> /help       # Show available commands
> /exit       # Exit the CLI
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
│   ├── api/                  # [M7+M8] FastAPI endpoints (10 endpoints)
│   └── cli/                  # [M7] Interactive CLI
├── frontend/                 # [M8] React + Vite + TypeScript (in progress)
├── tests/                    # All tests with pytest
│   ├── unit/                 # Fast unit tests (no ML model loading)
│   └── integration/          # Slow integration tests (real models + services)
├── data/
│   ├── documents/            # Documents to ingest (PDFs, HTML, MD)
│   └── qdrant_storage/       # Vector DB persistence (gitignored)
├── docs/                     # Technical documentation
│   ├── architecture.md       # System architecture overview
│   ├── milestone-*.md        # Per-milestone implementation docs (M1-M8)
│   └── internal_guide/       # Internal guides (Spanish, Java comparisons)
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

# LLM Configuration
LLM_PROVIDER=ollama_local
# LLM_MODEL=llama3.2:3b
# LLM_SERVER_URL=http://localhost:11434
# LLM_TEMPERATURE=0.7
# LLM_MAX_TOKENS=1024
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# RAG Pipeline
# RAG_TOP_K=5
# RAG_MIN_SIMILARITY=0.3

# API Server
# API_HOST=0.0.0.0
# API_PORT=8000
```

## 🛠️ Technology Stack

### Backend (M1-M7 + M8 Phase 1)
- **Python 3.10+** — Modern type hints and async support
- **Pydantic 2.x + pydantic-settings** — Type-safe configuration from .env
- **pathlib** — Cross-platform path handling
- **sentence-transformers** — Local multilingual embeddings (M2)
- **qdrant-client** — Vector database for similarity search (M3)
- **pypdf** — PDF text and metadata extraction (M4)
- **BeautifulSoup4 + lxml** — HTML content extraction with boilerplate removal (M4)
- **python-frontmatter** — Markdown YAML frontmatter parsing (M4)
- **uuid5** — Deterministic chunk IDs for re-indexing (M5)
- **ollama** — Local LLM inference via Ollama SDK (M6)
- **openai** — OpenAI GPT models via official SDK (M6)
- **anthropic** — Anthropic Claude models via official SDK (M6)
- **FastAPI + uvicorn** — REST API with 10 endpoints and streaming support (M7+M8)
- **python-multipart** — Multipart form data for file uploads (M8)
- **rich** — Terminal formatting for interactive CLI (M7)

### Frontend (M8 — in progress)
- **React 18 + TypeScript** — UI framework with type safety
- **Vite** — Fast build tool with HMR and dev proxy
- **Tailwind CSS** — Utility-first styling
- **React Router v6** — Client-side page navigation
- **react-markdown + remark-gfm** — Render LLM responses as markdown
- **react-dropzone** — Drag & drop file upload
- **lucide-react** — Icon library

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
# Run all tests (unit + integration)
pytest

# Run only fast unit tests
pytest tests/unit/

# Run only integration tests (slower, loads ML models)
pytest tests/integration/

# Run tests for a specific module
pytest tests/ -k embeddings
pytest tests/ -k vector
pytest tests/ -k parsers
pytest tests/ -k ingestion
pytest tests/ -k llm
pytest tests/ -k rag
pytest tests/ -k api

# Run with coverage
pytest --cov=src
```

## 📚 Documentation

- **[README.md](README.md)** — This file, project overview and quick start
- **[AGENTS.md](AGENTS.md)** — Comprehensive guide for AI agents and developers
- **[docs/architecture.md](docs/architecture.md)** — System architecture, data flow, and design decisions
- **[docs/](docs/)** — Per-milestone implementation documentation (M1-M8)
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

**Status:** M1-M7 completed. M8 (Web Frontend) in progress — Phase 1 (Backend API) done.

**Last Updated:** 2026-02-12
