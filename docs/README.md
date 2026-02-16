# DocVault Documentation

This directory contains comprehensive technical documentation for the DocVault project, organized by architectural layers and development milestones.

## 📚 Documentation Index

### Architecture
- **[architecture.md](architecture.md)** - Complete system architecture overview
  - Layer-by-layer breakdown
  - Data flow diagrams
  - Design decisions and trade-offs
  - Technology stack rationale
  - Scalability considerations

### Milestones

Development milestones are implemented sequentially, each building on the previous:

#### ✅ Completed

1. **[milestone-01-foundation.md](milestone-01-foundation.md)** - Project Foundation
   - Project structure with modular organization
   - Pydantic-based configuration system
   - Environment variables management
   - Installation verification

2. **[milestone-02-embeddings.md](milestone-02-embeddings.md)** - Local Embeddings
   - Sentence-transformers integration
   - 384-dimensional multilingual embeddings
   - Cosine similarity for semantic search
   - L2 normalization optimization
   - Comprehensive testing strategy

3. **[milestone-03-vector-db.md](milestone-03-vector-db.md)** - Vector Database (Qdrant)
   - Qdrant integration for vector storage
   - HNSW index configuration
   - Abstract database interface
   - In-memory and persistent modes
   - Similarity search implementation

4. **[milestone-04-parsers.md](milestone-04-parsers.md)** - Document Parsers
   - PDF parser (pypdf)
   - HTML parser (BeautifulSoup)
   - Markdown parser (frontmatter)
   - Parser factory pattern
   - Metadata extraction

5. **[milestone-05-ingestion.md](milestone-05-ingestion.md)** - Document Ingestion Pipeline
   - Text chunking with paragraph-first strategy and overlap
   - Incremental indexing with state tracking (JSON + mtime)
   - End-to-end pipeline orchestrating M2+M3+M4
   - Deterministic UUID5 chunk IDs

6. **[milestone-06-llm.md](milestone-06-llm.md)** - Flexible LLM Layer
   - LLMProvider abstract interface (Strategy Pattern)
   - OllamaProvider for local and remote Ollama servers
   - OpenAIProvider for GPT-4/GPT-3.5 models
   - AnthropicProvider for Claude models
   - LLMProviderFactory with config/settings.py integration
   - RAG prompt template and streaming generation

7. **[milestone-07-rag.md](milestone-07-rag.md)** - Complete RAG Pipeline
   - RAGPipeline integrating embeddings + vector DB + LLM
   - FastAPI REST API (health, query, query/stream, sources)
   - Interactive CLI with rich formatting
   - Streaming responses via StreamingResponse
   - Source citations with similarity scores
   - 15 unit + 9 API + 4 integration tests

## 🎯 Reading Recommendations

### For New Contributors
1. Start with [architecture.md](architecture.md) to understand the system design
2. Read [milestone-01-foundation.md](milestone-01-foundation.md) to understand project structure
3. Follow milestones in order (M1 → M2 → M3 → ...)

### For Understanding Specific Components
- **Embeddings**: [milestone-02-embeddings.md](milestone-02-embeddings.md)
- **Vector Search**: [milestone-03-vector-db.md](milestone-03-vector-db.md)
- **Document Processing**: [milestone-04-parsers.md](milestone-04-parsers.md) + [milestone-05-ingestion.md](milestone-05-ingestion.md)
- **LLM Integration**: [milestone-06-llm.md](milestone-06-llm.md)
- **Complete System**: [milestone-07-rag.md](milestone-07-rag.md)

### For Implementation Details
Each milestone document includes:
- **Overview**: What the milestone achieves
- **Architecture**: How components fit together
- **Implementation Plan**: Step-by-step tasks with code examples
- **Testing Strategy**: Unit tests and verification
- **Integration**: How it connects to other milestones

## 📊 Document Structure

Each milestone document follows this structure:

```
# Milestone X: Title

**Status:** ✅ Done | 🚧 Next | ⏸️ Pending
**Dependencies:** Previous milestones required
**Goal:** One-sentence objective

## Overview
High-level description of what this milestone achieves

## Architecture
Diagrams and component relationships

## Implementation Plan
Task-by-task breakdown with code examples

## Testing Strategy
Unit tests and verification scripts

## Integration
How this milestone connects to others

## Verification Criteria
Checklist for completion

## Next Steps
Preview of the next milestone
```

## 🔧 Technical Depth

Documentation is written for:
- **AI Agents**: Clear, structured, implementable instructions
- **Senior Developers**: Architecture rationale and trade-offs
- **Contributors**: Step-by-step implementation guides

## 📝 Maintaining Documentation

When updating documentation:
1. Keep milestone docs synchronized with actual implementation
2. Update architecture.md when system design changes
3. Add code examples for clarity
4. Include diagrams for complex flows
5. Document trade-offs and design decisions

## 🔗 Related Documentation

- **[../README.md](../README.md)** - Project overview and quick start
- **[../AGENTS.md](../AGENTS.md)** - Guide for AI agents and developers
- **[../.env.example](../.env.example)** - Environment configuration reference

---

**Last Updated:** 2026-02-12
**All Milestones Completed** (M1-M7) — Project feature-complete
