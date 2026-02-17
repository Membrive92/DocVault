"""
FastAPI server for the DocVault RAG pipeline.

Provides REST endpoints for querying documents, streaming responses,
checking health, viewing indexed sources, managing documents, and
triggering ingestion.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import settings
from src.ingestion import IngestionPipeline
from src.ingestion.config import SUPPORTED_EXTENSIONS
from src.rag import RAGConfig, RAGPipeline

logger = logging.getLogger(__name__)


# ==========================================
# Request/Response Models
# ==========================================

class QueryRequest(BaseModel):
    """Request body for query endpoints."""

    query: str = Field(..., min_length=1, description="The question to ask")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of chunks to retrieve")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096, description="Max tokens in response")


class IngestRequest(BaseModel):
    """Request body for ingestion endpoint."""

    file_path: Optional[str] = Field(default=None, description="Specific file to ingest (None = all)")
    force_reindex: bool = Field(default=False, description="Force re-indexing of already indexed files")


class SourceResponse(BaseModel):
    """A single source in the query response."""

    source_file: str
    chunk_index: int
    similarity_score: float
    chunk_text: str
    document_title: Optional[str] = None
    document_format: Optional[str] = None


class QueryResponse(BaseModel):
    """Response body for query endpoint."""

    answer: str
    sources: list[SourceResponse]
    query: str
    model_used: str
    retrieval_count: int
    retrieval_time_ms: Optional[float] = None
    generation_time_ms: Optional[float] = None


class DocumentInfo(BaseModel):
    """Metadata for a document file."""

    filename: str
    format: str
    size_bytes: int
    modified_at: str


class UploadResponse(BaseModel):
    """Response body for document upload."""

    filename: str
    size_bytes: int
    message: str


class IngestResponse(BaseModel):
    """Response body for ingestion endpoint."""

    processed: int
    skipped: int
    failed: int
    total_chunks: int
    results: list[dict]


class ConfigResponse(BaseModel):
    """Public configuration (no API keys)."""

    llm_provider: str
    llm_model: Optional[str]
    rag_top_k: int
    rag_min_similarity: float
    api_host: str
    api_port: int


# ==========================================
# Application State
# ==========================================

rag_pipeline: Optional[RAGPipeline] = None
ingestion_pipeline: Optional[IngestionPipeline] = None
last_ingestion_summary: Optional[dict] = None


# ==========================================
# Lifespan
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG and ingestion pipelines on startup, cleanup on shutdown."""
    global rag_pipeline, ingestion_pipeline

    logger.info("Initializing RAG pipeline...")

    try:
        rag_pipeline = RAGPipeline()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize RAG pipeline: %s", e)
        rag_pipeline = None

    try:
        ingestion_pipeline = IngestionPipeline()
        logger.info("Ingestion pipeline initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize ingestion pipeline: %s", e)
        ingestion_pipeline = None

    yield

    logger.info("Shutting down pipelines")
    rag_pipeline = None
    ingestion_pipeline = None


# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(
    title="DocVault API",
    description="RAG pipeline REST API for document question-answering",
    version="1.0.0",
    lifespan=lifespan,
)


# ==========================================
# CORS Middleware
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Endpoints
# ==========================================

@app.get("/health")
def health_check():
    """Check API and pipeline health status."""
    return {
        "status": "healthy",
        "pipeline_initialized": rag_pipeline is not None,
    }


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Query the RAG pipeline and get an answer with sources.

    Embeds the query, searches the vector database for relevant chunks,
    and generates an answer using the configured LLM.
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Check server logs.",
        )

    try:
        response = rag_pipeline.query(
            query_text=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return QueryResponse(
            answer=response.answer,
            sources=[
                SourceResponse(
                    source_file=s.source_file,
                    chunk_index=s.chunk_index,
                    similarity_score=s.similarity_score,
                    chunk_text=s.chunk_text,
                    document_title=s.document_title,
                    document_format=s.document_format,
                )
                for s in response.sources
            ],
            query=response.query,
            model_used=response.model_used,
            retrieval_count=response.retrieval_count,
            retrieval_time_ms=response.retrieval_time_ms,
            generation_time_ms=response.generation_time_ms,
        )

    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    """
    Query the RAG pipeline with streaming response.

    Returns chunks of the LLM response as they are generated,
    using text/plain streaming.
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Check server logs.",
        )

    try:
        stream = rag_pipeline.query(
            query_text=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            streaming=True,
        )

        return StreamingResponse(stream, media_type="text/plain")

    except Exception as e:
        logger.error("Stream query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/sources")
def get_sources():
    """Get information about indexed document sources."""
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Check server logs.",
        )

    try:
        return {"sources": rag_pipeline.get_indexed_sources()}
    except Exception as e:
        logger.error("Failed to get sources: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==========================================
# Document Management Endpoints
# ==========================================


def _get_documents_dir() -> Path:
    """Resolve and ensure the documents directory exists."""
    docs_dir = settings.get_full_path(settings.documents_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile):
    """
    Upload a document file to the documents directory.

    Validates that the file has a supported extension (.pdf, .html, .htm, .md, .markdown).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    extension = Path(file.filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{extension}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    docs_dir = _get_documents_dir()
    dest = docs_dir / file.filename

    try:
        content = await file.read()
        dest.write_bytes(content)
    except Exception as e:
        logger.error("Failed to save uploaded file: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}") from e

    logger.info("Uploaded file: %s (%d bytes)", file.filename, dest.stat().st_size)

    return UploadResponse(
        filename=file.filename,
        size_bytes=dest.stat().st_size,
        message=f"File '{file.filename}' uploaded successfully",
    )


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    """List all documents in the documents directory with metadata."""
    docs_dir = _get_documents_dir()
    documents = []

    for file_path in sorted(docs_dir.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        stat = file_path.stat()
        modified_at = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat()

        documents.append(
            DocumentInfo(
                filename=file_path.name,
                format=file_path.suffix.lstrip("."),
                size_bytes=stat.st_size,
                modified_at=modified_at,
            )
        )

    return documents


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Delete a document file from the documents directory."""
    docs_dir = _get_documents_dir()
    file_path = docs_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"'{filename}' is not a file")

    file_path.unlink()
    logger.info("Deleted document: %s", filename)

    return {"message": f"File '{filename}' deleted successfully"}


@app.post("/ingest", response_model=IngestResponse)
def trigger_ingest(request: IngestRequest):
    """
    Trigger document ingestion.

    If file_path is provided, ingest that specific file.
    Otherwise, ingest all documents in the documents directory.
    """
    global last_ingestion_summary

    if ingestion_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Ingestion pipeline not initialized. Check server logs.",
        )

    docs_dir = _get_documents_dir()

    try:
        if request.file_path:
            file_path = docs_dir / request.file_path
            if not file_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"File '{request.file_path}' not found",
                )
            result = ingestion_pipeline.ingest_file(file_path)
            response = IngestResponse(
                processed=1 if result.status == "success" else 0,
                skipped=0,
                failed=1 if result.status == "failed" else 0,
                total_chunks=result.chunks_created,
                results=[{
                    "file_path": result.file_path,
                    "chunks_created": result.chunks_created,
                    "status": result.status,
                    "error": result.error,
                }],
            )
        else:
            summary = ingestion_pipeline.ingest_directory(
                docs_dir,
                force_reindex=request.force_reindex,
            )
            response = IngestResponse(
                processed=summary.processed,
                skipped=summary.skipped,
                failed=summary.failed,
                total_chunks=summary.total_chunks,
                results=[
                    {
                        "file_path": r.file_path,
                        "chunks_created": r.chunks_created,
                        "status": r.status,
                        "error": r.error,
                    }
                    for r in summary.results
                ],
            )

        last_ingestion_summary = response.model_dump()
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/ingest/status")
def get_ingest_status():
    """Get the summary of the last ingestion run."""
    if last_ingestion_summary is None:
        return {"message": "No ingestion has been run yet"}
    return last_ingestion_summary


@app.get("/config", response_model=ConfigResponse)
def get_config():
    """Get public configuration (no API keys exposed)."""
    return ConfigResponse(
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        rag_top_k=settings.rag_top_k,
        rag_min_similarity=settings.rag_min_similarity,
        api_host=settings.api_host,
        api_port=settings.api_port,
    )


# ==========================================
# Standalone runner
# ==========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
