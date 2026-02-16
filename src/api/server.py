"""
FastAPI server for the DocVault RAG pipeline.

Provides REST endpoints for querying documents, streaming responses,
checking health, and viewing indexed sources.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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


# ==========================================
# Application State
# ==========================================

rag_pipeline: Optional[RAGPipeline] = None


# ==========================================
# Lifespan
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG pipeline on startup, cleanup on shutdown."""
    global rag_pipeline

    logger.info("Initializing RAG pipeline...")

    try:
        rag_pipeline = RAGPipeline()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize RAG pipeline: %s", e)
        rag_pipeline = None

    yield

    logger.info("Shutting down RAG pipeline")
    rag_pipeline = None


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
# Standalone runner
# ==========================================

if __name__ == "__main__":
    import uvicorn

    from config.settings import settings

    uvicorn.run(
        "src.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
