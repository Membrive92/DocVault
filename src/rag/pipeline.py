"""
RAG (Retrieval-Augmented Generation) pipeline.

Integrates embedding service, vector database, and LLM provider
into a complete question-answering pipeline with source citations.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator, Optional

from src.database import QdrantDatabase
from src.embeddings import EmbeddingService
from src.llm import LLMProvider, LLMProviderFactory

from .config import CONTEXT_SEPARATOR, MAX_CONTEXT_CHUNKS, NO_RESULTS_MESSAGE
from .models import RAGConfig, RAGResponse, Source

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline: query -> embed -> search -> context -> LLM -> response.

    Uses dependency injection for all services, making it easy to test
    and swap implementations.

    Args:
        embedding_service: Service for generating query embeddings.
        vector_db: Vector database for similarity search.
        llm_provider: LLM provider instance for generation.
        config: Pipeline configuration (top_k, temperature, etc.).

    Example:
        >>> from src.rag import RAGPipeline
        >>> pipeline = RAGPipeline()
        >>> response = pipeline.query("How do I install Docker?")
        >>> print(response.answer)
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_db: Optional[QdrantDatabase] = None,
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[RAGConfig] = None,
    ) -> None:
        self.config = config or RAGConfig()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_db = vector_db or QdrantDatabase()
        self.llm = llm_provider or LLMProviderFactory.create_provider()

        logger.info(
            "RAGPipeline initialized with provider=%s, top_k=%s, min_similarity=%s",
            type(self.llm).__name__,
            self.config.top_k,
            self.config.min_similarity,
        )

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
    ) -> RAGResponse | Iterator[str]:
        """
        Execute a RAG query: embed -> search -> generate.

        Args:
            query_text: The user's question.
            top_k: Number of chunks to retrieve (overrides config).
            temperature: LLM temperature (overrides config).
            max_tokens: Max tokens for LLM (overrides config).
            streaming: If True, returns an Iterator[str] instead of RAGResponse.

        Returns:
            RAGResponse with answer and sources, or Iterator[str] if streaming.

        Raises:
            ValueError: If query_text is empty.
            RuntimeError: If any pipeline step fails.
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")

        effective_top_k = top_k if top_k is not None else self.config.top_k
        effective_temperature = temperature if temperature is not None else self.config.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        logger.info("RAG query: %s", query_text[:100])

        # Step 1: Generate query embedding
        retrieval_start = time.perf_counter()

        query_vector = self.embedding_service.generate_embedding(query_text)

        # Step 2: Search vector database
        results = self.vector_db.search_similar(
            query_vector=query_vector,
            limit=min(effective_top_k, MAX_CONTEXT_CHUNKS),
            score_threshold=self.config.min_similarity,
        )

        retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

        logger.info(
            "Retrieved %s chunks in %.1fms",
            len(results),
            retrieval_time_ms,
        )

        # Step 3: Handle no results
        if not results:
            model_info = self.llm.get_model_info()
            return RAGResponse(
                answer=NO_RESULTS_MESSAGE,
                sources=[],
                query=query_text,
                model_used=model_info.get("model", "unknown"),
                retrieval_count=0,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=None,
            )

        # Step 4: Build sources and context
        sources = self._build_sources(results)
        context = self._build_context(sources)

        # Step 5: Generate response
        if streaming:
            return self.llm.generate_stream(
                prompt=query_text,
                context=context,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
            )

        generation_start = time.perf_counter()

        answer = self.llm.generate(
            prompt=query_text,
            context=context,
            temperature=effective_temperature,
            max_tokens=effective_max_tokens,
        )

        generation_time_ms = (time.perf_counter() - generation_start) * 1000

        model_info = self.llm.get_model_info()

        logger.info("Generated response in %.1fms", generation_time_ms)

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=query_text,
            model_used=model_info.get("model", "unknown"),
            retrieval_count=len(sources),
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
        )

    def _build_sources(self, results: list[dict[str, Any]]) -> list[Source]:
        """Convert raw search results to Source objects."""
        sources = []
        for result in results:
            metadata = result.get("metadata", {})
            sources.append(
                Source(
                    chunk_text=metadata.get("chunk_text", ""),
                    source_file=metadata.get("source_file", "unknown"),
                    chunk_index=metadata.get("chunk_index", 0),
                    similarity_score=result.get("score", 0.0),
                    document_title=metadata.get("document_title"),
                    document_format=metadata.get("document_format"),
                )
            )
        return sources

    def _build_context(self, sources: list[Source]) -> str:
        """
        Format retrieved sources into a context string for the LLM.

        Each source is prefixed with a header showing its filename
        and similarity score.
        """
        chunks = []
        for i, source in enumerate(sources, 1):
            header = (
                f"[Source {i}] {source.source_file} "
                f"(similarity: {source.similarity_score:.2f})"
            )
            chunks.append(f"{header}\n{source.chunk_text}")

        return CONTEXT_SEPARATOR.join(chunks)

    def get_indexed_sources(self) -> dict[str, Any]:
        """Get information about the indexed document collection."""
        return self.vector_db.get_collection_info()
