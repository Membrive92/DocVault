"""
Text chunking utilities for document ingestion.

Splits long documents into overlapping chunks suitable for embedding
and vector storage. Uses paragraph-first splitting to preserve natural
document structure.
"""

from __future__ import annotations

import logging
import re

from .config import CHARS_PER_TOKEN, CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_SIZE

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits text into overlapping chunks for embedding generation.

    Uses a paragraph-first strategy: splits on double newlines first,
    then falls back to sentence splitting for oversized paragraphs.
    Applies overlap from the end of the previous chunk to maintain
    context across boundaries.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE,
    ) -> None:
        """
        Initialize the text chunker.

        Args:
            chunk_size: Target size per chunk in tokens.
            chunk_overlap: Overlap between chunks in tokens.
            min_chunk_size: Minimum chunk size to keep in tokens.

        Raises:
            ValueError: If parameters are invalid.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size cannot be negative")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        self._chunk_chars = chunk_size * CHARS_PER_TOKEN
        self._overlap_chars = chunk_overlap * CHARS_PER_TOKEN
        self._min_chars = min_chunk_size * CHARS_PER_TOKEN

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.

        Strategy:
        1. Split by paragraphs (double newlines).
        2. If a paragraph exceeds 2x chunk_chars, split by sentences.
        3. Accumulate paragraphs into chunks up to chunk_chars.
        4. Apply overlap by carrying the last paragraph of the previous chunk.
        5. Filter out chunks below min_chunk_size.

        Args:
            text: Input text to chunk.

        Returns:
            List of text chunks. Empty list if text is empty.
        """
        if not text or not text.strip():
            return []

        paragraphs = re.split(r"\n\s*\n", text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        chunks: list[str] = []
        current_parts: list[str] = []
        current_length = 0

        for paragraph in paragraphs:
            para_len = len(paragraph)

            # Oversized paragraph: flush accumulator and split by sentences
            if para_len > self._chunk_chars * 2:
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_length = 0

                sentence_chunks = self._split_by_sentences(paragraph)
                chunks.extend(sentence_chunks)
                continue

            # Would adding this paragraph exceed chunk size?
            if current_length + para_len > self._chunk_chars and current_parts:
                chunks.append("\n\n".join(current_parts))

                # Overlap: carry last paragraph if small enough
                last_part = current_parts[-1]
                if len(last_part) <= self._overlap_chars * 2:
                    current_parts = [last_part, paragraph]
                    current_length = len(last_part) + para_len
                else:
                    current_parts = [paragraph]
                    current_length = para_len
            else:
                current_parts.append(paragraph)
                current_length += para_len

        # Flush remaining
        if current_parts:
            chunks.append("\n\n".join(current_parts))

        # Filter chunks below minimum size
        chunks = [c for c in chunks if len(c) >= self._min_chars]

        logger.debug(
            "Created %d chunks from text of %d chars", len(chunks), len(text)
        )

        return chunks

    def _split_by_sentences(self, text: str) -> list[str]:
        """
        Split a large text block by sentences into chunks.

        Used as fallback when a single paragraph exceeds the chunk size.

        Args:
            text: Large text block to split.

        Returns:
            List of sentence-based chunks.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current_sentences: list[str] = []
        current_length = 0

        for sentence in sentences:
            sent_len = len(sentence)

            if current_length + sent_len > self._chunk_chars and current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_length = sent_len
            else:
                current_sentences.append(sentence)
                current_length += sent_len

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks
