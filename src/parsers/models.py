"""
Data models for parsed documents.

ParsedDocument is the standard format returned by all parsers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedDocument:
    """
    Represents a parsed document with extracted text and metadata.

    This is the standard output format for all document parsers.
    Every parser returns a ParsedDocument regardless of input format.

    Attributes:
        text: Extracted clean text content.
        source_path: Original file path.
        format: Document format ("pdf", "html", or "markdown").
        extracted_at: ISO timestamp of extraction.
        parser_version: Version of the parser used.
        title: Document title if available.
        author: Document author if available.
        created_date: Document creation date if available.
        language: Document language if detected.
        page_count: Number of pages (PDF only).
    """

    # Required fields
    text: str
    source_path: str
    format: str
    extracted_at: str
    parser_version: str

    # Optional metadata
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[str] = None
    language: Optional[str] = None
    page_count: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate document after creation."""
        if not self.text or not self.text.strip():
            raise ValueError("Parsed document cannot have empty text")

        if self.format not in ("pdf", "html", "markdown"):
            raise ValueError(
                f"Invalid format: {self.format}. "
                f"Must be one of: pdf, html, markdown"
            )

    @property
    def word_count(self) -> int:
        """Calculate approximate word count."""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """Get character count."""
        return len(self.text)
