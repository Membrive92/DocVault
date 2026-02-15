"""
PDF document parser using pypdf.

Extracts text and metadata from PDF files page by page.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from .base_parser import DocumentParser
from .config import MAX_TEXT_LENGTH, PDF_PASSWORD
from .models import ParsedDocument

logger = logging.getLogger(__name__)


class PDFParser(DocumentParser):
    """
    Parser for PDF documents.

    Uses pypdf to extract text and metadata from PDF files.
    Handles multi-page documents and encrypted PDFs.
    """

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if file is a PDF."""
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a PDF file and extract text + metadata.

        Args:
            file_path: Path to PDF file.

        Returns:
            ParsedDocument with extracted content.

        Raises:
            FileNotFoundError: If PDF doesn't exist.
            ValueError: If no text could be extracted.
            RuntimeError: If PDF is encrypted or parsing fails.
        """
        path = Path(file_path)
        self._validate_file(path)

        logger.info("Parsing PDF: %s", path.name)

        try:
            reader = PdfReader(path, password=PDF_PASSWORD)

            if reader.is_encrypted:
                raise RuntimeError(f"PDF is encrypted: {path}")

            # Extract text from all pages
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            full_text = "\n\n".join(text_parts).strip()

            if not full_text:
                raise ValueError(
                    f"No text could be extracted from PDF: {path.name}"
                )

            if len(full_text) > MAX_TEXT_LENGTH:
                raise ValueError(
                    f"PDF text exceeds max length ({len(full_text)} > {MAX_TEXT_LENGTH})"
                )

            # Extract metadata
            metadata = reader.metadata or {}

            logger.info(
                "PDF parsed successfully: %s (%d pages, %d chars)",
                path.name,
                len(reader.pages),
                len(full_text),
            )

            return ParsedDocument(
                text=full_text,
                source_path=str(path),
                format="pdf",
                extracted_at=datetime.now(timezone.utc).isoformat(),
                parser_version=self.parser_version,
                title=metadata.get("/Title"),
                author=metadata.get("/Author"),
                created_date=metadata.get("/CreationDate"),
                page_count=len(reader.pages),
            )

        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            logger.error("Failed to parse PDF %s: %s", path.name, e)
            raise RuntimeError(f"PDF parsing failed: {e}") from e
