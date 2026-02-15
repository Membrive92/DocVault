"""
Factory for selecting the correct parser based on file extension.

Provides a single entry point for parsing any supported document format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base_parser import DocumentParser
from .html_parser import HTMLParser
from .markdown_parser import MarkdownParser
from .models import ParsedDocument
from .pdf_parser import PDFParser

logger = logging.getLogger(__name__)


class ParserFactory:
    """
    Factory for creating the appropriate parser based on file type.

    Maintains a registry of all available parsers and selects
    the correct one based on file extension.
    """

    def __init__(self) -> None:
        """Initialize factory with all available parsers."""
        self.parsers: list[DocumentParser] = [
            PDFParser(),
            HTMLParser(),
            MarkdownParser(),
        ]

    def get_parser(self, file_path: str | Path) -> Optional[DocumentParser]:
        """
        Get the appropriate parser for the given file.

        Args:
            file_path: Path to the file.

        Returns:
            Parser instance if format is supported, None otherwise.
        """
        path = Path(file_path)

        for parser in self.parsers:
            if parser.can_parse(path):
                return parser

        return None

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a document using the appropriate parser.

        Convenience method that selects the parser and parses in one step.

        Args:
            file_path: Path to the document.

        Returns:
            ParsedDocument with extracted content.

        Raises:
            ValueError: If file format is not supported.
            FileNotFoundError: If file doesn't exist.
            RuntimeError: If parsing fails.
        """
        parser = self.get_parser(file_path)

        if parser is None:
            raise ValueError(
                f"Unsupported file format: {Path(file_path).suffix}"
            )

        logger.info(
            "Using %s for %s",
            type(parser).__name__,
            Path(file_path).name,
        )

        return parser.parse(file_path)
