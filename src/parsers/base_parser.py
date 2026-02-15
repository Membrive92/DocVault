"""
Abstract base class for document parsers.

Defines the interface that all document parsers must implement.
Follows the same Strategy pattern used in VectorDatabase (M3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .config import PARSER_VERSION
from .models import ParsedDocument


class DocumentParser(ABC):
    """
    Abstract base class for all document parsers.

    Each parser must implement parse() to extract text and metadata
    from a specific document format, and can_parse() to indicate
    which file types it supports.
    """

    def __init__(self) -> None:
        """Initialize parser with default version."""
        self.parser_version = PARSER_VERSION

    @abstractmethod
    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a document and extract text + metadata.

        Args:
            file_path: Path to the document file.

        Returns:
            ParsedDocument with extracted content and metadata.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is invalid or text is empty.
            RuntimeError: If parsing fails.
        """
        pass

    @abstractmethod
    def can_parse(self, file_path: str | Path) -> bool:
        """
        Check if this parser can handle the given file.

        Args:
            file_path: Path to check.

        Returns:
            True if this parser supports the file format.
        """
        pass

    def _validate_file(self, file_path: Path) -> None:
        """
        Validate that file exists and is a regular file.

        Args:
            file_path: Path to validate.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If path is not a file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")
