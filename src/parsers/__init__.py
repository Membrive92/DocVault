"""
Document parsers module.

Provides parsers for extracting text and metadata from PDF, HTML,
and Markdown documents. Use ParserFactory for automatic format detection.
"""

from __future__ import annotations

from .base_parser import DocumentParser
from .html_parser import HTMLParser
from .markdown_parser import MarkdownParser
from .models import ParsedDocument
from .parser_factory import ParserFactory
from .pdf_parser import PDFParser

__all__ = [
    "DocumentParser",
    "HTMLParser",
    "MarkdownParser",
    "ParsedDocument",
    "ParserFactory",
    "PDFParser",
]
