# Milestone 4: Document Parsers

**Status:** ⏸️ Pending
**Dependencies:** M1 (Foundation)
**Goal:** Extract clean text from PDF, HTML, and Markdown documents

---

## Overview

This milestone implements parsers for the three primary documentation formats:
- **PDF** - Technical documentation, manuals, research papers
- **HTML** - Web documentation, API references, knowledge bases
- **Markdown** - README files, wikis, developer docs

Each parser must extract clean, structured text suitable for embedding generation.

## Why These Formats?

### PDF
- **Use case**: Technical manuals, API documentation exports, research papers
- **Challenge**: Complex layout, tables, multi-column text
- **Library**: `pypdf` (pure Python, no system dependencies)

### HTML
- **Use case**: Web documentation (ReadTheDocs, Sphinx, MkDocs)
- **Challenge**: Navigation elements, ads, boilerplate
- **Library**: `beautifulsoup4` + `lxml` (robust HTML parsing)

### Markdown
- **Use case**: GitHub README, wikis, developer documentation
- **Challenge**: Simple format, minimal parsing needed
- **Library**: Standard library + simple regex

## Architecture

```
┌────────────────────────────────────────────────────────┐
│              DocumentParser (ABC)                      │
│  - parse(file_path) -> ParsedDocument                  │
│  - extract_metadata(file_path) -> dict                 │
└────────────────────────────────────────────────────────┘
                         ▲
                         │ implements
        ┌────────────────┼────────────────┐
        │                │                │
┌───────────────┐ ┌─────────────┐ ┌──────────────┐
│   PDFParser   │ │ HTMLParser  │ │ MDParser     │
│               │ │             │ │              │
│ - pypdf       │ │ - bs4       │ │ - stdlib     │
│ - extract     │ │ - clean     │ │ - frontmatter│
│   text        │ │   boiler    │ │   extraction │
│ - extract     │ │   plate     │ │              │
│   metadata    │ │             │ │              │
└───────────────┘ └─────────────┘ └──────────────┘
```

### ParsedDocument Data Model

```python
@dataclass
class ParsedDocument:
    """Result of parsing a document."""

    # Content
    text: str                    # Extracted clean text

    # Metadata
    title: Optional[str]         # Document title
    source_path: str             # Original file path
    format: str                  # "pdf", "html", or "markdown"

    # Optional metadata
    author: Optional[str] = None
    created_date: Optional[str] = None
    language: Optional[str] = None
    page_count: Optional[int] = None

    # Extraction info
    extracted_at: str            # ISO timestamp
    parser_version: str          # Parser version for tracking
```

## Implementation Plan

### Task 1: Install Parser Dependencies

```bash
pip install pypdf beautifulsoup4 lxml python-frontmatter
```

**Why these libraries?**
- **pypdf**: Pure Python, actively maintained, no system deps
- **beautifulsoup4**: Industry standard for HTML parsing
- **lxml**: Fast XML/HTML parser backend
- **python-frontmatter**: YAML frontmatter in Markdown files

### Task 2: Create Parser Configuration

**File:** `src/parsers/config.py`

```python
"""
Configuration for document parsers.
"""

from __future__ import annotations

# PDF Configuration
PDF_EXTRACT_IMAGES = False  # Don't extract images (text only)
PDF_PASSWORD = None         # Default no password

# HTML Configuration
HTML_REMOVE_TAGS = [
    "script", "style", "nav", "header", "footer",
    "aside", "form", "iframe", "noscript"
]
HTML_REMOVE_CLASSES = [
    "sidebar", "menu", "navigation", "ad", "advertisement",
    "cookie-notice", "social-share", "comments"
]
HTML_KEEP_LINKS = True      # Preserve link text
HTML_MIN_TEXT_LENGTH = 10   # Ignore elements with < 10 chars

# Markdown Configuration
MD_PARSE_FRONTMATTER = True  # Extract YAML frontmatter
MD_PRESERVE_CODE_BLOCKS = True  # Keep code blocks
MD_REMOVE_HTML_TAGS = True   # Strip HTML from Markdown

# General
DEFAULT_LANGUAGE = "en"
MAX_TEXT_LENGTH = 10_000_000  # 10MB max per document
```

### Task 3: Create ParsedDocument Model

**File:** `src/parsers/models.py`

```python
"""
Data models for parsed documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ParsedDocument:
    """
    Represents a parsed document with extracted text and metadata.

    This is the standard format returned by all parsers.
    """

    # Required fields
    text: str
    source_path: str
    format: str  # "pdf", "html", "markdown"
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
            raise ValueError(f"Invalid format: {self.format}")

    @property
    def word_count(self) -> int:
        """Calculate approximate word count."""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """Get character count."""
        return len(self.text)
```

### Task 4: Abstract Parser Interface

**File:** `src/parsers/base_parser.py`

```python
"""
Abstract base class for document parsers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .models import ParsedDocument


class DocumentParser(ABC):
    """
    Abstract base class for all document parsers.

    Each parser must implement parse() to extract text and metadata
    from a specific document format.
    """

    def __init__(self) -> None:
        """Initialize parser."""
        self.parser_version = "1.0.0"

    @abstractmethod
    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a document and extract text + metadata.

        Args:
            file_path: Path to the document file

        Returns:
            ParsedDocument with extracted content and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
            RuntimeError: If parsing fails
        """
        pass

    @abstractmethod
    def can_parse(self, file_path: str | Path) -> bool:
        """
        Check if this parser can handle the given file.

        Args:
            file_path: Path to check

        Returns:
            True if this parser supports the file format
        """
        pass

    def _validate_file(self, file_path: Path) -> None:
        """Validate that file exists and is readable."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")
```

### Task 5: Implement PDF Parser

**File:** `src/parsers/pdf_parser.py`

```python
"""
PDF document parser using pypdf.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

from .base_parser import DocumentParser
from .config import PDF_EXTRACT_IMAGES, PDF_PASSWORD, MAX_TEXT_LENGTH
from .models import ParsedDocument


logger = logging.getLogger(__name__)


class PDFParser(DocumentParser):
    """
    Parser for PDF documents.

    Uses pypdf to extract text and metadata from PDF files.
    """

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if file is a PDF."""
        path = Path(file_path)
        return path.suffix.lower() == ".pdf"

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a PDF file and extract text + metadata.

        Args:
            file_path: Path to PDF file

        Returns:
            ParsedDocument with extracted content

        Raises:
            FileNotFoundError: If PDF doesn't exist
            RuntimeError: If PDF is encrypted or corrupted
        """
        path = Path(file_path)
        self._validate_file(path)

        logger.info(f"Parsing PDF: {path.name}")

        try:
            reader = PdfReader(path, password=PDF_PASSWORD)

            # Check if encrypted
            if reader.is_encrypted:
                raise RuntimeError(f"PDF is encrypted: {path}")

            # Extract text from all pages
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            full_text = "\n\n".join(text_parts).strip()

            # Validate text length
            if len(full_text) > MAX_TEXT_LENGTH:
                raise ValueError(f"PDF text exceeds max length: {len(full_text)}")

            if not full_text:
                logger.warning(f"No text extracted from: {path.name}")

            # Extract metadata
            metadata = reader.metadata or {}

            return ParsedDocument(
                text=full_text,
                source_path=str(path),
                format="pdf",
                extracted_at=datetime.utcnow().isoformat(),
                parser_version=self.parser_version,
                title=metadata.get("/Title"),
                author=metadata.get("/Author"),
                created_date=metadata.get("/CreationDate"),
                page_count=len(reader.pages),
            )

        except Exception as e:
            logger.error(f"Failed to parse PDF {path.name}: {e}")
            raise RuntimeError(f"PDF parsing failed: {e}") from e
```

### Task 6: Implement HTML Parser

**File:** `src/parsers/html_parser.py`

```python
"""
HTML document parser using BeautifulSoup.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from .base_parser import DocumentParser
from .config import (
    HTML_REMOVE_TAGS,
    HTML_REMOVE_CLASSES,
    HTML_MIN_TEXT_LENGTH,
    MAX_TEXT_LENGTH,
)
from .models import ParsedDocument


logger = logging.getLogger(__name__)


class HTMLParser(DocumentParser):
    """
    Parser for HTML documents.

    Uses BeautifulSoup to extract main content while removing
    navigation, ads, and other boilerplate.
    """

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if file is HTML."""
        path = Path(file_path)
        return path.suffix.lower() in (".html", ".htm")

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse an HTML file and extract main content.

        Args:
            file_path: Path to HTML file

        Returns:
            ParsedDocument with extracted content
        """
        path = Path(file_path)
        self._validate_file(path)

        logger.info(f"Parsing HTML: {path.name}")

        try:
            # Read HTML
            with open(path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, "lxml")

            # Remove unwanted tags
            for tag in HTML_REMOVE_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()

            # Remove elements with unwanted classes
            for class_name in HTML_REMOVE_CLASSES:
                for element in soup.find_all(class_=class_name):
                    element.decompose()

            # Extract title
            title_tag = soup.find("title")
            title = title_tag.get_text().strip() if title_tag else None

            # Try to find main content
            main_content = (
                soup.find("main") or
                soup.find("article") or
                soup.find("div", class_="content") or
                soup.body or
                soup
            )

            # Extract text
            text = main_content.get_text(separator="\n", strip=True)

            # Clean up extra whitespace
            lines = [line.strip() for line in text.split("\n")]
            lines = [line for line in lines if len(line) >= HTML_MIN_TEXT_LENGTH]
            full_text = "\n".join(lines)

            if len(full_text) > MAX_TEXT_LENGTH:
                raise ValueError(f"HTML text exceeds max length: {len(full_text)}")

            return ParsedDocument(
                text=full_text,
                source_path=str(path),
                format="html",
                extracted_at=datetime.utcnow().isoformat(),
                parser_version=self.parser_version,
                title=title,
            )

        except Exception as e:
            logger.error(f"Failed to parse HTML {path.name}: {e}")
            raise RuntimeError(f"HTML parsing failed: {e}") from e
```

### Task 7: Implement Markdown Parser

**File:** `src/parsers/markdown_parser.py`

```python
"""
Markdown document parser.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import frontmatter

from .base_parser import DocumentParser
from .config import MD_PARSE_FRONTMATTER, MD_REMOVE_HTML_TAGS, MAX_TEXT_LENGTH
from .models import ParsedDocument


logger = logging.getLogger(__name__)


class MarkdownParser(DocumentParser):
    """
    Parser for Markdown documents.

    Extracts text and frontmatter metadata from Markdown files.
    """

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if file is Markdown."""
        path = Path(file_path)
        return path.suffix.lower() in (".md", ".markdown")

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a Markdown file.

        Args:
            file_path: Path to Markdown file

        Returns:
            ParsedDocument with extracted content
        """
        path = Path(file_path)
        self._validate_file(path)

        logger.info(f"Parsing Markdown: {path.name}")

        try:
            # Read file
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse frontmatter if enabled
            if MD_PARSE_FRONTMATTER:
                post = frontmatter.loads(content)
                text = post.content
                metadata = post.metadata
                title = metadata.get("title")
                author = metadata.get("author")
            else:
                text = content
                title = None
                author = None

            # Remove HTML tags if configured
            if MD_REMOVE_HTML_TAGS:
                text = re.sub(r"<[^>]+>", "", text)

            # Clean up
            text = text.strip()

            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(f"Markdown text exceeds max length: {len(text)}")

            # If no frontmatter title, try to extract from first heading
            if not title:
                heading_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
                if heading_match:
                    title = heading_match.group(1).strip()

            return ParsedDocument(
                text=text,
                source_path=str(path),
                format="markdown",
                extracted_at=datetime.utcnow().isoformat(),
                parser_version=self.parser_version,
                title=title,
                author=author,
            )

        except Exception as e:
            logger.error(f"Failed to parse Markdown {path.name}: {e}")
            raise RuntimeError(f"Markdown parsing failed: {e}") from e
```

### Task 8: Parser Factory

**File:** `src/parsers/parser_factory.py`

```python
"""
Factory for selecting the correct parser based on file extension.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base_parser import DocumentParser
from .html_parser import HTMLParser
from .markdown_parser import MarkdownParser
from .pdf_parser import PDFParser


class ParserFactory:
    """
    Factory for creating appropriate parser based on file type.
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
        Get appropriate parser for the given file.

        Args:
            file_path: Path to file

        Returns:
            Parser instance if format is supported, None otherwise
        """
        path = Path(file_path)

        for parser in self.parsers:
            if parser.can_parse(path):
                return parser

        return None

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a document using the appropriate parser.

        Args:
            file_path: Path to document

        Returns:
            ParsedDocument with extracted content

        Raises:
            ValueError: If file format is not supported
        """
        parser = self.get_parser(file_path)

        if parser is None:
            raise ValueError(f"Unsupported file format: {file_path}")

        return parser.parse(file_path)
```

### Task 9: Unit Tests

**File:** `tests/test_parsers.py`

Tests for each parser:
- PDF: Extract text, metadata, handle encrypted PDFs
- HTML: Remove boilerplate, extract main content
- Markdown: Parse frontmatter, extract headings
- Factory: Correct parser selection

### Task 10: Interactive Verification

**File:** `scripts/test_parsers.py`

Create sample documents and verify parsing:
1. Generate test PDF, HTML, Markdown files
2. Parse each format
3. Display extracted text and metadata
4. Verify accuracy

## Testing Strategy

### Sample Documents
Create in `data/documents/samples/`:
- `sample.pdf` - Multi-page technical doc
- `sample.html` - Documentation page with navigation
- `sample.md` - README with frontmatter

### Validation
- Text extraction accuracy > 95%
- Metadata extraction where available
- Proper handling of edge cases (empty files, malformed content)

## Next Steps (M5)

M5 will use these parsers to build the ingestion pipeline that:
1. Scans document directories
2. Selects appropriate parser
3. Chunks text into segments
4. Generates embeddings
5. Stores in vector database

---

**Related Files:**
- `src/parsers/config.py` - Parser configuration
- `src/parsers/models.py` - ParsedDocument model
- `src/parsers/base_parser.py` - Abstract interface
- `src/parsers/pdf_parser.py` - PDF implementation
- `src/parsers/html_parser.py` - HTML implementation
- `src/parsers/markdown_parser.py` - Markdown implementation
- `src/parsers/parser_factory.py` - Parser selection
- `tests/test_parsers.py` - Unit tests
