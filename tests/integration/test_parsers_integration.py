"""
Integration tests for document parsers (M4).

These tests use more realistic document content to verify parsing
quality. They test the full pipeline: file reading, content extraction,
metadata parsing, and boilerplate removal.

Run with: pytest tests/integration/test_parsers_integration.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.parsers import (
    HTMLParser,
    MarkdownParser,
    ParsedDocument,
    ParserFactory,
    PDFParser,
)


def _create_text_pdf(path: Path, filename: str = "integration_test.pdf") -> Path:
    """Create a minimal PDF with extractable text."""
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000280 00000 n \n"
        b"0000000376 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n457\n%%EOF"
    )

    pdf_path = path / filename
    pdf_path.write_bytes(pdf_content)
    return pdf_path


class TestHTMLParserIntegration:
    """Integration tests for HTML parser with realistic content."""

    @pytest.fixture()
    def parser(self) -> HTMLParser:
        """Create an HTMLParser instance."""
        return HTMLParser()

    def test_parse_documentation_page_with_boilerplate(
        self, parser: HTMLParser, tmp_path: Path
    ) -> None:
        """Test parsing a realistic documentation page with navigation and sidebar."""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <title>API Reference - DocVault</title>
    <style>
        body { font-family: Arial; }
        .sidebar { width: 200px; }
    </style>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Page loaded');
        });
    </script>
</head>
<body>
    <nav>
        <ul>
            <li><a href="/">Home</a></li>
            <li><a href="/docs">Docs</a></li>
            <li><a href="/api">API</a></li>
        </ul>
    </nav>

    <aside class="sidebar">
        <h3>Navigation</h3>
        <ul>
            <li><a href="#auth">Authentication</a></li>
            <li><a href="#endpoints">Endpoints</a></li>
        </ul>
    </aside>

    <main>
        <h1>REST API Reference</h1>

        <section id="auth">
            <h2>Authentication</h2>
            <p>All API requests require a valid API key passed in the Authorization header.
            You can generate an API key from the dashboard settings page.</p>
        </section>

        <section id="endpoints">
            <h2>Endpoints</h2>
            <p>The API provides endpoints for document management, search, and user operations.
            All endpoints return JSON responses with standard HTTP status codes.</p>

            <h3>POST /api/documents</h3>
            <p>Upload a new document for processing. The document will be parsed, chunked,
            and indexed automatically. Supported formats: PDF, HTML, Markdown.</p>

            <h3>GET /api/search</h3>
            <p>Search across indexed documents using natural language queries.
            Returns ranked results with relevance scores and source references.</p>
        </section>
    </main>

    <footer>
        <p>Copyright 2026 DocVault Team. All rights reserved.</p>
        <div class="social-share">Share on Twitter | LinkedIn</div>
    </footer>
</body>
</html>"""

        html_path = tmp_path / "api_docs.html"
        html_path.write_text(html_content, encoding="utf-8")

        result = parser.parse(html_path)

        # Title should be extracted
        assert result.title == "API Reference - DocVault"

        # Main content should be present
        assert "REST API Reference" in result.text
        assert "Authentication" in result.text
        assert "API requests require a valid API key" in result.text
        assert "POST /api/documents" in result.text

        # Boilerplate should be removed
        assert "console.log" not in result.text
        assert "font-family" not in result.text
        assert "Copyright 2026" not in result.text
        assert "Share on Twitter" not in result.text

    def test_parse_html_with_content_div(
        self, parser: HTMLParser, tmp_path: Path
    ) -> None:
        """Test parsing HTML where main content is in a div.content."""
        html_content = """<!DOCTYPE html>
<html>
<head><title>Blog Post</title></head>
<body>
    <header><h1>My Blog</h1></header>
    <div class="content">
        <h2>Understanding Vector Databases</h2>
        <p>Vector databases are specialized systems designed to store and query
        high-dimensional vector data efficiently. They use approximate nearest
        neighbor algorithms like HNSW for fast similarity search.</p>
    </div>
    <footer>Blog Footer Content</footer>
</body>
</html>"""

        html_path = tmp_path / "blog.html"
        html_path.write_text(html_content, encoding="utf-8")

        result = parser.parse(html_path)

        assert "Vector Databases" in result.text
        assert "HNSW" in result.text


class TestMarkdownParserIntegration:
    """Integration tests for Markdown parser with realistic content."""

    @pytest.fixture()
    def parser(self) -> MarkdownParser:
        """Create a MarkdownParser instance."""
        return MarkdownParser()

    def test_parse_full_readme(
        self, parser: MarkdownParser, tmp_path: Path
    ) -> None:
        """Test parsing a realistic README with frontmatter and mixed content."""
        md_content = """---
title: DocVault
author: Development Team
date: 2026-01-15
tags:
  - rag
  - python
  - ai
---

# DocVault

A RAG system for querying enterprise documentation.

## Features

- **PDF parsing**: Extract text from PDF documents
- **HTML parsing**: Clean extraction from web pages
- **Markdown parsing**: Parse developer documentation
- **Vector search**: Semantic similarity search with Qdrant

## Installation

```bash
git clone https://github.com/example/docvault.git
cd docvault
pip install -r requirements.txt
```

## Quick Start

```python
from src.parsers import ParserFactory

factory = ParserFactory()
doc = factory.parse("path/to/document.pdf")
print(doc.text)
```

## Architecture

The system follows a layered architecture:

1. **Parsers** - Extract text from documents
2. **Embeddings** - Convert text to vectors
3. **Vector DB** - Store and search vectors
4. **RAG Pipeline** - Retrieve context and generate answers

## License

MIT License
"""

        md_path = tmp_path / "README.md"
        md_path.write_text(md_content, encoding="utf-8")

        result = parser.parse(md_path)

        # Frontmatter metadata extracted
        assert result.title == "DocVault"
        assert result.author == "Development Team"

        # Content preserved
        assert "RAG system" in result.text
        assert "PDF parsing" in result.text
        assert "pip install" in result.text
        assert "Architecture" in result.text

        # Code blocks preserved
        assert "ParserFactory" in result.text

    def test_parse_markdown_with_html_embedded(
        self, parser: MarkdownParser, tmp_path: Path
    ) -> None:
        """Test that embedded HTML is stripped from Markdown."""
        md_content = """# Guide

Some text with <div class="warning">embedded HTML warning</div> content.

<table>
<tr><td>Cell 1</td><td>Cell 2</td></tr>
</table>

More regular markdown content follows here.
"""

        md_path = tmp_path / "html_mixed.md"
        md_path.write_text(md_content, encoding="utf-8")

        result = parser.parse(md_path)

        # HTML tags removed but text preserved
        assert "<div" not in result.text
        assert "<table>" not in result.text
        assert "embedded HTML warning" in result.text
        assert "Cell 1" in result.text


class TestPDFParserIntegration:
    """Integration tests for PDF parser."""

    @pytest.fixture()
    def parser(self) -> PDFParser:
        """Create a PDFParser instance."""
        return PDFParser()

    def test_parse_pdf_returns_valid_document(
        self, parser: PDFParser, tmp_path: Path
    ) -> None:
        """Test parsing a PDF returns a valid ParsedDocument."""
        pdf_path = _create_text_pdf(tmp_path)
        result = parser.parse(pdf_path)

        assert isinstance(result, ParsedDocument)
        assert result.format == "pdf"
        assert result.page_count >= 1
        assert result.word_count > 0
        assert result.char_count > 0
        assert result.extracted_at is not None


class TestParserFactoryIntegration:
    """Integration tests for ParserFactory with multiple formats."""

    @pytest.fixture()
    def factory(self) -> ParserFactory:
        """Create a ParserFactory instance."""
        return ParserFactory()

    def test_factory_parses_all_formats(
        self, factory: ParserFactory, tmp_path: Path
    ) -> None:
        """Test that factory can parse HTML, Markdown, and PDF files."""
        # Create HTML
        html_path = tmp_path / "test.html"
        html_path.write_text(
            "<html><head><title>HTML</title></head><body>"
            "<p>This is an HTML document with enough content to pass the minimum length filter.</p>"
            "</body></html>",
            encoding="utf-8",
        )

        # Create Markdown
        md_path = tmp_path / "test.md"
        md_path.write_text(
            "# Markdown\n\nThis is a Markdown document.",
            encoding="utf-8",
        )

        # Create PDF
        pdf_path = _create_text_pdf(tmp_path, "test.pdf")

        # Parse all
        html_result = factory.parse(html_path)
        md_result = factory.parse(md_path)
        pdf_result = factory.parse(pdf_path)

        assert html_result.format == "html"
        assert md_result.format == "markdown"
        assert pdf_result.format == "pdf"

    def test_factory_rejects_unsupported_format(
        self, factory: ParserFactory, tmp_path: Path
    ) -> None:
        """Test that factory raises ValueError for unsupported formats."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("Plain text content", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported file format"):
            factory.parse(txt_path)
