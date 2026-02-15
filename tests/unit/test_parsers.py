"""
Unit tests for document parsers module.

Tests the PDFParser, HTMLParser, MarkdownParser, ParserFactory,
and ParsedDocument model. Uses temporary files created with pytest's
tmp_path fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.parsers import (
    DocumentParser,
    HTMLParser,
    MarkdownParser,
    ParsedDocument,
    ParserFactory,
    PDFParser,
)


# ==========================================
# Helper: Create test files
# ==========================================


def _create_pdf(path: Path, text: str, num_pages: int = 1) -> Path:
    """Create a simple PDF file with text content using pypdf."""
    writer = PdfWriter()

    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)

    # pypdf PdfWriter doesn't easily support adding text to blank pages,
    # so we create a PDF with annotations that contain the text.
    # For unit tests, we'll use a real PDF created via reportlab-like approach.
    # Instead, let's write a minimal valid PDF with text directly.
    pdf_path = path / "test.pdf"

    # Write the PDF
    with open(pdf_path, "wb") as f:
        writer.write(f)

    return pdf_path


def _create_text_pdf(path: Path, filename: str = "test.pdf") -> Path:
    """
    Create a minimal PDF with extractable text.

    Uses pypdf's append mechanism to build a PDF with real text content.
    """
    from io import BytesIO

    # Minimal valid PDF with text "Hello World"
    # This is a hand-crafted minimal PDF that pypdf can read
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


def _create_html(path: Path, filename: str = "test.html") -> Path:
    """Create a simple HTML file for testing."""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
</head>
<body>
    <main>
        <h1>Machine Learning Overview</h1>
        <p>Machine learning is a branch of artificial intelligence that focuses on building
        systems that learn from data. These systems improve their performance over time
        without being explicitly programmed.</p>
        <p>Common applications include image recognition, natural language processing,
        and recommendation systems.</p>
    </main>
</body>
</html>"""

    html_path = path / filename
    html_path.write_text(html_content, encoding="utf-8")
    return html_path


def _create_markdown(path: Path, filename: str = "test.md") -> Path:
    """Create a simple Markdown file for testing."""
    md_content = """---
title: Test Document
author: John Doe
---

# Machine Learning Guide

Machine learning is a powerful technology that enables computers
to learn from data without being explicitly programmed.

## Supervised Learning

Supervised learning uses labeled data to train models.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.
"""

    md_path = path / filename
    md_path.write_text(md_content, encoding="utf-8")
    return md_path


# ==========================================
# ParsedDocument model tests
# ==========================================


class TestParsedDocument:
    """Tests for the ParsedDocument dataclass."""

    def test_create_valid_document(self) -> None:
        """Test creating a valid ParsedDocument."""
        doc = ParsedDocument(
            text="Some content here",
            source_path="/path/to/file.pdf",
            format="pdf",
            extracted_at="2026-01-01T00:00:00",
            parser_version="1.0.0",
        )

        assert doc.text == "Some content here"
        assert doc.format == "pdf"
        assert doc.title is None

    def test_optional_metadata(self) -> None:
        """Test creating a document with optional metadata."""
        doc = ParsedDocument(
            text="Content",
            source_path="/path/to/file.html",
            format="html",
            extracted_at="2026-01-01T00:00:00",
            parser_version="1.0.0",
            title="My Title",
            author="Jane Doe",
            page_count=5,
        )

        assert doc.title == "My Title"
        assert doc.author == "Jane Doe"
        assert doc.page_count == 5

    def test_word_count_property(self) -> None:
        """Test that word_count returns correct count."""
        doc = ParsedDocument(
            text="one two three four five",
            source_path="/test",
            format="pdf",
            extracted_at="2026-01-01T00:00:00",
            parser_version="1.0.0",
        )

        assert doc.word_count == 5

    def test_char_count_property(self) -> None:
        """Test that char_count returns correct count."""
        doc = ParsedDocument(
            text="hello",
            source_path="/test",
            format="pdf",
            extracted_at="2026-01-01T00:00:00",
            parser_version="1.0.0",
        )

        assert doc.char_count == 5

    def test_empty_text_raises_error(self) -> None:
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="empty text"):
            ParsedDocument(
                text="",
                source_path="/test",
                format="pdf",
                extracted_at="2026-01-01T00:00:00",
                parser_version="1.0.0",
            )

    def test_whitespace_only_text_raises_error(self) -> None:
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="empty text"):
            ParsedDocument(
                text="   \n\t  ",
                source_path="/test",
                format="pdf",
                extracted_at="2026-01-01T00:00:00",
                parser_version="1.0.0",
            )

    def test_invalid_format_raises_error(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid format"):
            ParsedDocument(
                text="Some content",
                source_path="/test",
                format="docx",
                extracted_at="2026-01-01T00:00:00",
                parser_version="1.0.0",
            )

    def test_valid_formats(self) -> None:
        """Test that all valid formats are accepted."""
        for fmt in ("pdf", "html", "markdown"):
            doc = ParsedDocument(
                text="Content",
                source_path="/test",
                format=fmt,
                extracted_at="2026-01-01T00:00:00",
                parser_version="1.0.0",
            )
            assert doc.format == fmt


# ==========================================
# PDFParser tests
# ==========================================


class TestPDFParser:
    """Tests for the PDFParser class."""

    @pytest.fixture()
    def parser(self) -> PDFParser:
        """Create a PDFParser instance."""
        return PDFParser()

    def test_can_parse_pdf(self, parser: PDFParser) -> None:
        """Test that can_parse returns True for .pdf files."""
        assert parser.can_parse("document.pdf") is True
        assert parser.can_parse("document.PDF") is True
        assert parser.can_parse(Path("path/to/file.pdf")) is True

    def test_cannot_parse_other_formats(self, parser: PDFParser) -> None:
        """Test that can_parse returns False for non-PDF files."""
        assert parser.can_parse("document.html") is False
        assert parser.can_parse("document.md") is False
        assert parser.can_parse("document.txt") is False

    def test_parse_pdf_extracts_text(
        self, parser: PDFParser, tmp_path: Path
    ) -> None:
        """Test that parse extracts text from a PDF."""
        pdf_path = _create_text_pdf(tmp_path)
        result = parser.parse(pdf_path)

        assert isinstance(result, ParsedDocument)
        assert "Hello World" in result.text
        assert result.format == "pdf"
        assert result.page_count == 1

    def test_parse_pdf_has_metadata(
        self, parser: PDFParser, tmp_path: Path
    ) -> None:
        """Test that parse sets correct metadata fields."""
        pdf_path = _create_text_pdf(tmp_path)
        result = parser.parse(pdf_path)

        assert result.source_path == str(pdf_path)
        assert result.extracted_at is not None
        assert result.parser_version == "1.0.0"

    def test_parse_nonexistent_file_raises_error(
        self, parser: PDFParser
    ) -> None:
        """Test that parsing a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            parser.parse("/nonexistent/path/document.pdf")

    def test_parser_has_version(self, parser: PDFParser) -> None:
        """Test that parser has a version attribute."""
        assert parser.parser_version == "1.0.0"

    def test_parser_is_document_parser(self, parser: PDFParser) -> None:
        """Test that PDFParser is a DocumentParser subclass."""
        assert isinstance(parser, DocumentParser)


# ==========================================
# HTMLParser tests
# ==========================================


class TestHTMLParser:
    """Tests for the HTMLParser class."""

    @pytest.fixture()
    def parser(self) -> HTMLParser:
        """Create an HTMLParser instance."""
        return HTMLParser()

    def test_can_parse_html(self, parser: HTMLParser) -> None:
        """Test that can_parse returns True for .html and .htm files."""
        assert parser.can_parse("page.html") is True
        assert parser.can_parse("page.htm") is True
        assert parser.can_parse("page.HTML") is True

    def test_cannot_parse_other_formats(self, parser: HTMLParser) -> None:
        """Test that can_parse returns False for non-HTML files."""
        assert parser.can_parse("document.pdf") is False
        assert parser.can_parse("document.md") is False

    def test_parse_html_extracts_text(
        self, parser: HTMLParser, tmp_path: Path
    ) -> None:
        """Test that parse extracts text from HTML."""
        html_path = _create_html(tmp_path)
        result = parser.parse(html_path)

        assert isinstance(result, ParsedDocument)
        assert "Machine Learning" in result.text or "machine learning" in result.text.lower()
        assert result.format == "html"

    def test_parse_html_extracts_title(
        self, parser: HTMLParser, tmp_path: Path
    ) -> None:
        """Test that parse extracts the title tag."""
        html_path = _create_html(tmp_path)
        result = parser.parse(html_path)

        assert result.title == "Test Document"

    def test_parse_html_removes_script_tags(
        self, parser: HTMLParser, tmp_path: Path
    ) -> None:
        """Test that script and style tags are removed."""
        html_content = """<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
    <script>alert('malicious');</script>
    <style>.hidden { display: none; }</style>
    <p>This is the actual content of the document that should be extracted.</p>
</body>
</html>"""

        html_path = tmp_path / "script_test.html"
        html_path.write_text(html_content, encoding="utf-8")

        result = parser.parse(html_path)

        assert "alert" not in result.text
        assert "display: none" not in result.text
        assert "actual content" in result.text

    def test_parse_html_removes_nav_elements(
        self, parser: HTMLParser, tmp_path: Path
    ) -> None:
        """Test that navigation elements are removed."""
        html_content = """<!DOCTYPE html>
<html>
<head><title>Nav Test</title></head>
<body>
    <nav><a href="/">Home</a><a href="/about">About</a></nav>
    <main>
        <p>This is the main content that should remain in the parsed output.</p>
    </main>
    <footer>Copyright 2026</footer>
</body>
</html>"""

        html_path = tmp_path / "nav_test.html"
        html_path.write_text(html_content, encoding="utf-8")

        result = parser.parse(html_path)

        assert "main content" in result.text
        # nav and footer should be removed
        assert "Copyright" not in result.text

    def test_parse_nonexistent_file_raises_error(
        self, parser: HTMLParser
    ) -> None:
        """Test that parsing a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            parser.parse("/nonexistent/path/page.html")

    def test_parser_is_document_parser(self, parser: HTMLParser) -> None:
        """Test that HTMLParser is a DocumentParser subclass."""
        assert isinstance(parser, DocumentParser)


# ==========================================
# MarkdownParser tests
# ==========================================


class TestMarkdownParser:
    """Tests for the MarkdownParser class."""

    @pytest.fixture()
    def parser(self) -> MarkdownParser:
        """Create a MarkdownParser instance."""
        return MarkdownParser()

    def test_can_parse_markdown(self, parser: MarkdownParser) -> None:
        """Test that can_parse returns True for .md and .markdown files."""
        assert parser.can_parse("README.md") is True
        assert parser.can_parse("guide.markdown") is True
        assert parser.can_parse("NOTES.MD") is True

    def test_cannot_parse_other_formats(self, parser: MarkdownParser) -> None:
        """Test that can_parse returns False for non-Markdown files."""
        assert parser.can_parse("document.pdf") is False
        assert parser.can_parse("page.html") is False

    def test_parse_markdown_extracts_text(
        self, parser: MarkdownParser, tmp_path: Path
    ) -> None:
        """Test that parse extracts text from Markdown."""
        md_path = _create_markdown(tmp_path)
        result = parser.parse(md_path)

        assert isinstance(result, ParsedDocument)
        assert "Machine Learning" in result.text or "machine learning" in result.text.lower()
        assert result.format == "markdown"

    def test_parse_markdown_extracts_frontmatter_title(
        self, parser: MarkdownParser, tmp_path: Path
    ) -> None:
        """Test that parse extracts title from YAML frontmatter."""
        md_path = _create_markdown(tmp_path)
        result = parser.parse(md_path)

        assert result.title == "Test Document"

    def test_parse_markdown_extracts_frontmatter_author(
        self, parser: MarkdownParser, tmp_path: Path
    ) -> None:
        """Test that parse extracts author from YAML frontmatter."""
        md_path = _create_markdown(tmp_path)
        result = parser.parse(md_path)

        assert result.author == "John Doe"

    def test_parse_markdown_without_frontmatter(
        self, parser: MarkdownParser, tmp_path: Path
    ) -> None:
        """Test parsing Markdown without frontmatter extracts heading as title."""
        md_content = """# Simple Guide

This is a simple document without YAML frontmatter.

## Section One

Some content here.
"""
        md_path = tmp_path / "no_frontmatter.md"
        md_path.write_text(md_content, encoding="utf-8")

        result = parser.parse(md_path)

        assert result.title == "Simple Guide"
        assert result.author is None

    def test_parse_markdown_removes_html_tags(
        self, parser: MarkdownParser, tmp_path: Path
    ) -> None:
        """Test that embedded HTML tags are removed."""
        md_content = """# Title

This has <strong>bold</strong> and <em>italic</em> HTML tags mixed in.
"""
        md_path = tmp_path / "html_in_md.md"
        md_path.write_text(md_content, encoding="utf-8")

        result = parser.parse(md_path)

        assert "<strong>" not in result.text
        assert "<em>" not in result.text
        assert "bold" in result.text
        assert "italic" in result.text

    def test_parse_nonexistent_file_raises_error(
        self, parser: MarkdownParser
    ) -> None:
        """Test that parsing a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            parser.parse("/nonexistent/path/README.md")

    def test_parser_is_document_parser(self, parser: MarkdownParser) -> None:
        """Test that MarkdownParser is a DocumentParser subclass."""
        assert isinstance(parser, DocumentParser)


# ==========================================
# ParserFactory tests
# ==========================================


class TestParserFactory:
    """Tests for the ParserFactory class."""

    @pytest.fixture()
    def factory(self) -> ParserFactory:
        """Create a ParserFactory instance."""
        return ParserFactory()

    def test_get_parser_for_pdf(self, factory: ParserFactory) -> None:
        """Test that factory returns PDFParser for .pdf files."""
        parser = factory.get_parser("document.pdf")
        assert isinstance(parser, PDFParser)

    def test_get_parser_for_html(self, factory: ParserFactory) -> None:
        """Test that factory returns HTMLParser for .html files."""
        parser = factory.get_parser("page.html")
        assert isinstance(parser, HTMLParser)

    def test_get_parser_for_htm(self, factory: ParserFactory) -> None:
        """Test that factory returns HTMLParser for .htm files."""
        parser = factory.get_parser("page.htm")
        assert isinstance(parser, HTMLParser)

    def test_get_parser_for_markdown(self, factory: ParserFactory) -> None:
        """Test that factory returns MarkdownParser for .md files."""
        parser = factory.get_parser("README.md")
        assert isinstance(parser, MarkdownParser)

    def test_get_parser_for_unsupported_format(
        self, factory: ParserFactory
    ) -> None:
        """Test that factory returns None for unsupported formats."""
        parser = factory.get_parser("document.docx")
        assert parser is None

    def test_parse_unsupported_format_raises_error(
        self, factory: ParserFactory
    ) -> None:
        """Test that parse raises ValueError for unsupported formats."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            factory.parse("document.docx")

    def test_parse_html_through_factory(
        self, factory: ParserFactory, tmp_path: Path
    ) -> None:
        """Test parsing an HTML file through the factory."""
        html_path = _create_html(tmp_path)
        result = factory.parse(html_path)

        assert isinstance(result, ParsedDocument)
        assert result.format == "html"

    def test_parse_markdown_through_factory(
        self, factory: ParserFactory, tmp_path: Path
    ) -> None:
        """Test parsing a Markdown file through the factory."""
        md_path = _create_markdown(tmp_path)
        result = factory.parse(md_path)

        assert isinstance(result, ParsedDocument)
        assert result.format == "markdown"

    def test_factory_has_all_parsers(self, factory: ParserFactory) -> None:
        """Test that factory has PDF, HTML, and Markdown parsers."""
        parser_types = [type(p) for p in factory.parsers]
        assert PDFParser in parser_types
        assert HTMLParser in parser_types
        assert MarkdownParser in parser_types
