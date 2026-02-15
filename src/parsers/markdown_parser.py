"""
Markdown document parser.

Extracts text and frontmatter metadata from Markdown files.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import frontmatter

from .base_parser import DocumentParser
from .config import MAX_TEXT_LENGTH, MD_PARSE_FRONTMATTER, MD_REMOVE_HTML_TAGS
from .models import ParsedDocument

logger = logging.getLogger(__name__)


class MarkdownParser(DocumentParser):
    """
    Parser for Markdown documents.

    Extracts text and optionally parses YAML frontmatter for metadata.
    """

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if file is Markdown."""
        return Path(file_path).suffix.lower() in (".md", ".markdown")

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a Markdown file and extract text + metadata.

        Args:
            file_path: Path to Markdown file.

        Returns:
            ParsedDocument with extracted content.

        Raises:
            FileNotFoundError: If Markdown file doesn't exist.
            ValueError: If no text could be extracted.
            RuntimeError: If parsing fails.
        """
        path = Path(file_path)
        self._validate_file(path)

        logger.info("Parsing Markdown: %s", path.name)

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            title = None
            author = None

            # Parse frontmatter if enabled
            if MD_PARSE_FRONTMATTER:
                post = frontmatter.loads(content)
                text = post.content
                metadata = post.metadata
                title = metadata.get("title")
                author = metadata.get("author")
            else:
                text = content

            # Remove HTML tags if configured
            if MD_REMOVE_HTML_TAGS:
                text = re.sub(r"<[^>]+>", "", text)

            text = text.strip()

            if not text:
                raise ValueError(
                    f"No text could be extracted from Markdown: {path.name}"
                )

            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(
                    f"Markdown text exceeds max length ({len(text)} > {MAX_TEXT_LENGTH})"
                )

            # If no frontmatter title, extract from first heading
            if not title:
                heading_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
                if heading_match:
                    title = heading_match.group(1).strip()

            logger.info(
                "Markdown parsed successfully: %s (%d chars)",
                path.name,
                len(text),
            )

            return ParsedDocument(
                text=text,
                source_path=str(path),
                format="markdown",
                extracted_at=datetime.now(timezone.utc).isoformat(),
                parser_version=self.parser_version,
                title=title,
                author=author,
            )

        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            logger.error("Failed to parse Markdown %s: %s", path.name, e)
            raise RuntimeError(f"Markdown parsing failed: {e}") from e
