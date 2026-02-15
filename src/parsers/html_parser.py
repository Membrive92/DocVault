"""
HTML document parser using BeautifulSoup.

Extracts main content while removing navigation, ads, and boilerplate.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from .base_parser import DocumentParser
from .config import (
    HTML_MIN_TEXT_LENGTH,
    HTML_REMOVE_CLASSES,
    HTML_REMOVE_TAGS,
    MAX_TEXT_LENGTH,
)
from .models import ParsedDocument

logger = logging.getLogger(__name__)


class HTMLParser(DocumentParser):
    """
    Parser for HTML documents.

    Uses BeautifulSoup to extract main content while removing
    navigation, ads, and other boilerplate elements.
    """

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if file is HTML."""
        return Path(file_path).suffix.lower() in (".html", ".htm")

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse an HTML file and extract main content.

        Args:
            file_path: Path to HTML file.

        Returns:
            ParsedDocument with extracted content.

        Raises:
            FileNotFoundError: If HTML file doesn't exist.
            ValueError: If no text could be extracted.
            RuntimeError: If parsing fails.
        """
        path = Path(file_path)
        self._validate_file(path)

        logger.info("Parsing HTML: %s", path.name)

        try:
            with open(path, "r", encoding="utf-8") as f:
                html_content = f.read()

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

            # Find main content area
            main_content = (
                soup.find("main")
                or soup.find("article")
                or soup.find("div", class_="content")
                or soup.body
                or soup
            )

            # Extract text
            text = main_content.get_text(separator="\n", strip=True)

            # Clean up: remove short lines and extra whitespace
            lines = [line.strip() for line in text.split("\n")]
            lines = [line for line in lines if len(line) >= HTML_MIN_TEXT_LENGTH]
            full_text = "\n".join(lines)

            if not full_text:
                raise ValueError(
                    f"No text could be extracted from HTML: {path.name}"
                )

            if len(full_text) > MAX_TEXT_LENGTH:
                raise ValueError(
                    f"HTML text exceeds max length ({len(full_text)} > {MAX_TEXT_LENGTH})"
                )

            logger.info(
                "HTML parsed successfully: %s (%d chars)",
                path.name,
                len(full_text),
            )

            return ParsedDocument(
                text=full_text,
                source_path=str(path),
                format="html",
                extracted_at=datetime.now(timezone.utc).isoformat(),
                parser_version=self.parser_version,
                title=title,
            )

        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            logger.error("Failed to parse HTML %s: %s", path.name, e)
            raise RuntimeError(f"HTML parsing failed: {e}") from e
