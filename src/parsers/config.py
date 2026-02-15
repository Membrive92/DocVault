"""
Configuration for document parsers.

Module-level constants for PDF, HTML, and Markdown parsing.
"""

from __future__ import annotations

# ==========================================
# PDF Configuration
# ==========================================
PDF_EXTRACT_IMAGES = False  # Don't extract images (text only)
PDF_PASSWORD = None         # Default no password

# ==========================================
# HTML Configuration
# ==========================================
HTML_REMOVE_TAGS = [
    "script", "style", "nav", "header", "footer",
    "aside", "form", "iframe", "noscript",
]
HTML_REMOVE_CLASSES = [
    "sidebar", "menu", "navigation", "ad", "advertisement",
    "cookie-notice", "social-share", "comments",
]
HTML_KEEP_LINKS = True       # Preserve link text
HTML_MIN_TEXT_LENGTH = 10    # Ignore elements with < 10 chars

# ==========================================
# Markdown Configuration
# ==========================================
MD_PARSE_FRONTMATTER = True      # Extract YAML frontmatter
MD_PRESERVE_CODE_BLOCKS = True   # Keep code blocks
MD_REMOVE_HTML_TAGS = True       # Strip HTML from Markdown

# ==========================================
# General
# ==========================================
DEFAULT_LANGUAGE = "en"
MAX_TEXT_LENGTH = 10_000_000  # 10MB max per document
PARSER_VERSION = "1.0.0"
