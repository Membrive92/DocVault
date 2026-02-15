# Milestone 4: Document Parsers

**Status:** ✅ Completed
**Dependencies:** M1 (Foundation)
**Goal:** Extract clean text from PDF, HTML, and Markdown documents

---

## Overview

This milestone implements parsers for the three primary documentation formats:
- **PDF** - Technical documentation, manuals, research papers
- **HTML** - Web documentation, API references, knowledge bases
- **Markdown** - README files, wikis, developer docs

Each parser extracts clean, structured text suitable for embedding generation,
returning a standardized `ParsedDocument` model.

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
- **Challenge**: YAML frontmatter, embedded HTML
- **Library**: `python-frontmatter` + regex

## Architecture

```
┌────────────────────────────────────────────────────────┐
│              DocumentParser (ABC)                       │
│  - parse(file_path) -> ParsedDocument                  │
│  - can_parse(file_path) -> bool                        │
│  - _validate_file(file_path) [concrete]                │
└────────────────────────────────────────────────────────┘
                         ▲
                         │ implements
        ┌────────────────┼────────────────┐
        │                │                │
┌───────────────┐ ┌─────────────┐ ┌──────────────┐
│   PDFParser   │ │ HTMLParser  │ │ MarkdownParser│
│               │ │             │ │              │
│ - pypdf       │ │ - bs4       │ │ - frontmatter│
│ - text +      │ │ - clean     │ │ - yaml meta  │
│   metadata    │ │   boiler    │ │ - heading    │
│ - page count  │ │   plate     │ │   extraction │
└───────────────┘ └─────────────┘ └──────────────┘
                         │
                         │ uses
                         ▼
              ┌────────────────────┐
              │   ParserFactory    │
              │  (auto-detection)  │
              └────────────────────┘
```

### ParsedDocument Data Model

```python
@dataclass
class ParsedDocument:
    """Standard output for all parsers."""

    # Required fields
    text: str                    # Extracted clean text
    source_path: str             # Original file path
    format: str                  # "pdf", "html", or "markdown"
    extracted_at: str            # ISO timestamp
    parser_version: str          # Parser version for tracking

    # Optional metadata
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[str] = None
    language: Optional[str] = None
    page_count: Optional[int] = None

    # Properties
    word_count: int              # Calculated from text
    char_count: int              # Calculated from text
```

Validation in `__post_init__`:
- Text cannot be empty or whitespace-only
- Format must be one of: `pdf`, `html`, `markdown`

## Implementation Details

### Parser Configuration (`src/parsers/config.py`)

```python
# PDF
PDF_EXTRACT_IMAGES = False
PDF_PASSWORD = None

# HTML — tags and classes to remove (boilerplate)
HTML_REMOVE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "iframe", "noscript"]
HTML_REMOVE_CLASSES = ["sidebar", "menu", "navigation", "ad", "advertisement", "cookie-notice", "social-share", "comments"]
HTML_MIN_TEXT_LENGTH = 10

# Markdown
MD_PARSE_FRONTMATTER = True
MD_REMOVE_HTML_TAGS = True

# General
MAX_TEXT_LENGTH = 10_000_000  # 10MB max per document
PARSER_VERSION = "1.0.0"
```

### Abstract Interface (`src/parsers/base_parser.py`)

Strategy pattern interface — same approach as `VectorDatabase` in M3:

```python
class DocumentParser(ABC):
    @abstractmethod
    def parse(self, file_path: str | Path) -> ParsedDocument: ...
    @abstractmethod
    def can_parse(self, file_path: str | Path) -> bool: ...
    def _validate_file(self, file_path: Path) -> None: ...  # concrete helper
```

### PDFParser (`src/parsers/pdf_parser.py`)

- Extracts text page by page with `PdfReader`
- Joins pages with `\n\n`
- Extracts metadata: `/Title`, `/Author`, `/CreationDate`
- Raises `RuntimeError` for encrypted PDFs
- Raises `ValueError` if no text extracted (image-only PDFs)

### HTMLParser (`src/parsers/html_parser.py`)

- Parses with `BeautifulSoup` using `lxml` backend
- Removes unwanted tags (`script`, `style`, `nav`, etc.)
- Removes elements with boilerplate classes (`sidebar`, `menu`, etc.)
- Finds main content: `<main>` > `<article>` > `div.content` > `<body>`
- Filters lines shorter than 10 chars (noise)
- Extracts `<title>` tag

### MarkdownParser (`src/parsers/markdown_parser.py`)

- Parses YAML frontmatter with `python-frontmatter` (title, author)
- Strips embedded HTML tags with regex
- Falls back to first `# Heading` as title if no frontmatter

### ParserFactory (`src/parsers/parser_factory.py`)

- Registers all three parsers
- Selects parser by file extension: `.pdf`, `.html`/`.htm`, `.md`/`.markdown`
- `get_parser()` returns parser or `None`
- `parse()` convenience method — select + parse in one call
- Raises `ValueError` for unsupported formats

## Usage Examples

### Direct parser usage

```python
from src.parsers import PDFParser, HTMLParser, MarkdownParser

pdf_parser = PDFParser()
doc = pdf_parser.parse("docs/manual.pdf")
print(f"Title: {doc.title}")
print(f"Pages: {doc.page_count}")
print(f"Words: {doc.word_count}")
```

### Factory usage (recommended)

```python
from src.parsers import ParserFactory

factory = ParserFactory()

# Auto-detects format by extension
doc = factory.parse("docs/manual.pdf")
doc = factory.parse("docs/api.html")
doc = factory.parse("docs/README.md")

# Check if format is supported
parser = factory.get_parser("file.docx")  # Returns None
```

## Testing

### Unit Tests (`tests/unit/test_parsers.py`)

41 tests covering:
- ParsedDocument model (8 tests): validation, properties, formats
- PDFParser (7 tests): can_parse, parse, metadata, errors
- HTMLParser (8 tests): can_parse, parse, title, script/nav removal, errors
- MarkdownParser (8 tests): can_parse, parse, frontmatter, heading extraction, HTML removal
- ParserFactory (10 tests): parser selection, factory parse, unsupported formats

```bash
pytest tests/unit/test_parsers.py -v
```

### Integration Tests (`tests/integration/test_parsers_integration.py`)

7 tests with realistic documents:
- HTML with full boilerplate (nav, sidebar, scripts, footer)
- HTML with `div.content` as main area
- Markdown with complete YAML frontmatter
- Markdown with embedded HTML
- PDF with extractable text
- ParserFactory with all formats
- ParserFactory rejects unsupported format

```bash
pytest tests/integration/test_parsers_integration.py -v
```

## Verification Criteria

**M4 is complete when:**
- [x] pypdf, beautifulsoup4, lxml, python-frontmatter installed
- [x] DocumentParser ABC defines parse() and can_parse()
- [x] PDFParser extracts text and metadata from PDFs
- [x] HTMLParser removes boilerplate and extracts main content
- [x] MarkdownParser extracts frontmatter and cleans HTML
- [x] ParserFactory auto-detects format by extension
- [x] ParsedDocument validates text and format
- [x] All 41 unit tests pass
- [x] All 7 integration tests pass
- [x] Documentation updated (README.md, AGENTS.md)

## Next Steps (M5)

With parsers ready, M5 will build the ingestion pipeline that:
1. Scans document directories for supported files
2. Selects the appropriate parser via ParserFactory
3. Chunks parsed text into segments (~500 tokens, 50 overlap)
4. Generates embeddings with EmbeddingService (M2)
5. Stores vectors in QdrantDatabase (M3)

---

**Related Files:**
- `src/parsers/__init__.py` - Module exports
- `src/parsers/config.py` - Parser configuration
- `src/parsers/models.py` - ParsedDocument dataclass
- `src/parsers/base_parser.py` - Abstract interface
- `src/parsers/pdf_parser.py` - PDF implementation
- `src/parsers/html_parser.py` - HTML implementation
- `src/parsers/markdown_parser.py` - Markdown implementation
- `src/parsers/parser_factory.py` - Parser selection factory
- `tests/unit/test_parsers.py` - Unit tests (41)
- `tests/integration/test_parsers_integration.py` - Integration tests (7)
