# Guia Interna - Milestone 4: Document Parsers (PDF, HTML, Markdown)

> Guia escrita para desarrolladores con experiencia en Java.
> Cada concepto de Python se compara con su equivalente en Java.

---

## Indice

1. [Que hemos construido](#1-que-hemos-construido)
2. [Arquitectura general - Mismo patron que M3](#2-arquitectura-general---mismo-patron-que-m3)
3. [ParsedDocument - El DTO de salida](#3-parseddocument---el-dto-de-salida)
4. [@dataclass vs Java record/POJO](#4-dataclass-vs-java-recordpojo)
5. [DocumentParser - La interfaz abstracta](#5-documentparser---la-interfaz-abstracta)
6. [PDFParser - Extraccion de texto con pypdf](#6-pdfparser---extraccion-de-texto-con-pypdf)
7. [HTMLParser - Limpieza de boilerplate con BeautifulSoup](#7-htmlparser---limpieza-de-boilerplate-con-beautifulsoup)
8. [MarkdownParser - Frontmatter y regex](#8-markdownparser---frontmatter-y-regex)
9. [ParserFactory - El Factory Pattern](#9-parserfactory---el-factory-pattern)
10. [El manejo de excepciones en los parsers](#10-el-manejo-de-excepciones-en-los-parsers)
11. [Los tests - tmp_path y archivos temporales](#11-los-tests---tmp_path-y-archivos-temporales)
12. [Flujo completo: de archivo a ParsedDocument](#12-flujo-completo-de-archivo-a-parseddocument)
13. [Tabla resumen Java vs Python en M4](#13-tabla-resumen-java-vs-python-en-m4)

---

## 1. Que hemos construido

En M2 creamos el servicio que convierte texto en vectores. En M3 creamos la base de datos
donde guardar esos vectores. Pero... de donde viene el texto?

En M4 hemos construido los **parsers** — programas que leen archivos (PDF, HTML, Markdown)
y extraen el texto limpio. Piensalo como un "lector universal" que abre cualquier formato
y te devuelve solo el contenido util, sin basura (menus, scripts, navegacion, etc.).

Es como si en una empresa te dieran documentos en Word, PDF y paginas web, y tu trabajo
fuera copiar solo el texto relevante a un formato estandar. Eso es lo que hacen los parsers.

La estructura es:

```
Archivo PDF  ──→  PDFParser      ──→  ParsedDocument (texto + metadata)
Archivo HTML ──→  HTMLParser     ──→  ParsedDocument (texto + metadata)
Archivo .md  ──→  MarkdownParser ──→  ParsedDocument (texto + metadata)
```

Todos devuelven el mismo formato de salida (`ParsedDocument`), sin importar el formato
de entrada. Esto es el **Strategy Pattern** — la misma interfaz, diferentes implementaciones.

---

## 2. Arquitectura general - Mismo patron que M3

En M3 usamos `VectorDatabase` (ABC) con `QdrantDatabase` (implementacion).
En M4 usamos exactamente el mismo patron:

| M3 (Vector DB) | M4 (Parsers) |
|----------------|-------------|
| `VectorDatabase` (ABC) | `DocumentParser` (ABC) |
| `QdrantDatabase` (impl) | `PDFParser`, `HTMLParser`, `MarkdownParser` (impls) |
| `get_collection_info()` | `parse()`, `can_parse()` |
| No hay factory | `ParserFactory` (nuevo) |

La diferencia es que en M4 tenemos **tres** implementaciones en vez de una, asi que
hemos anadido un **Factory** para seleccionar automaticamente el parser correcto.

En Java seria:

```java
// M3
interface VectorDatabase { ... }
class QdrantDatabase implements VectorDatabase { ... }

// M4
interface DocumentParser { ... }
class PDFParser implements DocumentParser { ... }
class HTMLParser implements DocumentParser { ... }
class MarkdownParser implements DocumentParser { ... }
class ParserFactory {
    DocumentParser getParser(Path filePath) { ... }
}
```

---

## 3. ParsedDocument - El DTO de salida

Todos los parsers devuelven un `ParsedDocument`. Es como un **DTO** (Data Transfer Object)
en Java — una clase que solo contiene datos, sin logica de negocio.

```python
@dataclass
class ParsedDocument:
    # Campos obligatorios
    text: str              # El texto extraido
    source_path: str       # Ruta del archivo original
    format: str            # "pdf", "html" o "markdown"
    extracted_at: str      # Timestamp ISO
    parser_version: str    # Version del parser

    # Campos opcionales
    title: Optional[str] = None
    author: Optional[str] = None
    page_count: Optional[int] = None
```

En Java seria:

```java
public record ParsedDocument(
    String text,
    String sourcePath,
    String format,
    String extractedAt,
    String parserVersion,
    String title,       // puede ser null
    String author,      // puede ser null
    Integer pageCount   // puede ser null
) {
    // Validacion en el constructor
    public ParsedDocument {
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("Text cannot be empty");
        }
        if (!Set.of("pdf", "html", "markdown").contains(format)) {
            throw new IllegalArgumentException("Invalid format: " + format);
        }
    }

    public int wordCount() {
        return text.split("\\s+").length;
    }

    public int charCount() {
        return text.length();
    }
}
```

---

## 4. @dataclass vs Java record/POJO

`@dataclass` en Python es como `record` en Java 16+ o un POJO con Lombok `@Data`:

### Python @dataclass

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ParsedDocument:
    text: str
    source_path: str
    title: Optional[str] = None

    def __post_init__(self) -> None:
        """Se ejecuta despues del constructor."""
        if not self.text.strip():
            raise ValueError("Text cannot be empty")

    @property
    def word_count(self) -> int:
        return len(self.text.split())
```

### Java record (equivalente)

```java
public record ParsedDocument(
    String text,
    String sourcePath,
    String title
) {
    // "Compact constructor" = __post_init__
    public ParsedDocument {
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("Text cannot be empty");
        }
    }

    public int wordCount() {
        return text.split("\\s+").length;
    }
}
```

### Java POJO con Lombok (alternativa)

```java
@Data  // Genera getters, setters, equals, hashCode, toString
@AllArgsConstructor
public class ParsedDocument {
    private final String text;
    private final String sourcePath;
    private String title;  // nullable

    @PostConstruct
    void validate() {
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("Text cannot be empty");
        }
    }

    public int getWordCount() {
        return text.split("\\s+").length;
    }
}
```

### Comparacion

| Aspecto | Python `@dataclass` | Java `record` | Java POJO + Lombok |
|---------|--------------------|--------------|--------------------|
| Constructor | Auto-generado | Auto-generado | `@AllArgsConstructor` |
| Getters | Acceso directo `doc.text` | `doc.text()` | `doc.getText()` |
| Setters | Acceso directo `doc.text = "..."` | Inmutable (no hay) | `doc.setText("...")` |
| equals/hashCode | Auto-generado | Auto-generado | `@Data` genera |
| toString | Auto-generado | Auto-generado | `@Data` genera |
| Validacion post-creacion | `__post_init__()` | Compact constructor | `@PostConstruct` |
| Propiedades calculadas | `@property` | Metodo normal | Getter con logica |

### __post_init__ = Compact constructor

El metodo `__post_init__` se ejecuta **automaticamente despues del constructor**.
Es donde ponemos la validacion:

```python
def __post_init__(self) -> None:
    if not self.text or not self.text.strip():
        raise ValueError("Parsed document cannot have empty text")

    if self.format not in ("pdf", "html", "markdown"):
        raise ValueError(f"Invalid format: {self.format}")
```

En Java `record`, esto seria el compact constructor:

```java
public ParsedDocument {  // Sin parentesis = compact constructor
    Objects.requireNonNull(text, "Text is required");
    if (text.isBlank()) throw new IllegalArgumentException("Empty text");
    if (!VALID_FORMATS.contains(format)) throw new IllegalArgumentException("Bad format");
}
```

### @property = Getter con logica

```python
@property
def word_count(self) -> int:
    return len(self.text.split())

# Uso:
doc.word_count  # ← Se llama como atributo, no como metodo
```

En Java:

```java
public int getWordCount() {
    return text.split("\\s+").length;
}

// Uso:
doc.getWordCount()  // ← Se llama como metodo
```

`@property` permite que un metodo se comporte como un atributo — sin parentesis.
En Java no existe esto; siempre necesitas `getXxx()`.

---

## 5. DocumentParser - La interfaz abstracta

Identica al patron de `VectorDatabase` en M3:

```python
class DocumentParser(ABC):
    def __init__(self) -> None:
        self.parser_version = PARSER_VERSION

    @abstractmethod
    def parse(self, file_path: str | Path) -> ParsedDocument:
        pass

    @abstractmethod
    def can_parse(self, file_path: str | Path) -> bool:
        pass

    def _validate_file(self, file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")
```

En Java:

```java
public abstract class DocumentParser {
    protected final String parserVersion = "1.0.0";

    public abstract ParsedDocument parse(Path filePath);
    public abstract boolean canParse(Path filePath);

    protected void validateFile(Path filePath) {
        if (!Files.exists(filePath)) {
            throw new FileNotFoundException("File not found: " + filePath);
        }
        if (!Files.isRegularFile(filePath)) {
            throw new IllegalArgumentException("Not a file: " + filePath);
        }
    }
}
```

**Nota:** `_validate_file` con guion bajo es un metodo **protegido** en Python.
No es privado (seria `__validate_file` con doble guion bajo), es "protegido"
por convencion — como `protected` en Java.

---

## 6. PDFParser - Extraccion de texto con pypdf

### La logica paso a paso

```python
class PDFParser(DocumentParser):
    def can_parse(self, file_path: str | Path) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        self._validate_file(path)                    # 1. Verificar que existe

        reader = PdfReader(path)                     # 2. Abrir el PDF

        if reader.is_encrypted:                      # 3. Verificar encriptacion
            raise RuntimeError(f"PDF is encrypted: {path}")

        text_parts = []
        for page in reader.pages:                    # 4. Iterar paginas
            text = page.extract_text()               # 5. Extraer texto de cada pagina
            if text:
                text_parts.append(text)

        full_text = "\n\n".join(text_parts).strip()  # 6. Unir con doble salto

        if not full_text:                            # 7. Verificar que hay texto
            raise ValueError("No text could be extracted")

        metadata = reader.metadata or {}             # 8. Extraer metadata

        return ParsedDocument(                       # 9. Crear resultado
            text=full_text,
            format="pdf",
            title=metadata.get("/Title"),
            author=metadata.get("/Author"),
            page_count=len(reader.pages),
            ...
        )
```

En Java con Apache PDFBox (la libreria mas comun):

```java
public class PDFParser extends DocumentParser {
    @Override
    public boolean canParse(Path filePath) {
        return filePath.toString().toLowerCase().endsWith(".pdf");
    }

    @Override
    public ParsedDocument parse(Path filePath) {
        validateFile(filePath);

        try (PDDocument document = Loader.loadPDF(filePath.toFile())) {
            if (document.isEncrypted()) {
                throw new RuntimeException("PDF is encrypted");
            }

            PDFTextStripper stripper = new PDFTextStripper();
            String text = stripper.getText(document);

            PDDocumentInformation info = document.getDocumentInformation();

            return new ParsedDocument(
                text.trim(),
                filePath.toString(),
                "pdf",
                Instant.now().toString(),
                parserVersion,
                info.getTitle(),
                info.getAuthor(),
                null, null,
                document.getNumberOfPages()
            );
        } catch (IOException e) {
            throw new RuntimeException("PDF parsing failed", e);
        }
    }
}
```

**Diferencia:** En Java usas `try-with-resources` para cerrar el `PDDocument`.
En Python, `PdfReader` no necesita cerrarse — lee todo al instante.

---

## 7. HTMLParser - Limpieza de boilerplate con BeautifulSoup

El HTMLParser tiene la logica mas interesante: no solo extrae texto, sino que
**limpia el HTML** eliminando menus, scripts, barras laterales, etc.

### El proceso

```
HTML original
    │
    ▼
BeautifulSoup (parsea el DOM)
    │
    ▼
Eliminar tags: <script>, <style>, <nav>, <header>, <footer>...
    │
    ▼
Eliminar clases: .sidebar, .menu, .ad, .cookie-notice...
    │
    ▼
Buscar contenido principal: <main> > <article> > div.content > <body>
    │
    ▼
Extraer texto limpio
    │
    ▼
Filtrar lineas cortas (< 10 chars = ruido)
    │
    ▼
ParsedDocument
```

### El codigo clave

```python
# Parsear HTML
soup = BeautifulSoup(html_content, "lxml")

# Eliminar tags no deseados
for tag in ["script", "style", "nav", "header", "footer"]:
    for element in soup.find_all(tag):
        element.decompose()  # ← Elimina el elemento del DOM

# Eliminar clases no deseadas
for class_name in ["sidebar", "menu", "ad"]:
    for element in soup.find_all(class_=class_name):
        element.decompose()

# Buscar contenido principal (cascada de fallbacks)
main_content = (
    soup.find("main")
    or soup.find("article")
    or soup.find("div", class_="content")
    or soup.body
    or soup
)

# Extraer texto
text = main_content.get_text(separator="\n", strip=True)
```

En Java con Jsoup (la libreria equivalente a BeautifulSoup):

```java
Document doc = Jsoup.parse(htmlContent);

// Eliminar tags
doc.select("script, style, nav, header, footer").remove();

// Eliminar clases
doc.select(".sidebar, .menu, .ad").remove();

// Buscar contenido principal
Element main = doc.selectFirst("main");
if (main == null) main = doc.selectFirst("article");
if (main == null) main = doc.selectFirst("div.content");
if (main == null) main = doc.body();

String text = main.text();
```

### Comparacion BeautifulSoup vs Jsoup

| Operacion | Python (BeautifulSoup) | Java (Jsoup) |
|-----------|----------------------|-------------|
| Parsear | `BeautifulSoup(html, "lxml")` | `Jsoup.parse(html)` |
| Buscar tag | `soup.find("main")` | `doc.selectFirst("main")` |
| Buscar todos | `soup.find_all("script")` | `doc.select("script")` |
| Buscar por clase | `soup.find_all(class_="sidebar")` | `doc.select(".sidebar")` |
| Eliminar | `element.decompose()` | `element.remove()` |
| Extraer texto | `element.get_text(separator="\n")` | `element.text()` |
| Titulo | `soup.find("title").get_text()` | `doc.title()` |

---

## 8. MarkdownParser - Frontmatter y regex

### YAML Frontmatter

Muchos archivos Markdown tienen metadata al inicio entre `---`:

```markdown
---
title: Mi Documento
author: Juan
date: 2026-01-15
tags:
  - python
  - rag
---

# El contenido real empieza aqui

Texto del documento...
```

La libreria `python-frontmatter` separa el YAML del contenido:

```python
import frontmatter

post = frontmatter.loads(content)
post.metadata  # → {"title": "Mi Documento", "author": "Juan", ...}
post.content   # → "# El contenido real empieza aqui\n\nTexto..."
```

En Java no hay equivalente directo. Tendrias que parsear manualmente:

```java
// Java — parsing manual de frontmatter
String[] parts = content.split("---", 3);
if (parts.length >= 3) {
    Map<String, Object> metadata = new Yaml().load(parts[1]);
    String text = parts[2];
}
```

### Limpieza de HTML en Markdown

Markdown puede contener HTML embebido: `<strong>negrita</strong>`.
Lo limpiamos con regex:

```python
text = re.sub(r"<[^>]+>", "", text)
# "Hello <strong>world</strong>" → "Hello world"
```

En Java:

```java
text = text.replaceAll("<[^>]+>", "");
```

### Extraccion de titulo por heading

Si no hay frontmatter, buscamos el primer heading:

```python
heading_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
if heading_match:
    title = heading_match.group(1).strip()
```

En Java:

```java
Pattern pattern = Pattern.compile("^#\\s+(.+)$", Pattern.MULTILINE);
Matcher matcher = pattern.matcher(text);
if (matcher.find()) {
    String title = matcher.group(1).trim();
}
```

---

## 9. ParserFactory - El Factory Pattern

El Factory selecciona automaticamente el parser por extension:

```python
class ParserFactory:
    def __init__(self) -> None:
        self.parsers: list[DocumentParser] = [
            PDFParser(),
            HTMLParser(),
            MarkdownParser(),
        ]

    def get_parser(self, file_path: str | Path) -> Optional[DocumentParser]:
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None

    def parse(self, file_path: str | Path) -> ParsedDocument:
        parser = self.get_parser(file_path)
        if parser is None:
            raise ValueError(f"Unsupported format: {Path(file_path).suffix}")
        return parser.parse(file_path)
```

En Java:

```java
public class ParserFactory {
    private final List<DocumentParser> parsers = List.of(
        new PDFParser(),
        new HTMLParser(),
        new MarkdownParser()
    );

    public Optional<DocumentParser> getParser(Path filePath) {
        return parsers.stream()
            .filter(p -> p.canParse(filePath))
            .findFirst();
    }

    public ParsedDocument parse(Path filePath) {
        return getParser(filePath)
            .orElseThrow(() -> new IllegalArgumentException(
                "Unsupported format: " + filePath))
            .parse(filePath);
    }
}
```

### Por que Factory y no un switch/if?

```python
# ❌ Sin factory — fragil, viola Open/Closed
if extension == ".pdf":
    parser = PDFParser()
elif extension == ".html":
    parser = HTMLParser()
elif extension == ".md":
    parser = MarkdownParser()
```

```python
# ✅ Con factory — abierto a extension
factory = ParserFactory()
parser = factory.get_parser(file_path)  # Automatico
```

Si manana anadimos un parser para `.docx`, solo anadimos una clase nueva
y la registramos en el factory. No tocamos ningun `if/else`.

---

## 10. El manejo de excepciones en los parsers

Cada parser usa un patron de excepciones consistente:

```python
def parse(self, file_path: str | Path) -> ParsedDocument:
    path = Path(file_path)
    self._validate_file(path)          # ← FileNotFoundError, ValueError

    try:
        # ... logica de parsing ...

        if not full_text:
            raise ValueError("No text extracted")  # ← Error de contenido

        return ParsedDocument(...)

    except (ValueError, RuntimeError):
        raise                          # ← Re-lanzar errores conocidos
    except Exception as e:
        raise RuntimeError(...) from e # ← Envolver errores inesperados
```

### El truco de `except (ValueError, RuntimeError): raise`

```python
except (ValueError, RuntimeError):
    raise        # ← Los deja pasar sin modificar
except Exception as e:
    raise RuntimeError(f"Parsing failed: {e}") from e  # ← Envuelve el resto
```

Esto evita que un `ValueError` (que nosotros lanzamos intencionalmente) se envuelva
en un `RuntimeError`. Sin este bloque, el `except Exception` lo capturaria todo.

En Java es lo mismo:

```java
try {
    // ...
} catch (IllegalArgumentException | RuntimeException e) {
    throw e;  // Re-lanzar sin envolver
} catch (Exception e) {
    throw new RuntimeException("Parsing failed", e);
}
```

---

## 11. Los tests - tmp_path y archivos temporales

Los tests de parsers necesitan **archivos reales** para parsear. Usamos `tmp_path`
de pytest para crear archivos temporales que se limpian automaticamente.

### tmp_path = @TempDir de JUnit

```python
# Python — pytest crea un directorio temporal automaticamente
def test_parse_html(self, parser: HTMLParser, tmp_path: Path) -> None:
    html_path = tmp_path / "test.html"
    html_path.write_text("<html><body><p>Hello</p></body></html>")

    result = parser.parse(html_path)
    assert "Hello" in result.text
```

```java
// Java — JUnit 5
@Test
void testParseHtml(@TempDir Path tempDir) throws Exception {
    Path htmlPath = tempDir.resolve("test.html");
    Files.writeString(htmlPath, "<html><body><p>Hello</p></body></html>");

    ParsedDocument result = parser.parse(htmlPath);
    assertTrue(result.getText().contains("Hello"));
}
```

### Crear PDFs para tests

No podemos simplemente escribir texto en un `.pdf` — los PDF son binarios.
Creamos PDFs validos con bytes crudos:

```python
def _create_text_pdf(path: Path) -> Path:
    """Crear un PDF minimo con texto extraible."""
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        # ... estructura PDF minima ...
        b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\n"
        # ... xref table y trailer ...
    )
    pdf_path = path / "test.pdf"
    pdf_path.write_bytes(pdf_content)
    return pdf_path
```

En Java usarias `iText` o `PDFBox` para generar el PDF de test:

```java
PDDocument doc = new PDDocument();
PDPage page = new PDPage();
doc.addPage(page);
PDPageContentStream stream = new PDPageContentStream(doc, page);
stream.beginText();
stream.setFont(PDType1Font.HELVETICA, 12);
stream.newLineAtOffset(100, 700);
stream.showText("Hello World");
stream.endText();
stream.close();
doc.save(tempDir.resolve("test.pdf").toFile());
doc.close();
```

### Helpers para crear archivos de test

Definimos funciones helper a nivel de modulo (no en la clase de test):

```python
def _create_html(path: Path, filename: str = "test.html") -> Path:
    """Crear un HTML de prueba."""
    html_content = """<!DOCTYPE html>
<html><head><title>Test</title></head>
<body><main><p>Content here.</p></main></body></html>"""

    html_path = path / filename
    html_path.write_text(html_content, encoding="utf-8")
    return html_path
```

En Java, estas serian metodos `static` en una clase de utilidad:

```java
private static Path createHtml(Path dir, String filename) throws IOException {
    String html = "<!DOCTYPE html>...";
    Path path = dir.resolve(filename);
    Files.writeString(path, html);
    return path;
}
```

---

## 12. Flujo completo: de archivo a ParsedDocument

```
                     ┌────────────────────┐
                     │  ParserFactory     │
     archivo.pdf ──→ │                    │ ──→ PDFParser.parse()
     pagina.html ──→ │  get_parser()      │ ──→ HTMLParser.parse()
     README.md   ──→ │  selecciona por    │ ──→ MarkdownParser.parse()
     doc.docx    ──→ │  extension         │ ──→ ValueError (no soportado)
                     └────────────────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │  ParsedDocument    │
                     │                    │
                     │  text: "..."       │
                     │  title: "..."      │
                     │  format: "pdf"     │
                     │  word_count: 1500  │
                     │  page_count: 5     │
                     └────────────────────┘
                                │
                                ▼ (M5 usara esto)
                     ┌────────────────────┐
                     │  Chunking          │
                     │  (dividir en       │
                     │   trozos de ~500   │
                     │   tokens)          │
                     └────────────────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │  EmbeddingService  │
                     │  (M2 - vectores)   │
                     └────────────────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │  QdrantDatabase    │
                     │  (M3 - almacenar)  │
                     └────────────────────┘
```

### Ejemplo de uso real

```python
from src.parsers import ParserFactory

factory = ParserFactory()

# Parsear diferentes formatos — mismo resultado
pdf_doc = factory.parse("data/documents/manual.pdf")
html_doc = factory.parse("data/documents/api_docs.html")
md_doc = factory.parse("data/documents/README.md")

# Todos son ParsedDocument con la misma interfaz
for doc in [pdf_doc, html_doc, md_doc]:
    print(f"Format: {doc.format}")
    print(f"Title: {doc.title}")
    print(f"Words: {doc.word_count}")
    print(f"First 100 chars: {doc.text[:100]}")
    print()
```

---

## 13. Tabla resumen Java vs Python en M4

| Concepto | Java | Python (nuestro proyecto) |
|----------|------|---------------------------|
| DTO / Data class | `record` (Java 16+) o POJO + Lombok | `@dataclass` |
| Validacion post-constructor | Compact constructor | `__post_init__()` |
| Propiedad calculada | `getWordCount()` | `@property word_count` |
| Interfaz abstracta | `abstract class` / `interface` | `ABC` + `@abstractmethod` |
| Metodo protegido | `protected void validate()` | `def _validate_file()` (con `_`) |
| Factory Pattern | Clase con `List<Parser>` + `stream().filter()` | Clase con `list[Parser]` + `for` loop |
| Leer PDF | Apache PDFBox (`Loader.loadPDF`) | pypdf (`PdfReader`) |
| Leer HTML | Jsoup (`Jsoup.parse()`, `doc.select()`) | BeautifulSoup (`soup.find_all()`, `decompose()`) |
| Leer Markdown frontmatter | Manual (`split("---")` + SnakeYAML) | `python-frontmatter` (una linea) |
| Regex | `Pattern.compile()` + `Matcher` | `re.search()` / `re.sub()` |
| Archivos temporales en tests | `@TempDir Path tempDir` | `tmp_path: Path` fixture |
| Escribir archivo de test | `Files.writeString(path, content)` | `path.write_text(content)` |
| Crear PDF de test | PDFBox `PDDocument` + `PDPageContentStream` | Bytes crudos de PDF minimo |
| Manejar Optional | `Optional<T>` + `orElseThrow()` | `Optional[T]` = `None` + `if/raise` |
| Re-lanzar excepciones | `catch (E e) { throw e; }` | `except E: raise` |
| Envolver excepciones | `throw new RE("msg", e)` | `raise RE("msg") from e` |

---

## Resumen final

M4 ha anadido la capa de **entrada de datos** al sistema RAG:

1. **ParsedDocument** → El formato estandar de salida (como un `record` de Java)
2. **DocumentParser** → La interfaz abstracta (como un `interface` de Java)
3. **PDFParser** → Extrae texto con pypdf (equivalente a Apache PDFBox)
4. **HTMLParser** → Limpia boilerplate con BeautifulSoup (equivalente a Jsoup)
5. **MarkdownParser** → Extrae frontmatter (no hay equivalente directo en Java)
6. **ParserFactory** → Seleccion automatica por extension (Factory Pattern clasico)

Con M4 completado, el pipeline RAG tiene:
- **M2**: Convertir texto → vectores
- **M3**: Guardar y buscar vectores
- **M4**: Leer archivos → texto limpio

Lo que falta (M5): conectar todo — leer archivos, parsearlos, dividirlos en chunks,
generar embeddings, y guardarlos en Qdrant. Eso es el **ingestion pipeline**.
