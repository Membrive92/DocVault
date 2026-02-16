# Guia Interna - Milestone 5: Document Ingestion Pipeline

> Guia escrita para desarrolladores con experiencia en Java.
> Cada concepto de Python se compara con su equivalente en Java.

---

## Indice

1. [Que hemos construido](#1-que-hemos-construido)
2. [La metafora del pipeline](#2-la-metafora-del-pipeline)
3. [Configuracion centralizada - config.py](#3-configuracion-centralizada---configpy)
4. [Modelos de datos - dataclasses con defaults](#4-modelos-de-datos---dataclasses-con-defaults)
5. [TextChunker - Dividir texto en trozos](#5-textchunker---dividir-texto-en-trozos)
6. [La estrategia de chunking paso a paso](#6-la-estrategia-de-chunking-paso-a-paso)
7. [IngestionStateManager - Persistencia JSON](#7-ingestionstatemanager---persistencia-json)
8. [IngestionPipeline - El orquestador](#8-ingestionpipeline---el-orquestador)
9. [UUID5 deterministico vs UUID4 aleatorio](#9-uuid5-deterministico-vs-uuid4-aleatorio)
10. [Dependency Injection sin framework](#10-dependency-injection-sin-framework)
11. [File discovery con glob y filtros](#11-file-discovery-con-glob-y-filtros)
12. [Los tests - Mocks y patches](#12-los-tests---mocks-y-patches)
13. [field(default_factory=list) - El truco de las listas mutables](#13-fielddefault_factorylist---el-truco-de-las-listas-mutables)
14. [Flujo completo: de directorio a vectores indexados](#14-flujo-completo-de-directorio-a-vectores-indexados)
15. [Tabla resumen Java vs Python en M5](#15-tabla-resumen-java-vs-python-en-m5)

---

## 1. Que hemos construido

En M2 creamos el servicio de embeddings (texto → vectores). En M3 la base de datos vectorial
(almacenar y buscar). En M4 los parsers (archivos → texto limpio). Pero cada pieza estaba
aislada. No habia forma de decir "procesa todos los PDFs de esta carpeta".

En M5 hemos construido el **pipeline de ingestion** — el programa que conecta todo:

```
Carpeta de documentos
    │
    ▼
 Descubrir archivos (.pdf, .html, .md)
    │
    ▼
 Parsear cada archivo (M4: ParserFactory)
    │
    ▼
 Dividir texto en chunks (~500 tokens)
    │
    ▼
 Generar embeddings por chunk (M2: EmbeddingService)
    │
    ▼
 Guardar vectores en Qdrant (M3: QdrantDatabase)
    │
    ▼
 Registrar estado (JSON: "este archivo ya esta indexado")
```

Piensalo como una **linea de montaje** en una fabrica: cada estacion hace una cosa,
y el pipeline mueve las piezas de una estacion a la siguiente.

---

## 2. La metafora del pipeline

En Java, un pipeline asi se implementaria con Spring Batch o similar. Tendrias:

```java
// Java — Spring Batch (simplificado)
@Bean
public Job ingestionJob(JobBuilderFactory jobs, StepBuilderFactory steps) {
    return jobs.get("ingestionJob")
        .start(discoverFilesStep())
        .next(parseFilesStep())
        .next(chunkTextsStep())
        .next(generateEmbeddingsStep())
        .next(storeVectorsStep())
        .build();
}
```

En Python no usamos ningun framework. Es una clase simple que llama metodos en orden:

```python
class IngestionPipeline:
    def ingest_file(self, file_path):
        parsed_doc = self.parser_factory.parse(file_path)     # 1. Parsear
        chunks = self.chunker.chunk_text(parsed_doc.text)      # 2. Chunk
        embeddings = self.embedding_service.generate_batch_embeddings(chunks)  # 3. Embed
        self.vector_db.insert_vectors(ids, embeddings, metadata)  # 4. Almacenar
        self.state_manager.mark_indexed(file_path, chunk_count)   # 5. Estado
```

Sin decoradores, sin framework, sin XML. Solo llamadas a metodos.

---

## 3. Configuracion centralizada - config.py

Mismo patron que M4 (`src/parsers/config.py`): constantes a nivel de modulo.

```python
# src/ingestion/config.py
CHUNK_SIZE = 500          # Tokens por chunk
CHUNK_OVERLAP = 50        # Tokens de solapamiento entre chunks
MIN_CHUNK_SIZE = 100      # Chunks mas pequenos se descartan
CHARS_PER_TOKEN = 4       # Aproximacion: 4 caracteres ≈ 1 token
BATCH_SIZE = 32           # Cuantos textos embeber a la vez
SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown"}
SKIP_PATTERNS = ["__pycache__", ".git", "node_modules", ".venv", "venv"]
INDEX_STATE_FILE = "data/index_state.json"
```

En Java esto seria un `application.properties` o una clase de constantes:

```java
// Java — clase de constantes
public final class IngestionConfig {
    public static final int CHUNK_SIZE = 500;
    public static final int CHUNK_OVERLAP = 50;
    public static final int MIN_CHUNK_SIZE = 100;
    public static final int CHARS_PER_TOKEN = 4;
    public static final int BATCH_SIZE = 32;
    public static final Set<String> SUPPORTED_EXTENSIONS =
        Set.of(".pdf", ".html", ".htm", ".md", ".markdown");
    public static final List<String> SKIP_PATTERNS =
        List.of("__pycache__", ".git", "node_modules", ".venv", "venv");
    public static final String INDEX_STATE_FILE = "data/index_state.json";

    private IngestionConfig() {} // No instanciable
}
```

### Por que `set` para extensiones?

```python
SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown"}
```

Las llaves `{}` crean un **set**, no un dict. La diferencia:

```python
{".pdf", ".html"}     # set → busqueda O(1)
{".pdf": True}         # dict → clave:valor
[".pdf", ".html"]      # list → busqueda O(n)
```

Usamos set porque despues hacemos `if extension in SUPPORTED_EXTENSIONS` — un set
comprueba en O(1), como un `HashSet` en Java. Una lista seria O(n).

```java
// Java — equivalente
Set.of(".pdf", ".html")       // Set inmutable, busqueda O(1)
List.of(".pdf", ".html")      // Lista, busqueda O(n) con contains()
```

---

## 4. Modelos de datos - dataclasses con defaults

### ChunkMetadata — los datos que guardamos por chunk

```python
@dataclass
class ChunkMetadata:
    chunk_text: str              # El texto del chunk
    source_file: str             # Ruta del archivo original
    chunk_index: int             # Posicion del chunk (0, 1, 2...)
    total_chunks: int            # Total de chunks del documento
    document_title: Optional[str]  # Titulo (puede ser None)
    document_format: str         # "pdf", "html" o "markdown"
    chunked_at: str              # Timestamp ISO
```

En Java:

```java
public record ChunkMetadata(
    String chunkText,
    String sourceFile,
    int chunkIndex,
    int totalChunks,
    String documentTitle,    // puede ser null
    String documentFormat,
    String chunkedAt
) {}
```

### IngestionResult — resultado por archivo

```python
@dataclass
class IngestionResult:
    file_path: str
    chunks_created: int
    status: str                  # "success", "skipped", "failed"
    error: Optional[str] = None  # Solo si status == "failed"
```

El `= None` significa que `error` tiene un valor por defecto. Puedes crear el objeto
sin pasarlo:

```python
# Solo 3 argumentos — error sera None automaticamente
result = IngestionResult(file_path="/doc.pdf", chunks_created=10, status="success")
```

En Java necesitarias un segundo constructor o un builder:

```java
// Java — con record
public record IngestionResult(String filePath, int chunksCreated, String status, String error) {
    // Constructor sin error
    public IngestionResult(String filePath, int chunksCreated, String status) {
        this(filePath, chunksCreated, status, null);
    }
}
```

### IngestionSummary — resumen del directorio

```python
@dataclass
class IngestionSummary:
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    total_chunks: int = 0
    results: list[IngestionResult] = field(default_factory=list)
```

Aqui **todos** los campos tienen defaults. Puedes crear un objeto vacio:

```python
summary = IngestionSummary()  # Todo en 0, results = []
summary.processed += 1        # Modificar directamente
summary.results.append(result) # Anadir a la lista
```

En Java esto seria un builder pattern o un POJO mutable:

```java
// Java — POJO mutable
public class IngestionSummary {
    private int processed = 0;
    private int skipped = 0;
    private int failed = 0;
    private int totalChunks = 0;
    private List<IngestionResult> results = new ArrayList<>();

    public void incrementProcessed() { processed++; }
    public void addResult(IngestionResult r) { results.add(r); }
    // ... getters y setters
}
```

---

## 5. TextChunker - Dividir texto en trozos

### Por que dividir en chunks?

Los modelos de embedding tienen un limite de tokens (~512 para MiniLM). Si pasas un
documento de 5000 palabras, solo usa las primeras ~512. Ademas, para busqueda semantica
quieres vectores que representen **un tema concreto**, no un documento entero.

La solucion: dividir cada documento en chunks de ~500 tokens, y generar un vector por chunk.

### El constructor con validacion

```python
class TextChunker:
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,       # 500 tokens
        chunk_overlap: int = CHUNK_OVERLAP,  # 50 tokens
        min_chunk_size: int = MIN_CHUNK_SIZE,  # 100 tokens
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size cannot be negative")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Pre-calcular valores en caracteres (4 chars ≈ 1 token)
        self._chunk_chars = chunk_size * CHARS_PER_TOKEN       # 2000 chars
        self._overlap_chars = chunk_overlap * CHARS_PER_TOKEN   # 200 chars
        self._min_chars = min_chunk_size * CHARS_PER_TOKEN      # 400 chars
```

### El guion bajo `_chunk_chars`

Los atributos con `_` son **privados por convencion**. No son realmente privados
(puedes acceder a `chunker._chunk_chars`), pero el `_` dice "esto es interno, no lo uses".

En Java seria `private`:

```java
public class TextChunker {
    private final int chunkSize;
    private final int chunkChars;  // ← private real, no por convencion

    public TextChunker(int chunkSize, int chunkOverlap, int minChunkSize) {
        if (chunkOverlap >= chunkSize) {
            throw new IllegalArgumentException("overlap must be < chunk_size");
        }
        this.chunkSize = chunkSize;
        this.chunkChars = chunkSize * CHARS_PER_TOKEN;
    }
}
```

---

## 6. La estrategia de chunking paso a paso

### Ejemplo concreto

Imagina este texto (simplificado):

```
"La inteligencia artificial ha avanzado mucho.

Los modelos de lenguaje pueden generar texto.

La busqueda semantica usa vectores para encontrar documentos similares."
```

**Paso 1: Dividir por parrafos** (`re.split(r"\n\s*\n", text)`)

```python
paragraphs = [
    "La inteligencia artificial ha avanzado mucho.",
    "Los modelos de lenguaje pueden generar texto.",
    "La busqueda semantica usa vectores para encontrar documentos similares.",
]
```

**Paso 2: Acumular hasta el limite**

Si el chunk_size es 200 chars y cada parrafo tiene ~50 chars, los tres caben en un chunk.
Si cada parrafo tuviera 150 chars, el primero y segundo formarian un chunk,
y el tercero empezaria un nuevo chunk.

**Paso 3: Overlap — llevar contexto al siguiente chunk**

Cuando se cierra un chunk, la ultima parte se "copia" al inicio del siguiente:

```
Chunk 1: [Parrafo A] [Parrafo B] [Parrafo C]
Chunk 2: [Parrafo C] [Parrafo D] [Parrafo E]  ← C aparece en ambos
```

Esto garantiza que si una pregunta cruza la frontera entre dos chunks,
al menos uno de ellos tendra el contexto completo.

**Paso 4: Parrafos gigantes → fallback a oraciones**

Si un parrafo tiene mas de 2x el tamano del chunk (ej: un parrafo de 4000 chars
cuando el chunk es 2000), lo dividimos por oraciones en lugar de por parrafos:

```python
if para_len > self._chunk_chars * 2:
    sentence_chunks = self._split_by_sentences(paragraph)
```

### El regex para parrafos

```python
paragraphs = re.split(r"\n\s*\n", text.strip())
```

`\n\s*\n` significa: un salto de linea, cero o mas espacios en blanco, otro salto.
Captura `\n\n`, `\n   \n`, `\n\t\n`, etc.

En Java:

```java
String[] paragraphs = text.strip().split("\\n\\s*\\n");
```

### El regex para oraciones

```python
sentences = re.split(r"(?<=[.!?])\s+", text)
```

`(?<=[.!?])` es un **lookbehind**: "la posicion que viene despues de `.`, `!` o `?`".
`\s+` es uno o mas espacios. Asi divide por oraciones sin perder el punto final.

Ejemplo:

```
"Hola mundo. Esto es un test! Que tal?"
→ ["Hola mundo.", "Esto es un test!", "Que tal?"]
```

En Java:

```java
String[] sentences = text.split("(?<=[.!?])\\s+");
```

### Comparacion completa

| Aspecto | Python (TextChunker) | Java (equivalente) |
|---------|---------------------|-------------------|
| Split por parrafos | `re.split(r"\n\s*\n", text)` | `text.split("\\n\\s*\\n")` |
| Split por oraciones | `re.split(r"(?<=[.!?])\s+", text)` | `text.split("(?<=[.!?])\\s+")` |
| Unir chunks | `"\n\n".join(parts)` | `String.join("\n\n", parts)` |
| Filtrar | `[c for c in chunks if len(c) >= min]` | `chunks.stream().filter(c -> c.length() >= min).toList()` |
| Acumular | `current_parts.append(paragraph)` | `currentParts.add(paragraph)` |

---

## 7. IngestionStateManager - Persistencia JSON

### El problema

Si tienes 1000 archivos y ejecutas el pipeline dos veces, no quieres re-procesar
los 1000. Quieres procesar solo los nuevos o modificados.

### La solucion: un archivo JSON que registra que se ha indexado

```json
{
  "C:/docs/manual.pdf": {
    "indexed_at": "2026-01-15T10:30:00+00:00",
    "mtime": 1705312200.0,
    "chunk_count": 15,
    "metadata": {"title": "Manual", "format": "pdf"}
  },
  "C:/docs/readme.md": {
    "indexed_at": "2026-01-15T10:30:05+00:00",
    "mtime": 1705312205.0,
    "chunk_count": 3,
    "metadata": {"title": "README", "format": "markdown"}
  }
}
```

### mtime — como detectar cambios

`mtime` es el **modification time** del archivo. El sistema operativo lo actualiza
cada vez que el archivo se modifica.

```python
current_mtime = path.stat().st_mtime  # → 1705312200.0 (float, epoch seconds)
```

Si el mtime actual != mtime guardado, el archivo cambio y hay que re-indexar.

En Java:

```java
long mtime = Files.getLastModifiedTime(path).toMillis();
```

### path.resolve() — por que no path.absolute()

Usamos `path.resolve()` como clave del JSON:

```python
key = str(path.resolve())  # → "C:\Users\membr\docs\manual.pdf"
```

`resolve()` resuelve symlinks y rutas relativas, dando la ruta **canonica**.
`absolute()` solo anade el directorio actual pero no resuelve symlinks.

```python
Path("../docs/file.md").resolve()   # → "/home/user/docs/file.md" (canonica)
Path("../docs/file.md").absolute()  # → "/home/user/project/../docs/file.md" (con ..)
```

En Java:

```java
path.toRealPath()    // ← Equivalente a resolve() — resuelve symlinks
path.toAbsolutePath() // ← Equivalente a absolute() — no resuelve symlinks
```

### Lazy import de settings

```python
def __init__(self, state_file=None):
    if state_file is None:
        from config.settings import settings    # ← Import DENTRO del metodo
        self.state_file = settings.get_full_path(Path(INDEX_STATE_FILE))
    else:
        self.state_file = Path(state_file)
```

El import de `settings` esta **dentro del `if`**, no arriba del archivo. Esto se llama
**lazy import** — solo importa cuando realmente se necesita.

Motivo: en los tests pasamos `state_file=tmp_path / "state.json"`, asi que nunca
necesitamos `settings`. Si el import estuviera arriba del archivo, se cargaria
siempre (incluso en tests), lo cual podria causar errores de importacion circular.

En Java no existe este patron porque los imports no ejecutan codigo.
El equivalente mas cercano seria inyectar la ruta por constructor:

```java
public class IngestionStateManager {
    private final Path stateFile;

    // Constructor con inyeccion — evita depender de Settings en tests
    public IngestionStateManager(Path stateFile) {
        this.stateFile = stateFile;
    }

    // Constructor por defecto — usa Settings
    public IngestionStateManager() {
        this(Settings.getInstance().getFullPath("data/index_state.json"));
    }
}
```

---

## 8. IngestionPipeline - El orquestador

### Las 5 etapas de ingest_file

```python
def ingest_file(self, file_path):
    path = Path(file_path)

    # 1. PARSEAR (M4)
    parsed_doc = self.parser_factory.parse(path)

    # 2. DIVIDIR EN CHUNKS
    chunks = self.chunker.chunk_text(parsed_doc.text)
    if not chunks:
        raise ValueError(f"No valid chunks from {path.name}")

    # 3. GENERAR EMBEDDINGS (M2)
    embeddings = self.embedding_service.generate_batch_embeddings(
        chunks, batch_size=BATCH_SIZE
    )

    # 4. CREAR IDs Y METADATA
    now = datetime.now(timezone.utc).isoformat()
    ids = []
    metadata_list = []
    for i, chunk_text in enumerate(chunks):
        chunk_id = str(uuid5(NAMESPACE_URL, f"{path.resolve()}::chunk::{i}"))
        ids.append(chunk_id)
        metadata_list.append({
            "chunk_text": chunk_text,
            "source_file": str(path),
            "chunk_index": i,
            "total_chunks": len(chunks),
            ...
        })

    # 5. INSERTAR EN QDRANT (M3)
    self.vector_db.insert_vectors(ids=ids, vectors=embeddings, metadata=metadata_list)

    # 6. REGISTRAR ESTADO
    self.state_manager.mark_indexed(path, chunk_count=len(chunks))

    return IngestionResult(file_path=str(path), chunks_created=len(chunks), status="success")
```

### ingest_directory — procesamiento por lotes con error handling

```python
def ingest_directory(self, directory, recursive=True, force_reindex=False):
    files = self._discover_files(dir_path, recursive)
    summary = IngestionSummary()

    for file_path in files:
        # Skip si ya esta indexado (a menos que force_reindex=True)
        if not force_reindex and self.state_manager.is_indexed(file_path):
            summary.skipped += 1
            continue

        try:
            result = self.ingest_file(file_path)
            summary.processed += 1
            summary.total_chunks += result.chunks_created
        except Exception as e:
            summary.failed += 1   # ← Un fallo NO cancela el resto

    return summary
```

Lo importante: **un fallo no cancela el lote**. Si tienes 100 archivos y el #47 esta
corrupto, los otros 99 se procesan normalmente. El fallo se registra en `summary.failed`.

En Java tendrias el mismo patron con try/catch dentro del for:

```java
for (Path file : files) {
    try {
        IngestionResult result = ingestFile(file);
        summary.incrementProcessed();
        summary.addChunks(result.chunksCreated());
    } catch (Exception e) {
        summary.incrementFailed();
        logger.error("Failed: {}", file, e);
    }
}
```

---

## 9. UUID5 deterministico vs UUID4 aleatorio

### El problema de UUID4

```python
from uuid import uuid4
id = str(uuid4())  # → "a1b2c3d4-e5f6-..."  ← Diferente cada vez
```

Si ejecutas el pipeline dos veces sobre el mismo archivo, UUID4 genera IDs diferentes.
Qdrant crearia **vectores duplicados** — el mismo chunk guardado dos veces.

### La solucion: UUID5

```python
from uuid import NAMESPACE_URL, uuid5

# Siempre genera el MISMO UUID para la misma entrada
id = str(uuid5(NAMESPACE_URL, f"{path.resolve()}::chunk::0"))
# → "7f3a2b1c-..." ← Siempre el mismo para este archivo y chunk #0
```

UUID5 es un **hash deterministico** (SHA-1) de un namespace + un string.
Misma entrada → mismo UUID → Qdrant hace **upsert** (actualiza en vez de duplicar).

```python
# Nuestro formato: "ruta_absoluta::chunk::indice"
f"{path.resolve()}::chunk::{i}"
# → "C:/docs/manual.pdf::chunk::0"
# → "C:/docs/manual.pdf::chunk::1"
# → "C:/docs/manual.pdf::chunk::2"
```

En Java:

```java
import java.util.UUID;

// UUID5 con namespace URL
UUID id = UUID.nameUUIDFromBytes(
    (path.toRealPath() + "::chunk::" + i).getBytes()
);
// Nota: Java solo tiene UUID.nameUUIDFromBytes() que usa UUID3 (MD5).
// Para UUID5 (SHA-1) necesitas una libreria externa.
```

### Comparacion

| Tipo | Metodo | Determinista | Uso |
|------|--------|-------------|-----|
| UUID4 | Random | No | IDs unicos para cosas nuevas |
| UUID5 | SHA-1(namespace + string) | Si | IDs reproducibles (re-indexacion) |

---

## 10. Dependency Injection sin framework

### El constructor de IngestionPipeline

```python
class IngestionPipeline:
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_db: Optional[QdrantDatabase] = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_db = vector_db or QdrantDatabase()
        self.parser_factory = ParserFactory()
        self.chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.state_manager = IngestionStateManager()
```

### El truco del `or`

```python
self.embedding_service = embedding_service or EmbeddingService()
```

Esto dice: "usa lo que me pasaron, o crea uno nuevo si no me pasaron nada".

- Si `embedding_service` es un objeto real → usa ese
- Si es `None` → `None or EmbeddingService()` → crea uno nuevo

En Java esto seria:

```java
this.embeddingService = embeddingService != null
    ? embeddingService
    : new EmbeddingService();

// O con Optional:
this.embeddingService = Optional.ofNullable(embeddingService)
    .orElseGet(EmbeddingService::new);
```

### Uso en produccion vs tests

```python
# Produccion — crea servicios reales automaticamente
pipeline = IngestionPipeline()

# Tests — inyecta mocks para evitar cargar el modelo ML
pipeline = IngestionPipeline(
    embedding_service=mock_embedding,
    vector_db=mock_db,
)
```

En Java con Spring, esto se haria con `@Autowired` y `@MockBean`:

```java
// Produccion — Spring inyecta automaticamente
@Service
public class IngestionPipeline {
    @Autowired private EmbeddingService embeddingService;
    @Autowired private QdrantDatabase vectorDb;
}

// Test — Spring inyecta mocks
@SpringBootTest
class IngestionPipelineTest {
    @MockBean private EmbeddingService embeddingService;
    @MockBean private QdrantDatabase vectorDb;
    @Autowired private IngestionPipeline pipeline;
}
```

La diferencia: en Python lo hacemos manualmente en el constructor. No hay framework
de inyeccion de dependencias. Es mas simple pero menos automatico.

---

## 11. File discovery con glob y filtros

### Como encontrar archivos

```python
def _discover_files(self, directory: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    supported_files = []

    for file_path in directory.glob(pattern):
        if not file_path.is_file():          # Solo archivos
            continue
        if file_path.name.startswith("."):    # Ignorar ocultos
            continue
        if any(skip in str(file_path) for skip in SKIP_PATTERNS):  # Ignorar patrones
            continue
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:  # Solo extensiones soportadas
            supported_files.append(file_path)

    supported_files.sort()  # Orden determinista
    return supported_files
```

### `**/*` — el patron glob recursivo

```python
directory.glob("**/*")   # Todos los archivos en todos los subdirectorios
directory.glob("*")      # Solo archivos en el directorio actual
```

En Java:

```java
// Recursivo
Files.walk(directory)
    .filter(Files::isRegularFile)
    .collect(Collectors.toList());

// Solo nivel actual
Files.list(directory)
    .filter(Files::isRegularFile)
    .collect(Collectors.toList());
```

### `any()` — comprobar si alguno cumple

```python
if any(skip in str(file_path) for skip in SKIP_PATTERNS):
    continue
```

`any()` devuelve `True` si **al menos un elemento** de la secuencia es `True`.
Aqui comprueba si alguno de los patrones (`__pycache__`, `.git`, etc.) aparece
en la ruta del archivo.

En Java:

```java
if (SKIP_PATTERNS.stream().anyMatch(skip -> filePath.toString().contains(skip))) {
    continue;
}
```

### Por que `sort()`?

```python
supported_files.sort()
```

Sin ordenar, el orden depende del sistema operativo (Windows vs Linux).
Ordenar garantiza que el pipeline procesa archivos **siempre en el mismo orden**,
lo cual facilita debugging y hace los tests deterministas.

---

## 12. Los tests - Mocks y patches

### Unit tests vs integration tests

- **Unit tests** (30): usan **mocks** para EmbeddingService y QdrantDatabase.
  No cargan el modelo ML. Rapidos (~2 segundos).
- **Integration tests** (6): usan servicios **reales**. Cargan el modelo ML.
  Lentos (~30 segundos) pero verifican que todo funciona end-to-end.

### MagicMock — el mock universal de Python

```python
from unittest.mock import MagicMock

mock_embedding = MagicMock()
mock_embedding.generate_batch_embeddings.return_value = [[0.1] * 384 for _ in range(20)]
```

`MagicMock()` crea un objeto que **acepta cualquier llamada** y devuelve lo que
le configuremos. Equivale a Mockito en Java:

```java
// Java — Mockito
EmbeddingService mockEmbedding = mock(EmbeddingService.class);
when(mockEmbedding.generateBatchEmbeddings(any(), anyInt()))
    .thenReturn(List.of(new float[]{0.1f, 0.1f, ...}));
```

### assert_called_once() — verificar que se llamo

```python
pipeline.embedding_service.generate_batch_embeddings.assert_called_once()
pipeline.vector_db.insert_vectors.assert_called_once()
pipeline.state_manager.mark_indexed.assert_called_once()
```

Verifica que cada metodo se llamo exactamente una vez. Si no se llamo,
o se llamo mas de una vez, el test falla.

En Java con Mockito:

```java
verify(embeddingService, times(1)).generateBatchEmbeddings(any(), anyInt());
verify(vectorDb, times(1)).insertVectors(any(), any(), any());
verify(stateManager, times(1)).markIndexed(any(), anyInt());
```

### patch() — reemplazar una clase temporalmente

```python
with patch("src.ingestion.pipeline.IngestionStateManager") as MockState:
    mock_state = MagicMock()
    mock_state.is_indexed.return_value = False
    MockState.return_value = mock_state

    pipe = IngestionPipeline(
        embedding_service=mock_embedding,
        vector_db=mock_db,
    )
    pipe.state_manager = mock_state
```

`patch()` reemplaza una clase (o funcion) **temporalmente** dentro del `with`.
Aqui reemplaza `IngestionStateManager` con un mock para que el constructor
del pipeline no intente cargar el archivo de estado real.

En Java con Mockito esto seria `@MockBean` o `@InjectMocks`:

```java
@ExtendWith(MockitoExtension.class)
class IngestionPipelineTest {
    @Mock private IngestionStateManager stateManager;
    @Mock private EmbeddingService embeddingService;
    @Mock private QdrantDatabase vectorDb;
    @InjectMocks private IngestionPipeline pipeline;
}
```

### Diferencia clave: Python parchea por path, Java por tipo

En Python: `patch("src.ingestion.pipeline.IngestionStateManager")` — tienes que
especificar **donde se importa la clase**, no donde se define.

Si `pipeline.py` hace `from .state_manager import IngestionStateManager`, tienes
que parchear `src.ingestion.pipeline.IngestionStateManager` (donde se usa), no
`src.ingestion.state_manager.IngestionStateManager` (donde se define).

En Java con Mockito no tienes este problema: simplemente inyectas el mock.

---

## 13. field(default_factory=list) - El truco de las listas mutables

### El problema

```python
@dataclass
class IngestionSummary:
    results: list[IngestionResult] = []  # ← ¡PELIGROSO!
```

Esto **no funciona**. Python lanza un error:

```
ValueError: mutable default value is not allowed: use `default_factory`
```

El motivo: si usaras `[]` como default, **todas las instancias compartiran la misma
lista**. Si un objeto anade algo a `results`, todos los objetos lo veran.

### La solucion

```python
from dataclasses import field

@dataclass
class IngestionSummary:
    results: list[IngestionResult] = field(default_factory=list)
```

`field(default_factory=list)` dice: "para cada nueva instancia, llama a `list()`
para crear una lista vacia **nueva**". Cada instancia tiene su propia lista.

### En Java no existe este problema

```java
// Java — cada instancia crea su propia lista en el constructor
public class IngestionSummary {
    private List<IngestionResult> results = new ArrayList<>();  // ← OK en Java
}
```

En Java, `new ArrayList<>()` se ejecuta por cada instancia. No hay riesgo de
compartir la misma lista. Este es un problema exclusivo de Python por como
funcionan los defaults en las definiciones de clase.

### La regla

```python
# ❌ Mutable como default — no funciona
results: list = []
metadata: dict = {}

# ✅ field(default_factory=...) — crea uno nuevo por instancia
results: list = field(default_factory=list)
metadata: dict = field(default_factory=dict)

# ✅ Inmutable como default — funciona directamente
processed: int = 0
status: str = "pending"
error: Optional[str] = None
```

**Regla simple**: si el default es mutable (list, dict, set), usa `field(default_factory=...)`.
Si es inmutable (int, str, None, tuple), ponlo directamente.

---

## 14. Flujo completo: de directorio a vectores indexados

```
     pipeline.ingest_directory("data/documents/")
                        │
                        ▼
     ┌──────────────────────────────────────────────┐
     │  _discover_files()                            │
     │  data/documents/                              │
     │    ├── manual.pdf        ✅ (.pdf soportado)  │
     │    ├── api_docs.html     ✅ (.html soportado) │
     │    ├── README.md         ✅ (.md soportado)   │
     │    ├── notes.txt         ❌ (.txt no soportado)│
     │    └── .gitkeep          ❌ (oculto)          │
     └──────────────────────────────────────────────┘
                        │
             3 archivos a procesar
                        │
     ┌──────────────────▼──────────────────┐
     │  Para cada archivo:                  │
     │                                      │
     │  state_manager.is_indexed(file)?     │
     │       │              │               │
     │      Si             No              │
     │       │              │               │
     │    skip         ingest_file()        │
     │                      │               │
     │        ┌─────────────▼─────────────┐ │
     │        │ 1. ParserFactory.parse()  │ │
     │        │    → ParsedDocument       │ │
     │        │                           │ │
     │        │ 2. TextChunker.chunk_text()│ │
     │        │    → ["chunk0", "chunk1"] │ │
     │        │                           │ │
     │        │ 3. EmbeddingService       │ │
     │        │    .generate_batch()      │ │
     │        │    → [[0.1, 0.2, ...],   │ │
     │        │        [0.3, 0.4, ...]]  │ │
     │        │                           │ │
     │        │ 4. UUID5 por chunk        │ │
     │        │    "file::chunk::0" → id0 │ │
     │        │    "file::chunk::1" → id1 │ │
     │        │                           │ │
     │        │ 5. Qdrant.insert_vectors()│ │
     │        │    ids + vectors + meta   │ │
     │        │                           │ │
     │        │ 6. state_manager          │ │
     │        │    .mark_indexed()        │ │
     │        └───────────────────────────┘ │
     │                                      │
     │  try/except por archivo              │
     │  Un fallo no cancela el resto        │
     └─────────────────────────────────────┘
                        │
                        ▼
     IngestionSummary(processed=3, skipped=0, failed=0, total_chunks=25)
```

### Ejemplo de uso real

```python
from src.ingestion import IngestionPipeline

# Produccion — crea todo automaticamente
pipeline = IngestionPipeline()

# Ingestion de un solo archivo
result = pipeline.ingest_file("data/documents/manual.pdf")
print(f"Chunks: {result.chunks_created}")  # → "Chunks: 15"

# Ingestion de un directorio completo
summary = pipeline.ingest_directory("data/documents/", recursive=True)
print(f"Procesados: {summary.processed}")   # → "Procesados: 10"
print(f"Saltados: {summary.skipped}")       # → "Saltados: 0"
print(f"Fallidos: {summary.failed}")        # → "Fallidos: 1"
print(f"Total chunks: {summary.total_chunks}")  # → "Total chunks: 85"

# Segunda ejecucion — solo procesa archivos nuevos/modificados
summary2 = pipeline.ingest_directory("data/documents/")
print(f"Procesados: {summary2.processed}")  # → "Procesados: 0"
print(f"Saltados: {summary2.skipped}")      # → "Saltados: 10"

# Forzar re-indexacion
summary3 = pipeline.ingest_directory("data/documents/", force_reindex=True)
print(f"Procesados: {summary3.processed}")  # → "Procesados: 10"
```

---

## 15. Tabla resumen Java vs Python en M5

| Concepto | Java | Python (nuestro proyecto) |
|----------|------|---------------------------|
| Constantes | `public static final int CHUNK_SIZE = 500;` | `CHUNK_SIZE = 500` (nivel de modulo) |
| Set literal | `Set.of(".pdf", ".html")` | `{".pdf", ".html"}` |
| Default factory | No necesario (cada `new` es independiente) | `field(default_factory=list)` |
| Optional con default | Constructor overloading o Builder | `param: Optional[X] = None` |
| `x or default` | `x != null ? x : new Default()` | `x or Default()` |
| UUID5 deterministico | Libreria externa | `uuid5(NAMESPACE_URL, string)` (stdlib) |
| UUID4 aleatorio | `UUID.randomUUID()` | `uuid4()` |
| Glob de archivos | `Files.walk(dir).filter(...)` | `dir.glob("**/*")` |
| Comprobar alguno | `stream().anyMatch(...)` | `any(... for ... in ...)` |
| Ordenar in-place | `list.sort()` | `list.sort()` (identico) |
| Mock universal | `Mockito.mock(Class.class)` | `MagicMock()` |
| Verificar llamada | `verify(mock).method()` | `mock.method.assert_called_once()` |
| Parchear clase | `@MockBean` / `@InjectMocks` | `patch("module.path.ClassName")` |
| mtime del archivo | `Files.getLastModifiedTime(path)` | `path.stat().st_mtime` |
| Ruta canonica | `path.toRealPath()` | `path.resolve()` |
| Lazy import | No aplica (imports son declarativos) | `from x import y` dentro de un metodo |
| JSON read/write | Jackson/Gson `ObjectMapper` | `json.load()` / `json.dump()` (stdlib) |
| Error handling en lote | `try/catch` en el for | `try/except` en el for (identico) |
| DI sin framework | Constructor con nullables + ternary | Constructor con `Optional` + `or` |
| DI con framework | Spring `@Autowired` | No usamos framework |
| Pipeline orchestration | Spring Batch `Job` + `Step` | Clase simple con metodos en secuencia |
| Regex lookbehind | `Pattern.compile("(?<=[.!?])\\s+")` | `re.split(r"(?<=[.!?])\s+", text)` |

---

## Resumen final

M5 ha conectado todos los componentes anteriores en un pipeline funcional:

1. **TextChunker** → Divide texto en trozos de ~500 tokens con overlap (sin equivalente directo en Java standard)
2. **IngestionStateManager** → JSON simple para trackear archivos indexados (como una base de datos minimalista)
3. **IngestionPipeline** → Orquestador que llama M4→chunk→M2→M3→estado (como un Spring Batch sin Spring)
4. **ChunkMetadata** → Metadata por vector en Qdrant (como un `record` de Java)
5. **IngestionResult/Summary** → Resultados tipados con defaults (como POJOs mutables con Builder)

Con M5 completado, el sistema RAG puede:
- **M2**: Texto → vectores
- **M3**: Guardar y buscar vectores
- **M4**: Archivos → texto limpio
- **M5**: Ejecutar todo automaticamente sobre carpetas enteras

Lo que falta (M6): la capa de LLM — conectar con Ollama/OpenAI/Anthropic para que
el sistema pueda **responder preguntas** usando los documentos indexados.
