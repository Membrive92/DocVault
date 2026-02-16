# Guia Interna - Milestone 7: Complete RAG Pipeline

> Guia escrita para desarrolladores con experiencia en Java.
> Cada concepto de Python se compara con su equivalente en Java.

---

## Indice

1. [Que hemos construido](#1-que-hemos-construido)
2. [RAGPipeline - el orquestador central](#2-ragpipeline---el-orquestador-central)
3. [Dependency Injection sin framework](#3-dependency-injection-sin-framework)
4. [dataclass vs Pydantic BaseModel](#4-dataclass-vs-pydantic-basemodel)
5. [El flujo query() paso a paso](#5-el-flujo-query-paso-a-paso)
6. [_build_context - formatear contexto para el LLM](#6-_build_context---formatear-contexto-para-el-llm)
7. [FastAPI - el servidor REST](#7-fastapi---el-servidor-rest)
8. [Lifespan y el patron global mutable](#8-lifespan-y-el-patron-global-mutable)
9. [StreamingResponse - respuestas en tiempo real](#9-streamingresponse---respuestas-en-tiempo-real)
10. [Interactive CLI con rich](#10-interactive-cli-con-rich)
11. [Lazy initialization en el CLI](#11-lazy-initialization-en-el-cli)
12. [Tests de API - parchear el constructor, no la variable](#12-tests-de-api---parchear-el-constructor-no-la-variable)
13. [Integration tests - real embeddings + mock LLM](#13-integration-tests---real-embeddings--mock-llm)
14. [time.perf_counter vs time.time](#14-timeperfcounter-vs-timetime)
15. [Tabla resumen Java vs Python en M7](#15-tabla-resumen-java-vs-python-en-m7)

---

## 1. Que hemos construido

En milestones anteriores construimos todas las piezas individuales:
- **M2**: Texto → vectores (EmbeddingService)
- **M3**: Almacenar y buscar vectores (QdrantDatabase)
- **M4**: Archivos → texto limpio (parsers)
- **M5**: Pipeline completo de ingestion (chunking + indexing)
- **M6**: Capa flexible de LLM (Strategy Pattern)

En M7 hemos **integrado todo** en un sistema completo:

```
Pregunta del usuario → embedding → busqueda → contexto → LLM → respuesta con fuentes
```

Ademas creamos dos interfaces para que el usuario interactue:
- **REST API** con FastAPI (para integraciones programaticas)
- **CLI interactivo** con rich (para uso directo en terminal)

Es como si en Java hubieras construido los Services por separado y ahora
creas el Controller (API) y un Main con Scanner (CLI) que los usan.

---

## 2. RAGPipeline - el orquestador central

El `RAGPipeline` es la clase que conecta todo. Recibe tres servicios por
inyeccion de dependencias y los orquesta en un flujo de 5 pasos.

```python
class RAGPipeline:
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_db: Optional[QdrantDatabase] = None,
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[RAGConfig] = None,
    ) -> None:
        self.config = config or RAGConfig()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_db = vector_db or QdrantDatabase()
        self.llm = llm_provider or LLMProviderFactory.create_provider()
```

En Java seria algo como:

```java
public class RAGPipeline {
    private final EmbeddingService embeddingService;
    private final QdrantDatabase vectorDb;
    private final LLMProvider llm;
    private final RAGConfig config;

    // Constructor con DI (Spring lo haria con @Autowired)
    public RAGPipeline(
        EmbeddingService embeddingService,
        QdrantDatabase vectorDb,
        LLMProvider llmProvider,
        RAGConfig config
    ) {
        this.embeddingService = embeddingService != null ? embeddingService : new EmbeddingService();
        this.vectorDb = vectorDb != null ? vectorDb : new QdrantDatabase();
        this.llm = llmProvider != null ? llmProvider : LLMProviderFactory.createProvider();
        this.config = config != null ? config : new RAGConfig();
    }
}
```

La diferencia clave: en Python usamos `Optional` con default `None` y el patron
`x or default()`. En Java usarias `@Nullable` o sobrecarga de constructores.

---

## 3. Dependency Injection sin framework

En Java normalmente usas Spring Boot con `@Autowired`, `@Service`, `@Component`.
En Python no hay framework de DI — lo hacemos con parametros opcionales:

```python
# Produccion — crea todo con defaults (lee .env)
pipeline = RAGPipeline()

# Testing — inyecta mocks
pipeline = RAGPipeline(
    embedding_service=mock_embedding,
    vector_db=mock_db,
    llm_provider=mock_llm,
)
```

El patron `self.x = param or Default()` es el equivalente Python de:
```java
// Java — constructor con Builder pattern
RAGPipeline.builder()
    .embeddingService(mockEmbedding)
    .vectorDb(mockDb)
    .build();
```

La ventaja: no necesitas Spring, no necesitas XML, no necesitas anotaciones.
El constructor hace todo. Y los tests son triviales — pasa mocks directamente.

---

## 4. dataclass vs Pydantic BaseModel

En M7 usamos **ambos**, pero para cosas diferentes:

**dataclass** (modelos internos del RAG):
```python
@dataclass
class Source:
    chunk_text: str
    source_file: str
    chunk_index: int
    similarity_score: float
```

**Pydantic BaseModel** (modelos de la API):
```python
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
```

¿Por que la diferencia?

| Aspecto | dataclass | Pydantic BaseModel |
|---------|-----------|-------------------|
| Validacion | No (acepta cualquier cosa) | Si (min_length, ge, le) |
| Serializacion JSON | Manual | Automatica |
| Uso | Modelos internos | API request/response |
| Performance | Mas rapida | Mas lenta (valida) |
| Equivalente Java | POJO / record | DTO con Bean Validation |

En Java:
```java
// Interno (como dataclass)
public record Source(String chunkText, String sourceFile, int chunkIndex, double score) {}

// API (como BaseModel)
public class QueryRequest {
    @NotBlank String query;
    @Min(1) @Max(20) Integer topK;
}
```

---

## 5. El flujo query() paso a paso

```python
def query(self, query_text, top_k=None, temperature=None, max_tokens=None, streaming=False):
    # 1. Validar
    if not query_text or not query_text.strip():
        raise ValueError("Query text cannot be empty")

    # 2. Usar parametros o defaults del config
    effective_top_k = top_k if top_k is not None else self.config.top_k

    # 3. Embed
    query_vector = self.embedding_service.generate_embedding(query_text)

    # 4. Search
    results = self.vector_db.search_similar(
        query_vector=query_vector,
        limit=min(effective_top_k, MAX_CONTEXT_CHUNKS),
        score_threshold=self.config.min_similarity,
    )

    # 5. No results → mensaje por defecto
    if not results:
        return RAGResponse(answer=NO_RESULTS_MESSAGE, sources=[], ...)

    # 6. Build sources + context
    sources = self._build_sources(results)
    context = self._build_context(sources)

    # 7. Generate (sync o streaming)
    if streaming:
        return self.llm.generate_stream(prompt=query_text, context=context, ...)

    answer = self.llm.generate(prompt=query_text, context=context, ...)
    return RAGResponse(answer=answer, sources=sources, ...)
```

**Detalle importante:** los parametros `top_k`, `temperature`, `max_tokens` se pasan
por llamada, no se mutan en `self.config`. Esto hace el pipeline **thread-safe**.

En la spec original la API hacia `rag_pipeline.config.top_k = request.top_k` — eso
es peligroso si hay multiples requests concurrentes.

---

## 6. _build_context - formatear contexto para el LLM

El LLM necesita contexto en texto plano. Construimos un string con headers:

```python
def _build_context(self, sources: list[Source]) -> str:
    chunks = []
    for i, source in enumerate(sources, 1):
        header = f"[Source {i}] {source.source_file} (similarity: {source.similarity_score:.2f})"
        chunks.append(f"{header}\n{source.chunk_text}")
    return CONTEXT_SEPARATOR.join(chunks)  # "\n\n"
```

Resultado:
```
[Source 1] docker_guide.md (similarity: 0.92)
Docker is a containerization platform...

[Source 2] docker_guide.md (similarity: 0.85)
To install Docker, download from docker.com...
```

Este contexto se pasa a `llm.generate(prompt=query, context=context)`, que usa
el template RAG de M6 (`format_prompt_with_context`).

---

## 7. FastAPI - el servidor REST

FastAPI es el equivalente Python de Spring Boot para APIs REST.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="DocVault API", version="1.0.0", lifespan=lifespan)

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    response = rag_pipeline.query(query_text=request.query, ...)
    return QueryResponse(answer=response.answer, ...)
```

Equivalente en Spring Boot:

```java
@RestController
public class QueryController {
    @Autowired private RAGPipeline ragPipeline;

    @PostMapping("/query")
    public QueryResponse query(@RequestBody @Valid QueryRequest request) {
        RAGResponse response = ragPipeline.query(request.getQuery(), ...);
        return new QueryResponse(response.getAnswer(), ...);
    }
}
```

Diferencias clave:

| Aspecto | FastAPI (Python) | Spring Boot (Java) |
|---------|-----------------|-------------------|
| Validacion | Pydantic `Field(ge=1, le=20)` | `@Min(1) @Max(20)` |
| Serializacion | Automatica (Pydantic) | Automatica (Jackson) |
| Anotacion ruta | `@app.post("/query")` | `@PostMapping("/query")` |
| DI | Global variable + lifespan | `@Autowired` |
| Async | Soportado pero no usado aqui | Spring WebFlux |

Usamos `def` (sync) en vez de `async def` porque todas las operaciones
subyacentes (embedding, Qdrant, LLM) son sincronas. Usar `async def` sin
`await` no aporta nada y puede confundir.

---

## 8. Lifespan y el patron global mutable

FastAPI usa `lifespan` para inicializar recursos al arrancar:

```python
rag_pipeline: Optional[RAGPipeline] = None  # global mutable

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline
    try:
        rag_pipeline = RAGPipeline()
    except Exception as e:
        rag_pipeline = None  # graceful degradation
    yield
    rag_pipeline = None  # cleanup
```

En Spring Boot esto seria `@PostConstruct` o `@Bean`:

```java
@Configuration
public class AppConfig {
    @Bean
    public RAGPipeline ragPipeline() {
        return new RAGPipeline();
    }
}
```

El `global` en Python es necesario porque no hay container de DI.
Es el equivalente de un `static` field en Java — no es elegante,
pero es el patron estandar de FastAPI para estado de aplicacion.

El `try/except` es nuestra adicion: si el pipeline falla al iniciar
(por ejemplo, no hay modelo de embeddings descargado), el servidor
arranca igual y responde 503 en vez de crashear.

---

## 9. StreamingResponse - respuestas en tiempo real

Para streaming usamos `StreamingResponse` de FastAPI:

```python
@app.post("/query/stream")
def query_stream(request: QueryRequest):
    stream = rag_pipeline.query(query_text=request.query, streaming=True)
    return StreamingResponse(stream, media_type="text/plain")
```

`stream` es un `Iterator[str]` que viene de `llm.generate_stream()`.
FastAPI lo consume chunk a chunk y los envia al cliente conforme llegan.

En Spring Boot seria `Flux<String>` con WebFlux:

```java
@PostMapping(value = "/query/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
public Flux<String> queryStream(@RequestBody QueryRequest request) {
    return ragPipeline.queryStream(request.getQuery());
}
```

La diferencia: en Python usamos generators (`yield`), en Java usamos
reactive streams (`Flux`). Los generators de Python son mucho mas simples.

---

## 10. Interactive CLI con rich

`rich` es una libreria de Python para formateo de terminal. No tiene
equivalente directo en Java — lo mas cercano seria JLine + Jansi.

```python
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

console = Console()

# Panel con borde verde para la respuesta
console.print(Panel(Markdown(response.answer), title="Answer", border_style="green"))

# Colores segun score
score_color = "green" if source.similarity_score > 0.7 else "yellow"
console.print(f"  1. [dim]{source.source_file}[/dim] [{score_color}](score: 0.92)[/{score_color}]")

# Input del usuario
user_input = Prompt.ask("[bold cyan]>[/bold cyan]")
```

El REPL es un bucle `while True` con manejo de `KeyboardInterrupt` (Ctrl+C)
y `EOFError` (Ctrl+D):

```python
while True:
    try:
        user_input = Prompt.ask(">")
        if user_input.startswith("/"):
            should_exit = self.handle_command(user_input)
            if should_exit:
                break
        else:
            self.execute_query(user_input)
    except (KeyboardInterrupt, EOFError):
        break
```

En Java seria un `Scanner` con `System.in` y `try/catch` para
`NoSuchElementException`.

---

## 11. Lazy initialization en el CLI

El CLI usa un patron interesante — no crea el pipeline en el constructor:

```python
class InteractiveCLI:
    def __init__(self, rag_pipeline=None):
        self.pipeline = rag_pipeline  # puede ser None

    def _ensure_pipeline(self):
        if self.pipeline is None:
            self.pipeline = RAGPipeline()

    def print_banner(self):
        self._ensure_pipeline()  # crea el pipeline aqui, no antes
        ...
```

¿Por que lazy? Porque crear el pipeline carga el modelo de embeddings (~120MB),
lo cual tarda varios segundos. Si lo hicieramos en `__init__`, el usuario veria
un delay antes de cualquier output. Con lazy initialization, primero mostramos
"Initializing..." y luego cargamos.

En Java seria como usar `@Lazy` en Spring o `Supplier<RAGPipeline>`.

---

## 12. Tests de API - parchear el constructor, no la variable

Este fue el problema mas interesante de M7.

**Primer intento (fallido):**
```python
# Parchear la variable global
with patch("src.api.server.rag_pipeline", mock_pipeline):
    client = TestClient(app)  # ← el lifespan SOBREESCRIBE nuestro patch!
```

El `TestClient` ejecuta el `lifespan`, que crea un nuevo `RAGPipeline()`
y asigna el resultado a `rag_pipeline`, pisando nuestro mock.

**Solucion: parchear el constructor:**
```python
# Parchear la CLASE, no la variable
with patch("src.api.server.RAGPipeline", return_value=mock_pipeline):
    client = TestClient(app)  # lifespan llama RAGPipeline() → devuelve nuestro mock
```

Para el caso "pipeline no inicializado":
```python
with patch("src.api.server.RAGPipeline", side_effect=RuntimeError("No pipeline")):
    client = TestClient(app)  # lifespan falla → rag_pipeline queda None → 503
```

En Java con Spring Boot, esto se resuelve con `@MockBean`:
```java
@MockBean
private RAGPipeline ragPipeline;
```

En Python no hay `@MockBean`, asi que tenemos que entender el flujo
de inicializacion y parchear en el punto correcto.

---

## 13. Integration tests - real embeddings + mock LLM

Los integration tests usan servicios reales donde es gratis (embeddings, Qdrant)
y mock donde cuesta dinero o no es determinista (LLM):

```python
@pytest.fixture(scope="class")
def embedding_service():
    return EmbeddingService()  # real, carga modelo una vez

@pytest.fixture
def vector_db():
    return QdrantDatabase(in_memory=True)  # real, en memoria

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate.side_effect = lambda prompt, context=None, **kwargs: (
        f"Based on the context, here is the answer about: {prompt}"
    )
    return llm
```

El test mas importante verifica que la busqueda semantica funciona end-to-end:

```python
def test_query_returns_relevant_sources(self, embedding_service, populated_db, mock_llm):
    pipeline = RAGPipeline(
        embedding_service=embedding_service,
        vector_db=populated_db,
        llm_provider=mock_llm,
    )
    response = pipeline.query("How do I use Docker containers?")

    source_files = [s.source_file for s in response.sources]
    assert "docker_guide.md" in source_files  # busqueda semantica funciona!
```

Esto prueba que la cadena completa (query → embed → search → sources) funciona
con datos reales. El unico mock es el LLM porque no tenemos Ollama corriendo.

Nota: los IDs de Qdrant deben ser UUIDs validos (`str(uuid.uuid4())`), no
strings arbitrarios como `"chunk_0"`.

---

## 14. time.perf_counter vs time.time

Usamos `time.perf_counter()` en vez de `time.time()` para medir tiempos:

```python
retrieval_start = time.perf_counter()
# ... operaciones ...
retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
```

| Funcion | Resolucion | Afectada por reloj | Uso |
|---------|-----------|-------------------|-----|
| `time.time()` | ~15ms (Windows) | Si (NTP, DST) | Timestamps absolutos |
| `time.perf_counter()` | ~100ns | No | Benchmarks, duraciones |

En Java es `System.nanoTime()` (perf_counter) vs `System.currentTimeMillis()` (time).

La spec original usaba `time.time()`. Lo cambiamos a `perf_counter()` porque
queremos medir duraciones precisas, no timestamps.

---

## 15. Tabla resumen Java vs Python en M7

| Concepto | Java | Python (nuestro codigo) |
|----------|------|------------------------|
| Orquestador | Service con @Autowired | RAGPipeline con Optional params |
| DI | Spring Container | Constructor con defaults |
| DTO interno | record / POJO | @dataclass |
| DTO de API | @Valid + Bean Validation | Pydantic BaseModel + Field |
| REST API | Spring Boot @RestController | FastAPI @app.post |
| Inicializacion | @PostConstruct / @Bean | lifespan + global |
| Streaming | Flux<String> (WebFlux) | Iterator[str] + StreamingResponse |
| CLI formatting | JLine + Jansi | rich (Console, Panel, Markdown) |
| REPL loop | Scanner + while(true) | Prompt.ask + while True |
| Lazy init | @Lazy / Supplier<T> | if self.x is None: self.x = X() |
| Test mock API | @MockBean | patch("module.Class", return_value=mock) |
| Timer preciso | System.nanoTime() | time.perf_counter() |
| Integration test | @SpringBootTest + H2 | Real EmbeddingService + in-memory Qdrant |
| Validacion HTTP | @Min, @Max, @NotBlank | Field(ge=1, le=20, min_length=1) |
| Error HTTP | @ExceptionHandler | raise HTTPException(status_code=503) |
| Global state | @Singleton bean | module-level variable (global) |

---

## Resumen

M7 es el milestone de **integracion**. No hay algoritmos nuevos ni patrones
nuevos — lo que hay es la orquestacion de todo lo anterior:

1. **RAGPipeline** conecta M2 + M3 + M6 con DI simple
2. **FastAPI** expone el pipeline como REST API con validacion
3. **CLI** lo expone como REPL interactivo con rich
4. **Tests** verifican que todo funciona junto

El proyecto esta ahora **feature-complete**. Los 7 milestones estan terminados
con 185 tests pasando.
