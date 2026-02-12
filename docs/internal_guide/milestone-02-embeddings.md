# Guia Interna - Milestone 2: Servicio de Embeddings

> Guia escrita para desarrolladores con experiencia en Java.
> Cada concepto de Python se compara con su equivalente en Java.

---

## Indice

1. [Estructura del proyecto vs Maven/Gradle](#1-estructura-del-proyecto)
2. [config.py - Las constantes](#2-configpy---las-constantes)
3. [\_\_init\_\_.py - El package-info de Python](#3-__init__py---el-package-info-de-python)
4. [EmbeddingService - La clase principal](#4-embeddingservice---la-clase-principal)
5. [El constructor \_\_init\_\_](#5-el-constructor-__init__)
6. [generate_embedding() - Generar un embedding](#6-generate_embedding---generar-un-embedding)
7. [generate_batch_embeddings() - Procesar en lote](#7-generate_batch_embeddings---procesar-en-lote)
8. [get_model_info() - Obtener metadatos](#8-get_model_info---obtener-metadatos)
9. [Los tests - pytest vs JUnit](#9-los-tests---pytest-vs-junit)
10. [Flujo completo de ejecucion](#10-flujo-completo-de-ejecucion)
11. [Diferencias culturales Python vs Java](#11-diferencias-culturales-python-vs-java)

---

## 1. Estructura del proyecto

### En Java (Maven)

```
src/
  main/
    java/
      com/docvault/embeddings/
        EmbeddingConfig.java        ← Constantes
        EmbeddingService.java       ← Clase principal
    resources/
      application.properties        ← Configuracion
  test/
    java/
      com/docvault/embeddings/
        EmbeddingServiceTest.java   ← Tests JUnit
pom.xml                             ← Dependencias
```

En Java, cada clase es un archivo. Los paquetes se definen con la ruta de carpetas
y la declaracion `package com.docvault.embeddings;` al inicio de cada archivo.

### En Python (nuestro proyecto)

```
src/
  embeddings/
    __init__.py              ← Define que es un paquete (como package-info.java)
    config.py                ← Constantes (no necesita clase)
    embedding_service.py     ← Clase principal
tests/
  test_embeddings.py         ← Tests pytest (como JUnit)
scripts/
  test_embedding.py          ← Script de verificacion manual
requirements.txt             ← Dependencias (como pom.xml)
```

**Diferencia clave:** En Java, cada archivo `.java` DEBE tener una clase publica.
En Python, un archivo `.py` puede tener funciones sueltas, constantes, multiples clases
o cualquier combinacion. No hay esa restriccion.

---

## 2. config.py - Las constantes

### El archivo Python

```python
# Linea 13
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Lineas 16-18
MODEL_DIMENSIONS = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
}

# Linea 21
SUPPORTED_LANGUAGES = ["en", "es"]
```

### Equivalente en Java

```java
public final class EmbeddingConfig {

    // Constantes (static final en Java = variable a nivel de modulo en Python)
    public static final String DEFAULT_MODEL =
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2";

    // Map<String, Integer> (dict en Python)
    public static final Map<String, Integer> MODEL_DIMENSIONS = Map.of(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384
    );

    // List<String> (list en Python)
    public static final List<String> SUPPORTED_LANGUAGES = List.of("en", "es");

    // Constructor privado para evitar instanciacion
    private EmbeddingConfig() {}
}
```

### Que esta pasando

| Python | Java | Explicacion |
|--------|------|-------------|
| `DEFAULT_MODEL = "..."` | `public static final String DEFAULT_MODEL = "..."` | Constante. En Python no hace falta clase contenedora |
| `MODEL_DIMENSIONS = {...}` | `Map.of(...)` | Diccionario / Mapa. Relaciona nombre del modelo con su dimension |
| `SUPPORTED_LANGUAGES = [...]` | `List.of(...)` | Lista. Idiomas que soporta el modelo |

**Por que 384?** Es el numero de dimensiones (numeros) que genera este modelo concreto.
Cada texto se convierte en una lista de 384 floats. Es una caracteristica fija del modelo,
como el tipo de retorno de un metodo.

**Nota:** En Python, las constantes se escriben en MAYUSCULAS por convencion, pero
NO son inmutables realmente. Es una convencion, no una restriccion del lenguaje.
En Java, `final` impide la reasignacion a nivel de compilador.

---

## 3. \_\_init\_\_.py - El package-info de Python

### El archivo Python

```python
# Linea 9
from .embedding_service import EmbeddingService

# Lineas 11-13
__all__ = [
    "EmbeddingService",
]
```

### Equivalente en Java

En Java no hay un equivalente directo. Lo mas parecido seria un `package-info.java`
combinado con la visibilidad de las clases:

```java
// En Java, la visibilidad la controla el modificador de acceso:
public class EmbeddingService { ... }   // Visible desde fuera del paquete
class InternalHelper { ... }            // Solo visible dentro del paquete
```

### Que esta pasando

Este archivo hace **dos cosas**:

**1. Define que esta carpeta es un paquete Python.**

Sin `__init__.py`, Python NO reconoce la carpeta `embeddings/` como un paquete
importable. Es como si en Java borraras la declaracion `package` de todos los archivos.

**2. Controla que se exporta.**

```python
from .embedding_service import EmbeddingService
```

El punto `.` significa "desde este mismo paquete". Es un **import relativo**.

Sin esta linea, para usar el servicio tendrias que escribir:
```python
from src.embeddings.embedding_service import EmbeddingService  # Largo y feo
```

Con esta linea, puedes escribir:
```python
from src.embeddings import EmbeddingService  # Corto y limpio
```

En Java seria como tener un `com.docvault.embeddings.Embeddings` que re-exporta las
clases internas:

```java
// Imaginario - Java no tiene esto, pero el concepto seria:
package com.docvault.embeddings;
public export EmbeddingService;  // "Cuando importes este paquete, te doy esto"
```

**`__all__`** es una lista que dice: "si alguien hace `from src.embeddings import *`,
solo le des `EmbeddingService`". Es como controlar que clases son `public` en Java.

---

## 4. EmbeddingService - La clase principal

### Estructura de la clase en Python

```python
class EmbeddingService:
    """Docstring de la clase (como Javadoc)."""

    def __init__(self, model_name=None):    # Constructor
        self.model_name = ...               # Atributos de instancia
        self.model = ...
        self.embedding_dimension = ...

    def generate_embedding(self, text):     # Metodo publico
        ...

    def generate_batch_embeddings(self, texts, batch_size=32):  # Metodo publico
        ...

    def get_model_info(self):               # Metodo publico (getter)
        ...
```

### Equivalente en Java

```java
public class EmbeddingService {

    // Campos (en Python no se declaran aqui, se crean en el constructor)
    private final String modelName;
    private final SentenceTransformer model;
    private final int embeddingDimension;

    // Constructor
    public EmbeddingService(String modelName) { ... }

    // Metodos publicos
    public List<Float> generateEmbedding(String text) { ... }
    public List<List<Float>> generateBatchEmbeddings(List<String> texts, int batchSize) { ... }
    public Map<String, Object> getModelInfo() { ... }
}
```

### Diferencias importantes

| Concepto | Java | Python |
|----------|------|--------|
| Declarar campos | Al inicio de la clase: `private String modelName;` | No se declaran. Se crean en `__init__` con `self.modelName = ...` |
| Constructor | `public EmbeddingService(...)` | `def __init__(self, ...)` |
| Referencia a "this" | `this.modelName` (opcional en Java) | `self.model_name` (OBLIGATORIO siempre) |
| Visibilidad | `public`, `private`, `protected` | Por convencion: sin prefijo = public, `_prefijo` = protected, `__prefijo` = private |
| Naming | camelCase: `generateEmbedding` | snake_case: `generate_embedding` |
| Tipos | Obligatorios: `String text` | Opcionales (type hints): `text: str` |

**Nota sobre `self`:** En Java, `this` es implicito. Puedes escribir `modelName` o `this.modelName`.
En Python, `self` es **siempre obligatorio**. Si escribes `model_name` sin `self.`, Python
piensa que es una variable local, no un atributo de la instancia. Es el error mas comun
para desarrolladores que vienen de Java.

---

## 5. El constructor \_\_init\_\_

### El codigo Python (lineas 40-74)

```python
def __init__(self, model_name: Optional[str] = None) -> None:
    # 1. Asignar modelo (usa default si no se pasa nada)
    self.model_name = model_name or DEFAULT_MODEL

    logger.info(f"Loading embedding model: {self.model_name}")

    try:
        # 2. Cargar el modelo de IA en memoria
        self.model = SentenceTransformer(self.model_name)

        # 3. Obtener dimension de los embeddings
        self.embedding_dimension = MODEL_DIMENSIONS.get(
            self.model_name,
            384  # Fallback si no esta en el diccionario
        )

        logger.info(f"Embedding model loaded successfully. Dimension: {self.embedding_dimension}")

    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise RuntimeError(f"Could not initialize embedding model '{self.model_name}': {e}") from e
```

### Equivalente linea a linea en Java

```java
public class EmbeddingService {

    private static final Logger logger = LoggerFactory.getLogger(EmbeddingService.class);

    private final String modelName;
    private final SentenceTransformer model;
    private final int embeddingDimension;

    /**
     * Constructor.
     * @param modelName Nombre del modelo. Si es null, usa DEFAULT_MODEL.
     * @throws RuntimeException Si el modelo no puede cargarse.
     */
    public EmbeddingService(String modelName) {
        // 1. model_name or DEFAULT_MODEL
        //    En Java: operador ternario
        this.modelName = (modelName != null) ? modelName : EmbeddingConfig.DEFAULT_MODEL;

        logger.info("Loading embedding model: {}", this.modelName);

        try {
            // 2. SentenceTransformer(self.model_name)
            //    Carga el modelo de IA. Primera vez descarga ~120MB.
            this.model = new SentenceTransformer(this.modelName);

            // 3. MODEL_DIMENSIONS.get(self.model_name, 384)
            //    En Java: getOrDefault()
            this.embeddingDimension = EmbeddingConfig.MODEL_DIMENSIONS
                .getOrDefault(this.modelName, 384);

            logger.info("Embedding model loaded successfully. Dimension: {}",
                this.embeddingDimension);

        } catch (Exception e) {
            logger.error("Failed to load embedding model: {}", e.getMessage());
            // raise RuntimeError(...) from e
            // En Java: throw new RuntimeException(..., e)
            //    El "from e" en Python es como pasar la causa (e) al constructor
            throw new RuntimeException(
                "Could not initialize embedding model '" + this.modelName + "': " + e.getMessage(),
                e  // causa original
            );
        }
    }
}
```

### Paso a paso lo que ocurre

**Paso 1 - Parametro con valor por defecto:**
```python
def __init__(self, model_name: Optional[str] = None) -> None:
```

En Java no existen parametros con valor por defecto. Harias sobrecarga:
```java
public EmbeddingService() {
    this(null);  // Llama al otro constructor
}

public EmbeddingService(String modelName) {
    // ...
}
```

En Python, `Optional[str] = None` significa: "acepta un String o None, y si no pasan nada,
sera None". Es un solo constructor que cubre los dos casos.

**Paso 2 - Operador `or`:**
```python
self.model_name = model_name or DEFAULT_MODEL
```

Esto NO es un OR logico como en Java. En Python, `or` devuelve el primer valor "truthy".
- Si `model_name` es `None` o `""` → devuelve `DEFAULT_MODEL`
- Si `model_name` tiene valor → devuelve ese valor

Equivale en Java a:
```java
this.modelName = (modelName != null && !modelName.isEmpty())
    ? modelName
    : DEFAULT_MODEL;
```

**Paso 3 - Cargar el modelo:**
```python
self.model = SentenceTransformer(self.model_name)
```

Esta es la linea mas pesada. Internamente:
1. Mira si el modelo ya esta descargado en `~/.cache/`
2. Si NO → lo descarga de Hugging Face (~120MB)
3. Si SI → lo carga desde disco
4. Inicializa la red neuronal en memoria

En Java, seria como hacer `new SentenceTransformer(modelName)` donde el constructor
hace una descarga HTTP, deserializa un modelo binario y lo carga en la JVM.

**Paso 4 - Buscar en el diccionario:**
```python
self.embedding_dimension = MODEL_DIMENSIONS.get(self.model_name, 384)
```

`.get(key, default)` busca la clave en el diccionario. Si no la encuentra, devuelve
el valor por defecto (384). Identico a Java:
```java
this.embeddingDimension = MODEL_DIMENSIONS.getOrDefault(this.modelName, 384);
```

**Paso 5 - Try/catch:**
```python
except Exception as e:
    raise RuntimeError("...") from e
```

`except Exception as e` = `catch (Exception e)` en Java.
`raise RuntimeError(...) from e` = `throw new RuntimeException(..., e)` en Java.
El `from e` preserva la excepcion original como causa, igual que pasar `e` como
segundo parametro en Java.

### Estado del objeto despues del constructor

```
EmbeddingService {
    model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model: <SentenceTransformer>  ← Red neuronal cargada en memoria
    embedding_dimension: 384
}
```

Equivale en Java a que, despues de `new EmbeddingService()`, el objeto tiene
3 campos inicializados y listos para usar.

---

## 6. generate_embedding() - Generar un embedding

### El codigo Python (lineas 76-115)

```python
def generate_embedding(self, text: str) -> list[float]:
    # 1. Validacion
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    try:
        # 2. Llamada al modelo
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 3. Conversion y retorno
        return embedding.tolist()

    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise RuntimeError(f"Embedding generation failed: {e}") from e
```

### Equivalente linea a linea en Java

```java
/**
 * Genera un vector embedding para un texto.
 *
 * @param text Texto de entrada
 * @return Lista de 384 floats representando el embedding
 * @throws IllegalArgumentException Si el texto esta vacio
 * @throws RuntimeException Si falla la generacion
 */
public List<Float> generateEmbedding(String text) {

    // 1. Validacion
    //    "not text" en Python verifica null Y vacio a la vez
    //    "not text.strip()" verifica que no sea solo espacios
    if (text == null || text.trim().isEmpty()) {
        throw new IllegalArgumentException("Input text cannot be empty");
    }

    try {
        // 2. Llamada al modelo
        //    self.model.encode() en Python = this.model.encode() en Java
        //
        //    Esta linea hace INTERNAMENTE:
        //      a) Tokenizar: "Hello world" → [101, 7592, 2088, 102]
        //      b) Forward pass: pasar tokens por 12 capas del transformer
        //      c) Mean pooling: promediar las salidas en un solo vector
        //      d) L2 normalize: escalar el vector para que su magnitud sea 1.0
        //
        //    Devuelve un float[] de 384 elementos
        float[] embedding = this.model.encode(
            text,
            /* convertToNumpy */    true,
            /* normalizeEmbeddings */ true   // L2 normalization
        );

        // 3. Conversion
        //    float[] → List<Float> (autoboxing)
        //    En Python: numpy.ndarray → list[float] con .tolist()
        return Arrays.stream(embedding)
            .boxed()
            .collect(Collectors.toList());

    } catch (Exception e) {
        logger.error("Failed to generate embedding: {}", e.getMessage());
        throw new RuntimeException("Embedding generation failed: " + e.getMessage(), e);
    }
}
```

### Desglose paso a paso

**Paso 1 - Validacion:**
```python
if not text or not text.strip():
```

En Python, `not text` cubre dos casos a la vez:
- Si `text` es `None` → `not None` es `True`
- Si `text` es `""` → `not ""` es `True`

En Java necesitas dos comprobaciones separadas: `text == null || text.isEmpty()`.

`text.strip()` = `text.trim()` en Java. Elimina espacios al inicio y final.

**Paso 2 - La llamada al modelo (`self.model.encode()`):**

Esta unica linea es donde ocurre toda la magia. Por eso puede parecer "desordenado":
en Java harias 5 pasos visibles, aqui la libreria lo encapsula en uno.

Lo que ocurre internamente dentro de `.encode()`:

```
Entrada: "Hello world"
    │
    ▼
[Tokenizacion]
    "Hello world" → [101, 7592, 2088, 102]
    (Convierte palabras a IDs numericos que el modelo entiende)
    En Java seria como: tokenizer.encode("Hello world") → int[]
    │
    ▼
[Forward pass - 12 capas transformer]
    [101, 7592, 2088, 102] → float[4][384]
    (Cada token genera un vector de 384 numeros.
     Son 4 tokens, asi que sale una matriz 4x384)
    En Java seria como: model.forward(tokenIds) → float[][]
    │
    ▼
[Mean pooling]
    float[4][384] → float[384]
    (Promedia los 4 vectores en uno solo.
     Es como hacer la media de cada columna de la matriz)
    En Java seria como: MathUtils.meanPool(hiddenStates) → float[]
    │
    ▼
[L2 Normalization]
    float[384] → float[384]  (pero con magnitud = 1.0)
    (Divide cada valor por la magnitud total del vector.
     Esto hace que todos los vectores "vivan" en una esfera unitaria,
     facilitando la comparacion con cosine similarity)
    En Java seria como: MathUtils.l2Normalize(vector) → float[]
    │
    ▼
Salida: numpy.ndarray de 384 floats (equivale a float[384] en Java)
```

**Por que en Python se hace en una sola linea y en Java harias 5?**

Porque la libreria `sentence-transformers` encapsula todo el pipeline.
Es como si en Java usaras Spring Boot: tu escribes `@GetMapping("/users")`
y Spring hace internamente 20 pasos (parsear HTTP, deserializar, rutear, etc.).
El concepto es el mismo: la libreria oculta la complejidad.

**Paso 3 - Conversion (`embedding.tolist()`):**

`self.model.encode()` devuelve un `numpy.ndarray`. Numpy es una libreria de calculo
numerico que usa arrays nativos de C por debajo (mucho mas rapidos que listas Python).

`.tolist()` convierte ese array nativo a una `list[float]` estandar de Python.

En Java seria como convertir un `float[]` (primitivo) a `List<Float>` (objeto):
```java
Arrays.stream(embedding).boxed().collect(Collectors.toList());
```

**Sobre `normalize_embeddings=True`:**

Esto es un parametro clave. La normalizacion L2 hace que el vector tenga magnitud 1.0.

Sin normalizar:
```
vector = [3.0, 4.0]
magnitud = sqrt(3² + 4²) = sqrt(25) = 5.0
```

Normalizado:
```
vector = [3.0/5.0, 4.0/5.0] = [0.6, 0.8]
magnitud = sqrt(0.6² + 0.8²) = sqrt(1.0) = 1.0
```

**Ventaja:** Cuando dos vectores estan normalizados, el cosine similarity se simplifica
a un simple dot product (multiplicar y sumar). En vez de calcular:

```
cosine = (A · B) / (|A| × |B|)
```

Como |A| = 1 y |B| = 1, queda:

```
cosine = A · B    ← Mucho mas rapido
```

En Java seria equivalente a preprocessing de datos antes de meterlos en un indice de busqueda.

---

## 7. generate_batch_embeddings() - Procesar en lote

### El codigo Python (lineas 117-173)

```python
def generate_batch_embeddings(
    self,
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = False
) -> list[list[float]]:
    # 1. Validacion
    if not texts:
        raise ValueError("Input text list cannot be empty")
    if any(not text or not text.strip() for text in texts):
        raise ValueError("Input texts cannot contain empty strings")

    try:
        # 2. Procesar en lote
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 3. Convertir y devolver
        return embeddings.tolist()

    except Exception as e:
        raise RuntimeError(f"Batch embedding generation failed: {e}") from e
```

### Equivalente en Java

```java
/**
 * Genera embeddings para multiples textos de forma eficiente.
 *
 * @param texts Lista de textos
 * @param batchSize Tamano del lote (default 32)
 * @param showProgress Mostrar barra de progreso
 * @return Lista de embeddings (uno por texto)
 */
public List<List<Float>> generateBatchEmbeddings(
        List<String> texts,
        int batchSize,
        boolean showProgress) {

    // 1. Validacion
    if (texts == null || texts.isEmpty()) {
        throw new IllegalArgumentException("Input text list cannot be empty");
    }

    // any() en Python = anyMatch() en Java Streams
    boolean hasEmpty = texts.stream()
        .anyMatch(t -> t == null || t.trim().isEmpty());
    if (hasEmpty) {
        throw new IllegalArgumentException("Input texts cannot contain empty strings");
    }

    try {
        // 2. Procesar en lote
        //    Misma llamada que generate_embedding, pero pasando una LISTA
        //    La libreria agrupa los textos en lotes de 32 y los procesa en paralelo
        float[][] embeddings = this.model.encode(
            texts.toArray(new String[0]),
            batchSize,
            showProgress,
            true,  // convertToNumpy
            true   // normalizeEmbeddings
        );

        // 3. Convertir float[][] → List<List<Float>>
        return Arrays.stream(embeddings)
            .map(row -> Arrays.stream(row).boxed().collect(Collectors.toList()))
            .collect(Collectors.toList());

    } catch (Exception e) {
        throw new RuntimeException("Batch embedding generation failed: " + e.getMessage(), e);
    }
}
```

### Diferencia clave con generate_embedding()

Es **exactamente la misma logica**, pero en vez de un texto, procesa una lista.
La diferencia importante es `batch_size=32`.

**Sin batch (llamando a generate_embedding() en un bucle):**
```python
# Procesa 1000 textos UNO A UNO
for text in texts:
    embedding = service.generate_embedding(text)  # 1000 llamadas al modelo
```

En Java:
```java
// Equivalente: 1000 iteraciones secuenciales
for (String text : texts) {
    List<Float> embedding = service.generateEmbedding(text);
}
```

**Con batch:**
```python
# Procesa 1000 textos en GRUPOS DE 32
embeddings = service.generate_batch_embeddings(texts, batch_size=32)
# Solo ~32 llamadas al modelo (1000/32 = 31.25 → 32 lotes)
```

En Java seria como la diferencia entre:
```java
// Sin batch: INSERT INTO ... VALUES (1); INSERT INTO ... VALUES (2); ...
// Con batch: INSERT INTO ... VALUES (1), (2), (3), ... (32);
```

El modelo procesa 32 textos en paralelo en cada pasada. La GPU/CPU puede paralelizar
operaciones sobre matrices, asi que procesar 32 textos cuesta casi lo mismo que procesar 1.
Resultado: **~31x mas rapido**.

### La validacion con `any()`

```python
if any(not text or not text.strip() for text in texts):
```

`any()` recorre la lista y devuelve `True` si ALGUN elemento cumple la condicion.
Es como `Stream.anyMatch()` en Java:

```java
texts.stream().anyMatch(t -> t == null || t.trim().isEmpty());
```

La expresion `not text or not text.strip() for text in texts` es un **generator expression**.
Es como un Stream en Java: no crea una lista intermedia, evalua "lazy" uno por uno.

---

## 8. get_model_info() - Obtener metadatos

### El codigo Python (lineas 175-195)

```python
def get_model_info(self) -> dict[str, str | int]:
    return {
        "model_name": self.model_name,
        "embedding_dimension": self.embedding_dimension,
        "max_seq_length": self.model.max_seq_length,
    }
```

### Equivalente en Java

```java
public Map<String, Object> getModelInfo() {
    // dict en Python = Map en Java
    // str | int en Python = Object en Java (union type)
    return Map.of(
        "model_name", this.modelName,                    // String
        "embedding_dimension", this.embeddingDimension,  // int (autoboxed a Integer)
        "max_seq_length", this.model.getMaxSeqLength()   // int
    );
}
```

### Nota sobre tipos

`dict[str, str | int]` es un type hint de Python que dice:
- Las claves son `str`
- Los valores pueden ser `str` O `int`

En Java no existe union type nativo. Lo mas cercano seria `Map<String, Object>`
o crear un record/DTO:

```java
// Alternativa mas "Java-like": usar un DTO
public record ModelInfo(
    String modelName,
    int embeddingDimension,
    int maxSeqLength
) {}

public ModelInfo getModelInfo() {
    return new ModelInfo(
        this.modelName,
        this.embeddingDimension,
        this.model.getMaxSeqLength()
    );
}
```

En Python, devolver un diccionario es la forma idiomatica. En Java, se prefiere un DTO/Record.
Ambos enfoques son validos. Python prioriza simplicidad, Java prioriza tipado fuerte.

---

## 9. Los tests - pytest vs JUnit

### Fixture vs @BeforeAll

**Python (pytest):**
```python
class TestEmbeddingService:

    @pytest.fixture(scope="class")
    def service(self) -> EmbeddingService:
        return EmbeddingService()
```

**Java (JUnit 5):**
```java
class EmbeddingServiceTest {

    private static EmbeddingService service;

    @BeforeAll
    static void setUp() {
        service = new EmbeddingService();
    }
}
```

`scope="class"` en pytest = `@BeforeAll` en JUnit. Crea la instancia UNA SOLA VEZ
para todos los tests de la clase. Sin esto, cada test cargaria el modelo de nuevo (~5 segundos).

La diferencia es que en pytest, el fixture se **inyecta como parametro** en cada test:
```python
def test_algo(self, service: EmbeddingService):  # ← recibe el fixture
```

En JUnit, accedes al campo estatico directamente:
```java
void testAlgo() {
    service.generateEmbedding(...);  // ← usa el campo estatico
}
```

### Assert vs assertEquals

**Python:**
```python
def test_generate_embedding_returns_correct_dimensions(self, service):
    embedding = service.generate_embedding("hello world")

    assert isinstance(embedding, list)      # Verifica tipo
    assert len(embedding) == 384            # Verifica tamanio
    assert all(isinstance(x, float) for x in embedding)  # Todos son float
```

**Java:**
```java
@Test
void testGenerateEmbeddingReturnsCorrectDimensions() {
    List<Float> embedding = service.generateEmbedding("hello world");

    assertNotNull(embedding);                           // No es null
    assertInstanceOf(List.class, embedding);             // Es una lista
    assertEquals(384, embedding.size());                 // Tamano correcto
    assertTrue(embedding.stream().allMatch(x -> x instanceof Float));  // Todos Float
}
```

**Diferencias de estilo:**

| pytest (Python) | JUnit (Java) |
|-----------------|-------------|
| `assert condicion` | `assertTrue(condicion)` |
| `assert a == b` | `assertEquals(a, b)` |
| `assert a > b, "mensaje"` | `assertTrue(a > b, "mensaje")` |
| `with pytest.raises(ValueError):` | `assertThrows(IllegalArgumentException.class, () -> ...)` |

### Test de excepciones

**Python:**
```python
def test_generate_embedding_raises_error_on_empty_string(self, service):
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        service.generate_embedding("")
```

**Java:**
```java
@Test
void testGenerateEmbeddingRaisesErrorOnEmptyString() {
    IllegalArgumentException ex = assertThrows(
        IllegalArgumentException.class,
        () -> service.generateEmbedding("")
    );
    assertTrue(ex.getMessage().contains("Input text cannot be empty"));
}
```

`pytest.raises(ValueError, match="...")` hace dos cosas:
1. Verifica que se lance `ValueError` (como `assertThrows`)
2. Verifica que el mensaje contenga "..." (como una comprobacion extra del mensaje)

### Test de cosine similarity

**Python:**
```python
def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)
```

**Java:**
```java
static double cosineSimilarity(List<Float> vec1, List<Float> vec2) {
    // zip(vec1, vec2) en Python = IntStream.range() en Java
    double dotProduct = IntStream.range(0, vec1.size())
        .mapToDouble(i -> vec1.get(i) * vec2.get(i))
        .sum();

    double magnitude1 = Math.sqrt(vec1.stream()
        .mapToDouble(a -> a * a).sum());

    double magnitude2 = Math.sqrt(vec2.stream()
        .mapToDouble(b -> b * b).sum());

    if (magnitude1 == 0 || magnitude2 == 0) return 0.0;

    return dotProduct / (magnitude1 * magnitude2);
}
```

**Nota sobre `zip()`:** En Python, `zip(vec1, vec2)` empareja elemento a elemento:
```python
zip([1, 2, 3], [4, 5, 6]) → [(1,4), (2,5), (3,6)]
```
En Java no existe `zip` nativo. Se simula con `IntStream.range(0, size)` para iterar
por indice.

---

## 10. Flujo completo de ejecucion

Esto es lo que pasa paso a paso cuando ejecutas la aplicacion:

```
PASO 1: Importacion
━━━━━━━━━━━━━━━━━━━
from src.embeddings import EmbeddingService

    Python busca: src/embeddings/__init__.py
        ├── Ejecuta: from .embedding_service import EmbeddingService
        │       ├── Carga embedding_service.py
        │       │       ├── Ejecuta: from .config import DEFAULT_MODEL, MODEL_DIMENSIONS
        │       │       │       └── Carga config.py → define las constantes
        │       │       └── Define la clase EmbeddingService (NO la instancia)
        │       └── Devuelve la clase al __init__.py
        └── Ahora "EmbeddingService" esta disponible para importar

    En Java: ClassLoader carga la clase cuando se referencia por primera vez.
    En Python: el modulo se "ejecuta" al importarse (las lineas de codigo se ejecutan).


PASO 2: Instanciacion
━━━━━━━━━━━━━━━━━━━━━
service = EmbeddingService()

    Llama a __init__(self, model_name=None)
        │
        ├── self.model_name = None or DEFAULT_MODEL
        │       → "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        │
        ├── self.model = SentenceTransformer(self.model_name)
        │       ├── Busca en cache: ~/.cache/torch/sentence_transformers/
        │       ├── Si no existe → descarga de huggingface.co (~120MB)
        │       ├── Carga el tokenizador (vocabulario de ~30K palabras)
        │       ├── Carga los pesos del modelo (12 capas transformer)
        │       └── Modelo listo en RAM (o GPU si hay CUDA)
        │
        └── self.embedding_dimension = MODEL_DIMENSIONS.get(...) → 384

    En Java: equivale a new EmbeddingService() donde el constructor
    descarga y carga un modelo ML. Como inicializar TensorFlow en Java.


PASO 3: Generar un embedding
━━━━━━━━━━━━━━━━━━━━━━━━━━━
embedding = service.generate_embedding("Hello world")

    generate_embedding(self, text="Hello world")
        │
        ├── Validacion: text no es null/vacio/espacios → OK
        │
        ├── self.model.encode("Hello world", normalize=True)
        │       │
        │       ├── Tokenizar: "Hello world" → [101, 7592, 2088, 102]
        │       │       (101 = [CLS], 102 = [SEP] son tokens especiales)
        │       │
        │       ├── Forward pass por 12 capas transformer:
        │       │       Capa 1:  [101, 7592, 2088, 102] → float[4][384]
        │       │       Capa 2:  float[4][384] → float[4][384]
        │       │       ...
        │       │       Capa 12: float[4][384] → float[4][384]
        │       │
        │       ├── Mean pooling:
        │       │       float[4][384] → float[384]
        │       │       (promedia los 4 vectores de tokens en 1 solo vector)
        │       │
        │       └── L2 normalization:
        │               float[384] → float[384] (magnitud = 1.0)
        │
        └── embedding.tolist()
                numpy.ndarray(384,) → list[float] de 384 elementos

    Resultado: [0.023, -0.152, 0.089, ..., 0.016]  (384 numeros entre -1 y 1)


PASO 4: Generar embeddings en lote
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
embeddings = service.generate_batch_embeddings(["texto1", "texto2", "texto3"])

    generate_batch_embeddings(self, texts=["texto1", "texto2", "texto3"], batch_size=32)
        │
        ├── Validacion: lista no vacia, ningun elemento vacio → OK
        │
        ├── self.model.encode(["texto1", "texto2", "texto3"], batch_size=32)
        │       │
        │       ├── Agrupa en lotes de 32 (aqui solo hay 3, asi que 1 lote)
        │       ├── Tokeniza los 3 textos
        │       ├── Procesa los 3 en PARALELO (no secuencial)
        │       ├── Mean pooling para cada uno
        │       └── L2 normalization para cada uno
        │
        └── embeddings.tolist()
                numpy.ndarray(3, 384) → list[list[float]]

    Resultado: [
        [0.023, -0.152, ...],  ← embedding de "texto1" (384 floats)
        [0.031, -0.098, ...],  ← embedding de "texto2" (384 floats)
        [0.019, -0.143, ...]   ← embedding de "texto3" (384 floats)
    ]
```

---

## 11. Diferencias culturales Python vs Java

### Resumen de equivalencias encontradas en este codigo

| Concepto | Python | Java |
|----------|--------|------|
| Constantes | `UPPER_CASE = valor` (convencion) | `public static final Type NAME = valor` (forzado) |
| Paquete | Carpeta + `__init__.py` | Carpeta + `package` en cada archivo |
| Constructor | `def __init__(self, ...)` | `public ClassName(...)` |
| This/Self | `self.campo` (obligatorio) | `this.campo` (opcional) |
| Null | `None` | `null` |
| Parametro default | `def f(x=5)` | Sobrecarga de metodos |
| Operador or | `a or b` (devuelve el primero truthy) | `a != null ? a : b` |
| Dict / Map | `{"key": value}` | `Map.of("key", value)` |
| List | `[1, 2, 3]` | `List.of(1, 2, 3)` |
| Excepciones checked | No existen | `throws IOException` |
| Tipado | Opcional (type hints) | Obligatorio |
| Naming | snake_case | camelCase |
| Getters/Setters | No se usan (acceso directo) | Se usan siempre |
| Stream/any | `any(x for x in lista)` | `lista.stream().anyMatch(...)` |
| Fixture/BeforeAll | `@pytest.fixture(scope="class")` | `@BeforeAll` |
| Assert | `assert condicion` | `assertTrue(condicion)` |
| Test excepciones | `with pytest.raises(Error):` | `assertThrows(Error.class, () -> ...)` |

### Filosofia general

**Java dice:** "Se explicito. Declara tipos. Crea clases para todo. Haz que el compilador
te proteja. Si algo puede ser privado, hazlo privado."

**Python dice:** "Se simple. Si funciona sin clase, no crees clase. Si el tipo es obvio,
no lo declares. Confia en el programador. Menos codigo = menos bugs."

Ninguno es mejor. Son filosofias diferentes. Lo importante es entender AMBAS para poder
leer y escribir codigo en cualquiera de los dos lenguajes.

---

**Archivos documentados:**
- [src/embeddings/config.py](../../src/embeddings/config.py) - Constantes de configuracion
- [src/embeddings/\_\_init\_\_.py](../../src/embeddings/__init__.py) - Exportaciones del paquete
- [src/embeddings/embedding_service.py](../../src/embeddings/embedding_service.py) - Servicio principal
- [tests/test_embeddings.py](../../tests/test_embeddings.py) - Tests unitarios
