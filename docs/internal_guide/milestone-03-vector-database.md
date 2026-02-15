# Guia Interna - Milestone 3: Base de Datos Vectorial (Qdrant)

> Guia escrita para desarrolladores con experiencia en Java.
> Cada concepto de Python se compara con su equivalente en Java.

---

## Indice

1. [Que hemos construido](#1-que-hemos-construido)
2. [La interfaz abstracta - vector_database.py](#2-la-interfaz-abstracta---vector_databasepy)
3. [Las constantes - config.py](#3-las-constantes---configpy)
4. [La implementacion - qdrant_database.py](#4-la-implementacion---qdrant_databasepy)
5. [El constructor \_\_init\_\_](#5-el-constructor-__init__)
6. [initialize_collection() - Crear la tabla](#6-initialize_collection---crear-la-tabla)
7. [insert_vectors() - El INSERT INTO](#7-insert_vectors---el-insert-into)
8. [search_similar() - La busqueda por similitud](#8-search_similar---la-busqueda-por-similitud)
9. [delete_by_id() - Borrar vectores](#9-delete_by_id---borrar-vectores)
10. [get_collection_info() - Info de la coleccion](#10-get_collection_info---info-de-la-coleccion)
11. [El \_\_init\_\_.py - La fachada publica](#11-el-__init__py---la-fachada-publica)
12. [Tests: unit vs integration](#12-tests-unit-vs-integration)
13. [Flujo completo M2 + M3](#13-flujo-completo-m2--m3)
14. [Tabla resumen Java vs Python en M3](#14-tabla-resumen-java-vs-python-en-m3)

---

## 1. Que hemos construido

En M2 creamos el servicio que convierte texto en vectores (listas de 384 numeros).
Pero esos vectores se generaban y se perdian — no los guardabamos en ningun sitio.

En M3 hemos creado la **"base de datos"** donde guardar esos vectores y buscar en ellos.

Piensalo como una biblioteca: M2 es el proceso de crear una "huella numerica" de cada
documento. M3 es el **archivador** donde guardar esas huellas y un **buscador** que,
cuando le das una pregunta, encuentra los documentos mas parecidos.

El archivador es **Qdrant**, una base de datos especializada en vectores.

---

## 2. La interfaz abstracta - vector_database.py

En Java harias:

```java
public interface VectorDatabase {
    void initializeCollection();
    void insertVectors(List<String> ids, List<float[]> vectors, List<Map<String, Object>> metadata);
    List<SearchResult> searchSimilar(float[] queryVector, int limit, Float scoreThreshold);
    void deleteById(List<String> ids);
    Map<String, Object> getCollectionInfo();
}
```

En Python es casi identico, pero usamos `ABC` (Abstract Base Class):

```python
class VectorDatabase(ABC):

    @abstractmethod
    def insert_vectors(self, ids: list[str], vectors: list[list[float]],
                       metadata: list[dict[str, Any]]) -> None:
        pass
```

| Java | Python |
|------|--------|
| `interface VectorDatabase` | `class VectorDatabase(ABC)` |
| Metodos sin cuerpo automaticamente | `@abstractmethod` + `pass` |
| `List<String>` | `list[str]` |
| `List<Map<String, Object>>` | `list[dict[str, Any]]` |
| `@Nullable Float` | `Optional[float] = None` |

### implements vs herencia

En Java hay una distincion clara:

```java
class QdrantDatabase extends AbstractClass { }   // Herencia
class QdrantDatabase implements VectorDatabase { }  // Interfaz
```

En Python **no existe esa distincion sintactica**. Ambos casos usan lo mismo:

```python
class QdrantDatabase(VectorDatabase):  # Herencia? Interfaz? Las dos cosas.
```

La diferencia la marca **la clase padre**, no la clase hija:

- Si `VectorDatabase` tiene metodos con codigo → es herencia (como `extends`)
- Si `VectorDatabase` tiene metodos con `@abstractmethod` y `pass` → es una interfaz (como `implements`)

En nuestro caso, `VectorDatabase` tiene **todos** los metodos abstractos, asi que
funcionalmente **es un `implements`**, aunque la sintaxis parezca un `extends`.

**Proteccion en tiempo de ejecucion:** Si `QdrantDatabase` no implementa algun metodo
abstracto, Python da error al instanciar:

```python
db = QdrantDatabase()  # TypeError: Can't instantiate abstract class
                       # QdrantDatabase with abstract method search_similar
```

Equivale al error de compilacion de Java: `"QdrantDatabase is not abstract and does
not override abstract method"`. La diferencia es que Java lo detecta al compilar y
Python al ejecutar.

---

## 3. Las constantes - config.py

### En Java

```java
public final class DatabaseConfig {
    public static final String DEFAULT_COLLECTION_NAME = "docvault_documents";
    public static final int VECTOR_SIZE = 384;
    public static final String DISTANCE_METRIC = "Cosine";
    public static final int HNSW_M = 16;
    public static final int HNSW_EF_CONSTRUCT = 100;
    public static final String DEFAULT_STORAGE_PATH = "data/qdrant_storage";

    private DatabaseConfig() {} // No instanciable
}
```

### En Python

```python
DEFAULT_COLLECTION_NAME = "docvault_documents"
VECTOR_SIZE = 384
DISTANCE_METRIC = "Cosine"
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100
DEFAULT_STORAGE_PATH = "data/qdrant_storage"
```

No necesitas clase, ni `final`, ni `static`. Python confia en la convencion:
**MAYUSCULAS = constante, no la toques.** No hay proteccion del compilador como
`final` en Java, pero es la practica estandar.

---

## 4. La implementacion - qdrant_database.py

### Estructura general

En Java:

```java
public class QdrantDatabase implements VectorDatabase {
    private final String collectionName;
    private final int vectorSize;
    private final QdrantClient client;

    public QdrantDatabase(String collectionName, int vectorSize, boolean inMemory) {
        this.collectionName = collectionName;
        this.vectorSize = vectorSize;
        if (inMemory) {
            this.client = new QdrantClient(":memory:");
        } else {
            this.client = new QdrantClient(storagePath);
        }
        initializeCollection();
    }
}
```

En Python:

```python
class QdrantDatabase(VectorDatabase):     # "implements VectorDatabase"

    def __init__(self, collection_name="docvault_documents", vector_size=384,
                 in_memory=False):
        self.collection_name = collection_name
        self.vector_size = vector_size

        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(path=str(path))

        self.initialize_collection()
```

| Java | Python |
|------|--------|
| `implements VectorDatabase` | `(VectorDatabase)` en la declaracion |
| Constructor `QdrantDatabase(...)` | `__init__(self, ...)` |
| `this.client` | `self.client` |
| Parametros sin default → sobrecarga | Parametros con default → un solo constructor |

---

## 5. El constructor \_\_init\_\_

### Flujo paso a paso

```
QdrantDatabase(in_memory=True)
    |
    +-- Guardar atributos: self.collection_name, self.vector_size, self.in_memory
    |
    +-- in_memory?
    |   +-- SI → QdrantClient(":memory:")     ← Todo en RAM
    |   +-- NO → path.mkdir(parents=True)     ← Crear directorio si no existe
    |            QdrantClient(path="data/qdrant_storage")  ← En disco
    |
    +-- self.initialize_collection()          ← Crear la "tabla"
    |
    +-- Si algo falla → RuntimeError          ← Como throw new RuntimeException(msg, e)
```

En Java seria como elegir entre `H2 in-memory` o `PostgreSQL` en el constructor de tu DAO.

### Parametros con valores por defecto

```python
def __init__(self, collection_name: str = DEFAULT_COLLECTION_NAME,
             vector_size: int = VECTOR_SIZE, in_memory: bool = False,
             storage_path: str = DEFAULT_STORAGE_PATH) -> None:
```

En Java necesitarias sobrecarga de constructores:

```java
public QdrantDatabase() {
    this(DEFAULT_COLLECTION_NAME, VECTOR_SIZE, false, DEFAULT_STORAGE_PATH);
}

public QdrantDatabase(String collectionName) {
    this(collectionName, VECTOR_SIZE, false, DEFAULT_STORAGE_PATH);
}

public QdrantDatabase(String collectionName, int vectorSize,
                      boolean inMemory, String storagePath) {
    // ... implementacion
}
```

En Python, un solo constructor con valores por defecto cubre todos estos casos.

### El patron try/except con re-raise

```python
try:
    self.client = QdrantClient(":memory:")
    self.initialize_collection()
except Exception as e:
    raise RuntimeError(f"Could not initialize Qdrant: {e}") from e
```

`from e` preserva la excepcion original como causa. Identico a Java:

```java
catch (Exception e) {
    throw new RuntimeException("Could not initialize Qdrant: " + e.getMessage(), e);
}
```

---

## 6. initialize_collection() - Crear la tabla

```python
def initialize_collection(self) -> None:
    collections = self.client.get_collections().collections
    exists = any(c.name == self.collection_name for c in collections)

    if exists:
        return  # No hacer nada

    self.client.create_collection(
        collection_name=self.collection_name,
        vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
    )
```

### Paso a paso

1. **`get_collections()`** — Pide a Qdrant la lista de colecciones. Como `SHOW TABLES` en SQL.

2. **`any(c.name == self.collection_name for c in collections)`** — Recorre la lista
   y devuelve `True` si alguna tiene el nombre que buscamos. En Java:
   ```java
   boolean exists = collections.stream()
       .anyMatch(c -> c.getName().equals(this.collectionName));
   ```

3. **Si ya existe** → `return` sin valor (como `return;` en un metodo void de Java).

4. **Si no existe** → La crea con dos configuraciones:
   - `VectorParams(size=384, distance=COSINE)` — "Cada vector tiene 384 numeros y se
     comparan con similitud coseno". Como definir el tipo de columna y el indice.
   - `HnswConfigDiff(m=16, ef_construct=100)` — Configura el algoritmo de indice HNSW.
     `m=16` es cuantas conexiones tiene cada nodo. `ef_construct=100` es la precision
     al construir el indice.

Piensalo como un `CREATE TABLE IF NOT EXISTS`, pero en vez de columnas defines las
dimensiones del vector y como se busca en ellos.

---

## 7. insert_vectors() - El INSERT INTO

```python
def insert_vectors(self, ids: list[str], vectors: list[list[float]],
                   metadata: list[dict[str, Any]]) -> None:
```

### Flujo

```
insert_vectors(ids=["uuid1"], vectors=[[0.1, 0.2, ...]], metadata=[{"text": "hello"}])
    |
    +-- Validar que las 3 listas tengan el mismo tamano
    |   └── Si no → ValueError (como IllegalArgumentException)
    |
    +-- Validar que no sea lista vacia
    |
    +-- Validar que cada vector tenga 384 dimensiones
    |   └── Si no → ValueError
    |
    +-- Crear PointStruct por cada documento:
    |   PointStruct(id="uuid1", vector=[0.1, 0.2, ...], payload={"text": "hello"})
    |
    +-- client.upsert(points)   ← INSERT OR UPDATE
```

### zip() - Recorrer listas en paralelo

Python usa `zip` para recorrer las 3 listas a la vez:

```python
points = [
    PointStruct(id=id_, vector=vector, payload=meta)
    for id_, vector, meta in zip(ids, vectors, metadata)
]
```

`zip(ids, vectors, metadata)` junta las 3 listas posicion a posicion:
```
ids      = ["uuid1",        "uuid2"       ]
vectors  = [[0.1, 0.2...],  [0.3, 0.4...] ]
metadata = [{"text": "a"},  {"text": "b"} ]
                 ↓                ↓
zip produce: ("uuid1", [0.1,0.2...], {"text":"a"}),
             ("uuid2", [0.3,0.4...], {"text":"b"})
```

En Java harias:

```java
List<PointStruct> points = new ArrayList<>();
for (int i = 0; i < ids.size(); i++) {
    points.add(new PointStruct(ids.get(i), vectors.get(i), metadata.get(i)));
}
```

### enumerate() - Indice + valor

Para validar dimensiones, se usa `enumerate`:

```python
for i, vector in enumerate(vectors):
    if len(vector) != self.vector_size:
        raise ValueError(f"Vector at index {i} has {len(vector)} dimensions")
```

`enumerate` da el indice y el valor a la vez. En Java:

```java
for (int i = 0; i < vectors.size(); i++) {
    if (vectors.get(i).length != this.vectorSize) {
        throw new IllegalArgumentException("Vector at index " + i + "...");
    }
}
```

### Separacion de errores

Las `ValueError` de validacion estan **fuera** del try/catch. Las excepciones de
Qdrant estan **dentro** y se re-lanzan como `RuntimeError`. Esto es intencionado:

- **ValueError** = error del que llama (datos incorrectos)
- **RuntimeError** = error del sistema (Qdrant fallo)

En Java seria la diferencia entre `IllegalArgumentException` (unchecked, culpa del caller)
y `RuntimeException` wrapping una `IOException` (error de infraestructura).

---

## 8. search_similar() - La busqueda por similitud

```python
def search_similar(self, query_vector: list[float], limit: int = 5,
                   score_threshold: Optional[float] = None) -> list[dict[str, Any]]:
```

### Flujo

```
search_similar(query_vector=[0.1, 0.2, ...], limit=5, score_threshold=0.7)
    |
    +-- Validar dimension del query_vector (debe ser 384)
    |
    +-- client.query_points(
    |       query = [0.1, 0.2, ...],    ← "WHERE similar to this"
    |       limit = 5,                   ← "LIMIT 5"
    |       score_threshold = 0.7        ← "WHERE score > 0.7"
    |   ).points
    |
    +-- Transformar resultados a lista de dicts
```

En SQL seria algo como:
```sql
SELECT id, similarity(vector, query) as score, metadata
FROM documents
WHERE similarity(vector, query) > 0.7
ORDER BY score DESC
LIMIT 5;
```

Pero en vez de comparacion exacta (`WHERE name = 'algo'`), Qdrant calcula la
**similitud coseno** entre el vector de la query y todos los vectores almacenados.

### La transformacion de resultados

```python
return [
    {
        "id": str(point.id),
        "score": point.score,
        "metadata": point.payload or {},
    }
    for point in results
]
```

Cada `point` de Qdrant tiene `.id`, `.score` y `.payload`. Lo convertimos a un
diccionario simple. El `or {}` es un safety check: si `payload` es `None`, devuelve
un dict vacio.

En Java seria:

```java
return results.stream()
    .map(point -> Map.of(
        "id", point.getId().toString(),
        "score", point.getScore(),
        "metadata", point.getPayload() != null ? point.getPayload() : Map.of()
    ))
    .collect(Collectors.toList());
```

### `.points` al final

```python
results = self.client.query_points(...).points
```

`query_points()` devuelve un objeto con varios campos, y nosotros solo queremos
la lista de resultados. El `.points` extrae esa lista. En Java seria como:

```java
List<ScoredPoint> results = client.queryPoints(...).getPoints();
```

---

## 9. delete_by_id() - Borrar vectores

```python
def delete_by_id(self, ids: list[str]) -> None:
    if not ids:
        raise ValueError("Cannot delete empty list of IDs")

    from qdrant_client.models import PointIdsList

    self.client.delete(
        collection_name=self.collection_name,
        points_selector=PointIdsList(points=ids),
    )
```

Como `DELETE FROM tabla WHERE id IN (...)` en SQL.

### Import local

El import de `PointIdsList` esta **dentro** de la funcion, no arriba del fichero.
En Python es valido hacer imports dentro de funciones para no cargar cosas innecesarias
al importar el modulo. En Java no puedes hacer esto (los imports siempre van arriba).

`PointIdsList` envuelve la lista de IDs en el formato que Qdrant espera. Qdrant tiene
varios tipos de selectores (por ID, por filtro, por condiciones), y `PointIdsList`
es el mas simple: "borra estos IDs concretos".

---

## 10. get_collection_info() - Info de la coleccion

```python
def get_collection_info(self) -> dict[str, Any]:
    info = self.client.get_collection(self.collection_name)

    return {
        "collection_name": self.collection_name,
        "vectors_count": info.points_count,
        "vector_size": self.vector_size,
        "distance_metric": DISTANCE_METRIC,
        "status": str(info.status),
    }
```

Equivale a `DESCRIBE TABLE` + `SELECT COUNT(*)`. Devuelve metadatos de la coleccion:
nombre, cuantos vectores hay, dimensiones, tipo de distancia, estado.

---

## 11. El \_\_init\_\_.py - La fachada publica

```python
from .qdrant_database import QdrantDatabase
from .vector_database import VectorDatabase
__all__ = ["QdrantDatabase", "VectorDatabase"]
```

Define **que se exporta** del paquete `database`. Cuando otro modulo hace:

```python
from src.database import QdrantDatabase
```

Funciona porque `__init__.py` lo re-exporta. Sin esto tendrias que escribir:

```python
from src.database.qdrant_database import QdrantDatabase  # Largo
```

En Java seria como el `module-info.java` o definir que clases son `public`.

---

## 12. Tests: unit vs integration

En M3 reorganizamos los tests en dos carpetas:

```
tests/
  unit/                              # 19 tests (rapidos, datos inventados)
    test_vector_database.py
  integration/                       # 7 tests (con modelo real de M2)
    test_vector_db_integration.py
```

### Unit tests (tests/unit/)

- Usan vectores inventados como `[0.5] * 384`
- No cargan el modelo de embeddings
- Tardan < 1 segundo
- Verifican que **el codigo funciona** correctamente

En Java: los tests de JUnit que usan mocks.

### Integration tests (tests/integration/)

- Generan embeddings reales con sentence-transformers
- Hacen busquedas semanticas de verdad
- Tardan ~15 segundos (carga el modelo)
- Verifican que **M2 + M3 funcionan juntos**

En Java: los tests con `@SpringBootTest` que levantan la aplicacion real.

### Fixtures en pytest

```python
@pytest.fixture(scope="class")
def db(self) -> QdrantDatabase:
    return QdrantDatabase(collection_name="test_collection", in_memory=True)

@pytest.fixture()
def fresh_db(self) -> QdrantDatabase:
    return QdrantDatabase(collection_name="fresh_test_collection", in_memory=True)
```

- `db` con `scope="class"` → se crea **una vez** para toda la clase. Como `@BeforeAll`.
  Se reutiliza entre tests (puede acumular datos).
- `fresh_db` sin scope → se crea **nueva** para cada test. Como `@BeforeEach`.
  Siempre empieza vacia.

En Java:

```java
private static QdrantDatabase db;          // @BeforeAll

@BeforeAll
static void setUpClass() {
    db = new QdrantDatabase("test_collection", true);
}

@BeforeEach
void setUp() {
    freshDb = new QdrantDatabase("fresh_test_collection", true);
}
```

### UUIDs para IDs

Qdrant requiere que los IDs string sean UUIDs validos. Por eso creamos un helper:

```python
def _uuid() -> str:
    return str(uuid4())  # "a8f3b2c1-..."
```

Y en los tests usamos `_uuid()` en vez de IDs inventados como `"doc_001"`.

### Test de similitud coseno

```python
similar_vector = [0.5] * 384
different_vector = [-0.5] * 384
medium_vector = [0.5] * 192 + [-0.5] * 192  # Mixto
```

**Punto importante:** La similitud coseno mide **direccion**, no magnitud.
`[0.5]*384` y `[0.3]*384` apuntan en la **misma direccion** (cosine = 1.0),
aunque tengan magnitudes diferentes. Para que sean "diferentes" en coseno,
los vectores deben apuntar en **direcciones distintas** (valores positivos vs negativos).

---

## 13. Flujo completo M2 + M3

```
Usuario: "What is machine learning?"
    |
    |  (1) EmbeddingService.generate_embedding("What is machine learning?")
    |     └── model.encode() → [0.12, -0.45, 0.33, ...]  (384 floats)
    |
    |  (2) QdrantDatabase.search_similar(query_vector=[0.12, -0.45, ...], limit=3)
    |     └── Qdrant compara con TODOS los vectores almacenados
    |     └── Devuelve los 3 mas parecidos, ordenados por score
    |
    |  (3) Resultado:
    |     [
    |       {score: 0.95, metadata: {text: "Machine learning is a branch of AI..."}},
    |       {score: 0.87, metadata: {text: "Neural networks are computing systems..."}},
    |       {score: 0.72, metadata: {text: "Python is a programming language..."}},
    |     ]
    v
```

Lo que falta (M4-M7): leer documentos reales → trocearlos → indexarlos → pasarle
los resultados a un LLM para que genere una respuesta en lenguaje natural.

---

## 14. Tabla resumen Java vs Python en M3

| Concepto | Java | Python (M3) |
|----------|------|-------------|
| Interfaz | `interface VectorDatabase` | `class VectorDatabase(ABC)` |
| Implementacion | `class QdrantDB implements VectorDatabase` | `class QdrantDatabase(VectorDatabase)` |
| Constantes | `public static final int VECTOR_SIZE = 384` | `VECTOR_SIZE = 384` |
| Constructor | `public QdrantDatabase(...)` | `def __init__(self, ...)` |
| Atributos | `this.client` | `self.client` |
| Null safety | `@Nullable Float threshold` | `Optional[float] = None` |
| Iterar 3 listas | `for (int i = 0; i < n; i++)` | `for id_, vec, meta in zip(...)` |
| Iterar con indice | `for (int i = 0; i < n; i++)` | `for i, vector in enumerate(vectors)` |
| Excepciones | `throw new IllegalArgumentException` | `raise ValueError(...)` |
| Re-throw con causa | `throw new RuntimeException(msg, e)` | `raise RuntimeError(msg) from e` |
| Paquete publico | `module-info.java` / public classes | `__init__.py` con `__all__` |
| Fixture de test | `@BeforeAll` / `@BeforeEach` | `@pytest.fixture(scope="class")` / `@pytest.fixture()` |
| Import local | No posible | Valido dentro de funciones |
| CREATE IF NOT EXISTS | `IF NOT EXISTS` en SQL | `any(...)` + `return` temprano |

---

**Archivos documentados:**
- [src/database/config.py](../../src/database/config.py) - Constantes de configuracion
- [src/database/vector_database.py](../../src/database/vector_database.py) - Interfaz abstracta
- [src/database/qdrant_database.py](../../src/database/qdrant_database.py) - Implementacion Qdrant
- [src/database/\_\_init\_\_.py](../../src/database/__init__.py) - Exportaciones del paquete
- [tests/unit/test_vector_database.py](../../tests/unit/test_vector_database.py) - Tests unitarios
- [tests/integration/test_vector_db_integration.py](../../tests/integration/test_vector_db_integration.py) - Tests de integracion
