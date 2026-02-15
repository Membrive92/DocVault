# Guia Interna - Milestone 1: Foundation (Estructura del Proyecto)

> Guia escrita para desarrolladores con experiencia en Java.
> Cada concepto de Python se compara con su equivalente en Java.

---

## Indice

1. [Que hemos construido](#1-que-hemos-construido)
2. [Estructura del proyecto - Python vs Maven/Gradle](#2-estructura-del-proyecto---python-vs-mavengradle)
3. [requirements.txt - El pom.xml de Python](#3-requirementstxt---el-pomxml-de-python)
4. [Entorno virtual (venv) - El classpath aislado](#4-entorno-virtual-venv---el-classpath-aislado)
5. [\_\_init\_\_.py - Definir paquetes en Python](#5-__init__py---definir-paquetes-en-python)
6. [config/settings.py - El application.properties con superpoderes](#6-configsettingspy---el-applicationproperties-con-superpoderes)
7. [Pydantic Settings vs Spring @ConfigurationProperties](#7-pydantic-settings-vs-spring-configurationproperties)
8. [Field() - Las anotaciones de configuracion](#8-field---las-anotaciones-de-configuracion)
9. [pathlib.Path - El java.nio.file.Path de Python](#9-pathlibpath---el-javaniofilepath-de-python)
10. [.env y .env.example - Variables de entorno](#10-env-y-envexample---variables-de-entorno)
11. [El singleton settings - Instancia global](#11-el-singleton-settings---instancia-global)
12. [Metodos utilitarios de Settings](#12-metodos-utilitarios-de-settings)
13. [test_setup.py - Script de verificacion](#13-test_setuppy---script-de-verificacion)
14. [.gitignore - Que ignorar en Python vs Java](#14-gitignore---que-ignorar-en-python-vs-java)
15. [Tabla resumen Java vs Python en M1](#15-tabla-resumen-java-vs-python-en-m1)

---

## 1. Que hemos construido

Milestone 1 es la **fundacion** del proyecto. No tiene logica de negocio todavia — es como
cuando en Java creas un proyecto Maven nuevo con `mvn archetype:generate` y configuras
el `pom.xml`, la estructura de carpetas, y las properties antes de escribir una sola clase.

Lo que hemos creado:

- **Estructura de carpetas** organizada por modulos
- **Sistema de configuracion** con tipos validados (como `@ConfigurationProperties` de Spring)
- **Gestion de variables de entorno** (como el `.properties` pero con `.env`)
- **Script de verificacion** para comprobar que todo esta bien montado
- **Dependencias** instaladas y documentadas

Piensa en M1 como el `mvn archetype:generate` + configurar Spring Boot antes de escribir
tu primer `@Controller`.

---

## 2. Estructura del proyecto - Python vs Maven/Gradle

### En Java (Maven)

```
mi-proyecto/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/docvault/
│   │   │       ├── Application.java
│   │   │       └── config/
│   │   │           └── AppConfig.java
│   │   └── resources/
│   │       ├── application.properties
│   │       └── application-dev.properties
│   └── test/
│       └── java/
│           └── com/docvault/
│               └── AppConfigTest.java
├── pom.xml
├── .gitignore
└── README.md
```

En Maven hay una **convencion rigida**: `src/main/java/`, `src/test/java/`, `src/main/resources/`.
Si no sigues esta estructura, Maven no compila.

### En Python (nuestro proyecto)

```
DocVault/
├── config/                  ← Configuracion (como resources/)
│   ├── __init__.py
│   └── settings.py          ← El "application.properties" con logica
├── src/                     ← Codigo fuente (como src/main/java/)
│   ├── __init__.py
│   ├── embeddings/          ← M2: Servicio de embeddings
│   └── database/            ← M3: Base de datos vectorial
├── tests/                   ← Tests (como src/test/java/)
│   ├── unit/                ← Tests rapidos con datos simulados
│   └── integration/         ← Tests lentos con modelos reales
├── data/                    ← Datos del proyecto
│   ├── documents/           ← Documentos a ingestar
│   └── qdrant_storage/      ← Storage de Qdrant
├── docs/                    ← Documentacion
├── .env.example             ← Template de variables de entorno
├── .env                     ← Variables reales (NO se sube a git)
├── .gitignore
├── requirements.txt         ← Dependencias (como pom.xml)
├── test_setup.py            ← Script de verificacion M1
├── README.md
└── AGENTS.md                ← Guia para agentes AI
```

**Diferencia clave:** En Python no hay convencion obligatoria. Tu decides la estructura.
Nosotros elegimos `src/` para codigo, `config/` para configuracion, y `tests/` para tests.
Es una convencion comun, pero no esta impuesta por ninguna herramienta.

---

## 3. requirements.txt - El pom.xml de Python

### En Java (pom.xml)

```xml
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter</artifactId>
        <version>3.2.0</version>
    </dependency>
    <dependency>
        <groupId>javax.validation</groupId>
        <artifactId>validation-api</artifactId>
        <version>2.0.1</version>
    </dependency>
</dependencies>
```

Maven resuelve dependencias transitivas automaticamente, descarga JARs al `.m2/repository`,
y compila todo junto.

### En Python (requirements.txt)

```
pydantic==2.11.7
pydantic-settings==2.10.1
python-dotenv==1.1.1
PyYAML==6.0.2
pytest==9.0.2
sentence-transformers==5.2.2
qdrant-client==1.16.2
```

Para instalar:
```bash
pip install -r requirements.txt
```

**Diferencias importantes:**

| Aspecto | Maven (pom.xml) | Python (requirements.txt) |
|---------|-----------------|---------------------------|
| Formato | XML estructurado | Texto plano, una linea por paquete |
| Versiones | `<version>3.2.0</version>` | `pydantic==2.11.7` |
| Transitivas | Resuelve automaticamente | `pip freeze` captura todo |
| Repositorio | Maven Central | PyPI (pip) |
| Lock file | No nativo (Gradle si) | `pip freeze > requirements.txt` |
| Scopes (test, compile) | `<scope>test</scope>` | No hay — todo se instala igual |

**Nota:** Nuestro `requirements.txt` fue generado con `pip freeze`, que lista TODAS las
dependencias instaladas (directas + transitivas). Es como si hicieras `mvn dependency:list`
y guardaras el resultado. Por eso ves ~90 paquetes cuando solo instalamos ~5 directamente.

---

## 4. Entorno virtual (venv) - El classpath aislado

### El problema

En Java, cada proyecto tiene su propio classpath con sus JARs. Si el Proyecto A usa
Spring 3.2 y el Proyecto B usa Spring 2.7, no hay conflicto — cada uno tiene sus JARs.

En Python, `pip install` instala paquetes **globalmente** por defecto. Si el Proyecto A
necesita `pydantic==2.0` y el Proyecto B necesita `pydantic==1.0`, hay conflicto.

### La solucion: venv

```bash
# Crear entorno virtual (como crear un "classpath" aislado)
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (Linux/Mac)
source venv/bin/activate

# Ahora pip instala SOLO en este entorno
pip install -r requirements.txt

# Desactivar
deactivate
```

### Comparacion

```
Java:                                Python:
┌─────────────┐                     ┌─────────────┐
│  Proyecto A │                     │  Proyecto A │
│  classpath:  │                     │  venv/:      │
│   spring-3.2│                     │   pydantic-2 │
│   guava-31  │                     │   torch-2.10 │
└─────────────┘                     └─────────────┘

┌─────────────┐                     ┌─────────────┐
│  Proyecto B │                     │  Proyecto B │
│  classpath:  │                     │  venv/:      │
│   spring-2.7│                     │   pydantic-1 │
│   guava-30  │                     │   flask-3    │
└─────────────┘                     └─────────────┘
```

**En Java** la separacion es automatica (cada JAR en su classpath).
**En Python** debes crear el `venv` explicitamente. Si no lo haces, todo se instala global
y puedes tener conflictos entre proyectos.

El directorio `venv/` esta en `.gitignore` — cada desarrollador crea el suyo.

---

## 5. __init__.py - Definir paquetes en Python

### En Java

En Java, un paquete se define por la **ruta de carpetas** + la declaracion `package`:

```java
// Archivo: src/main/java/com/docvault/config/AppConfig.java
package com.docvault.config;  // ← Esto define el paquete

public class AppConfig { ... }
```

No necesitas ningun archivo especial para que una carpeta sea un paquete.

### En Python

En Python, una carpeta **solo es un paquete** si contiene un archivo `__init__.py`:

```
config/
├── __init__.py      ← SIN esto, Python NO reconoce "config" como paquete
└── settings.py
```

Sin `__init__.py`, hacer `from config.settings import settings` daria un error de import.

### Nuestros __init__.py en M1

Son minimalistas — solo marcan la carpeta como paquete:

```python
# config/__init__.py
# Config package

# src/__init__.py
# Source package
```

En milestones posteriores (M2, M3), los `__init__.py` hacen mas — exportan clases
como una "fachada publica" (similar a `module-info.java` de Java 9):

```python
# src/embeddings/__init__.py (M2)
from src.embeddings.embedding_service import EmbeddingService

__all__ = ["EmbeddingService"]  # Solo se exporta esto
```

### Equivalencia

| Java | Python |
|------|--------|
| Carpeta con archivos `.java` | Carpeta con `__init__.py` |
| `package com.docvault.config;` | Tener `__init__.py` en la carpeta |
| `module-info.java` (Java 9+) | `__init__.py` con `__all__` |

---

## 6. config/settings.py - El application.properties con superpoderes

Este es el archivo mas importante de M1. En Java, la configuracion tipicamente se hace con:

1. **application.properties** — Valores clave-valor
2. **@ConfigurationProperties** — Clase Java que mapea las properties a campos tipados
3. **Spring Environment** — Lee de properties, env vars, profiles...

En Python, `config/settings.py` hace **todo eso en un solo archivo**.

### En Java (Spring Boot)

```properties
# application.properties
project.name=docvault
environment=development
log.level=INFO
data.dir=data
documents.dir=data/documents
```

```java
@ConfigurationProperties(prefix = "project")
public class ProjectConfig {
    private String name = "docvault";
    private String environment = "development";
    private String logLevel = "INFO";
    private Path dataDir = Path.of("data");
    private Path documentsDir = Path.of("data/documents");

    // getters, setters...
}
```

### En Python (nuestro settings.py)

```python
class Settings(BaseSettings):
    project_name: str = Field(default="docvault", description="Project name")
    environment: Literal["development", "production", "testing"] = Field(default="development")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    data_dir: Path = Field(default=Path("data"))
    documents_dir: Path = Field(default=Path("data/documents"))

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )
```

### Que hace Pydantic Settings por nosotros

1. **Lee variables de entorno** automaticamente (como Spring `@Value("${PROJECT_NAME}")`)
2. **Lee del archivo .env** (como Spring con `application.properties`)
3. **Valida tipos** automaticamente (string, int, bool, Path...)
4. **Valida valores** con `Literal` (solo acepta "development", "production", "testing")
5. **Tiene defaults** para cuando no hay variable de entorno ni `.env`

### Prioridad de carga (igual que Spring)

```
1. Variable de entorno del sistema     ← Maxima prioridad (como -D en Java)
2. Archivo .env                        ← Media prioridad (como application.properties)
3. Valor default en Field()            ← Minima prioridad (como @Value default)
```

Ejemplo:
```bash
# Si en .env tienes:
PROJECT_NAME=mi_proyecto

# Y en el sistema:
set PROJECT_NAME=override

# Resultado:
settings.project_name  # → "override" (la variable de sistema gana)
```

Esto es identico a como Spring prioriza: env vars > properties > defaults.

---

## 7. Pydantic Settings vs Spring @ConfigurationProperties

Veamos la clase completa comparada linea a linea:

### El import

```python
# Python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import Literal
```

```java
// Java
import org.springframework.boot.context.properties.ConfigurationProperties;
import jakarta.validation.constraints.NotNull;
import java.nio.file.Path;
```

### La clase

```python
# Python
class Settings(BaseSettings):       # ← Hereda de BaseSettings
```

```java
// Java
@ConfigurationProperties            // ← Anotacion en la clase
public class AppSettings {
```

En Java usas una **anotacion** (`@ConfigurationProperties`) para indicar que la clase
lee de properties. En Python **heredas** de `BaseSettings` — el comportamiento viene
por herencia, no por anotacion.

### Los campos

```python
# Python
project_name: str = Field(default="docvault", description="Project name")
environment: Literal["development", "production", "testing"] = Field(default="development")
log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
```

```java
// Java
@NotNull
private String projectName = "docvault";

@Pattern(regexp = "development|production|testing")
private String environment = "development";

@Pattern(regexp = "DEBUG|INFO|WARNING|ERROR")
private String logLevel = "INFO";

// + getters + setters (o usar Lombok @Data)
```

**Diferencias:**

| Aspecto | Java | Python |
|---------|------|--------|
| Tipo | Declarado con tipo Java | Type hint `: str` |
| Default | Asignacion directa `= "docvault"` | `Field(default="docvault")` |
| Validacion | `@Pattern`, `@NotNull`, etc. | `Literal[...]` restringe valores |
| Getters/Setters | Necesarios (o Lombok) | No necesarios — acceso directo |
| Inmutabilidad | `final` o `@Value` de Lombok | Los campos son mutables por defecto |

### La configuracion del "lector"

```python
# Python — DENTRO de la clase
model_config = SettingsConfigDict(
    env_file=".env",              # ← Lee de .env
    env_file_encoding="utf-8",
    case_sensitive=False,          # ← PROJECT_NAME == project_name
    extra="ignore"                 # ← Ignora variables extra en .env
)
```

```java
// Java — en application.properties o con anotacion
@ConfigurationProperties(prefix = "app")
// + spring.config.location=classpath:application.properties
```

En Python, la configuracion de "de donde leer" esta **dentro de la propia clase** como
un atributo especial `model_config`. En Spring, esto se configura externamente (en
`application.properties`, perfiles, `@PropertySource`, etc.).

---

## 8. Field() - Las anotaciones de configuracion

`Field()` en Pydantic es como las anotaciones de Bean Validation en Java:

```python
# Python
project_root: Path = Field(
    default_factory=lambda: Path(__file__).parent.parent,
    description="Project root (calculated automatically)"
)
```

```java
// Java
@Value("#{T(java.nio.file.Paths).get(systemProperties['user.dir'])}")
private Path projectRoot;
```

### Parametros de Field()

| Parametro Field() | Equivalente Java | Que hace |
|--------------------|-----------------|----------|
| `default="valor"` | `= "valor"` | Valor por defecto estatico |
| `default_factory=lambda: ...` | `@PostConstruct init()` | Valor por defecto calculado |
| `description="..."` | JavaDoc / `@Schema` | Documentacion del campo |
| `min_length=1` | `@Size(min=1)` | Validacion de longitud minima |
| `gt=0` | `@Positive` | Mayor que 0 |

### El caso especial: default_factory

```python
project_root: Path = Field(
    default_factory=lambda: Path(__file__).parent.parent
)
```

`default_factory` se usa cuando el valor por defecto **necesita calcularse** en tiempo
de ejecucion. Aqui, `Path(__file__).parent.parent` significa:

- `__file__` → ruta del archivo actual (`config/settings.py`)
- `.parent` → carpeta padre (`config/`)
- `.parent` → carpeta padre otra vez (`DocVault/` — la raiz del proyecto)

En Java esto seria algo como:

```java
@PostConstruct
void init() {
    if (this.projectRoot == null) {
        this.projectRoot = Paths.get(System.getProperty("user.dir"));
    }
}
```

La diferencia es que en Python `lambda` permite definirlo inline, mientras en Java
necesitas un metodo separado con `@PostConstruct`.

---

## 9. pathlib.Path - El java.nio.file.Path de Python

Python tiene su propio `Path`, muy similar al de Java NIO:

### Crear paths

```python
# Python
from pathlib import Path

path = Path("data/documents")
absolute = Path(__file__).parent.parent
```

```java
// Java
import java.nio.file.Path;
import java.nio.file.Paths;

Path path = Paths.get("data", "documents");
Path absolute = Paths.get(System.getProperty("user.dir"));
```

### Concatenar paths (el operador /)

```python
# Python — usa el operador / (muy elegante)
full_path = settings.project_root / settings.data_dir
# Equivale a: /Users/me/DocVault/data
```

```java
// Java — usa .resolve()
Path fullPath = projectRoot.resolve(dataDir);
// Equivale a: /Users/me/DocVault/data
```

El operador `/` en Python es **azucar sintactico** para `Path.resolve()`. Hace lo mismo
que en Java, pero se lee como una ruta de archivo.

### Operaciones comunes

| Operacion | Python | Java |
|-----------|--------|------|
| Crear path | `Path("data")` | `Paths.get("data")` |
| Concatenar | `path / "file.txt"` | `path.resolve("file.txt")` |
| Padre | `path.parent` | `path.getParent()` |
| Nombre | `path.name` | `path.getFileName().toString()` |
| Existe? | `path.exists()` | `Files.exists(path)` |
| Es absoluto? | `path.is_absolute()` | `path.isAbsolute()` |
| Crear dirs | `path.mkdir(parents=True, exist_ok=True)` | `Files.createDirectories(path)` |

### En nuestro settings.py

```python
def get_full_path(self, relative_path: Path) -> Path:
    if relative_path.is_absolute():    # ← Como path.isAbsolute() en Java
        return relative_path
    return self.project_root / relative_path  # ← Como projectRoot.resolve(path)
```

---

## 10. .env y .env.example - Variables de entorno

### El concepto

En Java con Spring Boot, la configuracion va en `application.properties`:

```properties
# application.properties (se sube a git)
project.name=docvault
environment=development

# application-secret.properties (NO se sube a git)
api.key=sk-12345
```

En Python, el patron es usar archivos `.env`:

```bash
# .env.example (se sube a git — es el template)
PROJECT_NAME=docvault
ENVIRONMENT=development
# OPENAI_API_KEY=sk-...   ← Comentado, el dev pone el suyo

# .env (NO se sube a git — tiene valores reales)
PROJECT_NAME=docvault
ENVIRONMENT=development
OPENAI_API_KEY=sk-mi-clave-real
```

### Como funciona la carga

```python
# settings.py
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",           # ← Pydantic lee este archivo automaticamente
        case_sensitive=False        # ← PROJECT_NAME mapea a project_name
    )
```

Pydantic Settings automaticamente:
1. Busca el archivo `.env` en el directorio actual
2. Lee cada linea `CLAVE=VALOR`
3. Mapea `PROJECT_NAME` al campo `project_name` (case insensitive)
4. Convierte el tipo: si el campo es `bool`, convierte "False" → `False`

### Nuestro .env.example

```bash
# General Configuration
PROJECT_NAME=docvault
ENVIRONMENT=development
LOG_LEVEL=INFO

# Data Paths
DATA_DIR=./data
DOCUMENTS_DIR=./data/documents

# Qdrant Vector Database (añadido en M3)
QDRANT_COLLECTION_NAME=docvault_documents
QDRANT_STORAGE_PATH=./data/qdrant_storage
QDRANT_IN_MEMORY=False

# Futuras expansiones (comentadas)
# OPENAI_API_KEY=sk-...
# OLLAMA_BASE_URL=http://localhost:11434
```

### Equivalencia con Spring

| Concepto | Spring | Python |
|----------|--------|--------|
| Config por defecto | `application.properties` | `.env.example` (template) |
| Config local | `application-local.properties` | `.env` (no en git) |
| Secretos | `application-secret.properties` o Vault | `.env` (no en git) |
| Profiles | `spring.profiles.active=dev` | `ENVIRONMENT=development` en `.env` |

---

## 11. El singleton settings - Instancia global

### En Java (Spring)

Spring maneja singletons con `@Component` + inyeccion de dependencias:

```java
@Component
@ConfigurationProperties
public class AppSettings {
    private String projectName = "docvault";
    // ...
}

// En otro archivo — Spring inyecta el singleton
@Service
public class MyService {
    @Autowired
    private AppSettings settings;  // ← Spring crea UNA instancia y la inyecta
}
```

### En Python (nuestro enfoque)

No hay framework de inyeccion de dependencias. El singleton se crea manualmente:

```python
# config/settings.py

class Settings(BaseSettings):
    project_name: str = Field(default="docvault")
    # ... campos ...

# Instancia global — SE CREA AL IMPORTAR EL MODULO
settings = Settings()
```

```python
# En cualquier otro archivo
from config.settings import settings

print(settings.project_name)  # → "docvault"
```

### Como funciona el singleton

Cuando Python ejecuta `from config.settings import settings`, hace:

1. **Ejecuta** `config/settings.py` completo (solo la primera vez)
2. **Crea** la instancia `settings = Settings()` (carga .env, valida, etc.)
3. **Cachea** el modulo — futuras importaciones usan la misma instancia

Esto es un **singleton de modulo** — Python garantiza que un modulo solo se ejecuta
una vez. Todas las importaciones posteriores reusan el objeto ya creado.

```python
# archivo_a.py
from config.settings import settings
id(settings)  # → 140234567890

# archivo_b.py
from config.settings import settings
id(settings)  # → 140234567890  ← MISMA instancia!
```

En Java esto seria como un `static final`:

```java
public class SettingsHolder {
    public static final AppSettings SETTINGS = new AppSettings();
}
```

### La funcion get_settings()

Tambien tenemos una funcion helper para testing:

```python
def get_settings() -> Settings:
    """Get configuration instance. Useful for dependency injection in testing."""
    return settings
```

Esto permite mockear la configuracion en tests, similar a cuando en Java usas
`@MockBean` para reemplazar un `@Component` en tests.

---

## 12. Metodos utilitarios de Settings

La clase `Settings` no solo tiene campos — tiene metodos utilitarios.

### get_full_path() - Resolver rutas

```python
def get_full_path(self, relative_path: Path) -> Path:
    if relative_path.is_absolute():
        return relative_path
    return self.project_root / relative_path
```

En Java seria:

```java
public Path getFullPath(Path relativePath) {
    if (relativePath.isAbsolute()) {
        return relativePath;
    }
    return this.projectRoot.resolve(relativePath);
}
```

Ejemplo de uso:
```python
settings.get_full_path(Path("data"))
# → C:\Users\membr\Desktop\Projects\DocVault\data

settings.get_full_path(Path("C:/absolute/path"))
# → C:\absolute\path  (ya es absoluta, se devuelve tal cual)
```

### ensure_directories() - Crear directorios

```python
def ensure_directories(self) -> None:
    directories = [
        self.get_full_path(self.data_dir),
        self.get_full_path(self.documents_dir),
        self.get_full_path(self.qdrant_storage_path),
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
```

En Java:

```java
public void ensureDirectories() throws IOException {
    List<Path> directories = List.of(
        getFullPath(dataDir),
        getFullPath(documentsDir),
        getFullPath(qdrantStoragePath)
    );
    for (Path dir : directories) {
        Files.createDirectories(dir);  // ← Crea padres automaticamente
    }
}
```

**Parametros de mkdir:**
- `parents=True` → Como `Files.createDirectories()` (crea padres intermedios)
- `exist_ok=True` → No lanza error si ya existe

Sin `exist_ok=True`, Python lanzaria `FileExistsError` si el directorio ya existe.
En Java, `Files.createDirectories()` ya ignora directorios existentes por defecto.

### display_config() - Mostrar configuracion

```python
def display_config(self) -> None:
    print(f"Environment:      {self.environment}")
    print(f"Log Level:        {self.log_level}")
    print(f"Project Root:     {self.project_root}")
    # ...
```

Esto no tiene equivalente directo en Java — seria como un `toString()` pero formateado
para la consola. En Spring Boot, puedes habilitar logging de properties con
`logging.level.org.springframework.boot.context.properties=DEBUG`, pero no es tan directo.

---

## 13. test_setup.py - Script de verificacion

Este script verifica que M1 esta bien configurado. Es como un "smoke test" manual.

### Estructura del script

```python
def main():
    tests = [
        ("Imports", test_imports),          # ← Verificar librerias instaladas
        (".env File", test_env_file),       # ← Verificar que .env existe
        ("Configuration", test_config),      # ← Verificar que Settings carga bien
        ("Directories", test_directories)    # ← Verificar/crear directorios
    ]

    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
```

### Comparacion con Java

En Java, esto seria un test JUnit o un `CommandLineRunner` de Spring Boot:

```java
// Java — CommandLineRunner
@Component
public class SetupVerification implements CommandLineRunner {
    @Autowired
    private AppSettings settings;

    @Override
    public void run(String... args) {
        System.out.println("Verifying imports...");
        // verificar que las clases existen
        Class.forName("com.fasterxml.jackson.databind.ObjectMapper");

        System.out.println("Verifying configuration...");
        assert settings.getProjectName().equals("docvault");

        System.out.println("Verifying directories...");
        settings.ensureDirectories();
    }
}
```

### El truco de __import__()

```python
def test_imports():
    required_packages = {
        "pydantic": "Data validation",
        "pydantic_settings": "Configuration management",
        "dotenv": "Environment variables",
        "yaml": "YAML reading"
    }

    for package, description in required_packages.items():
        __import__(package)  # ← Import dinamico (como Class.forName())
```

`__import__(package)` es equivalente a `Class.forName("com.example.MyClass")` en Java —
intenta cargar el paquete dinamicamente. Si no esta instalado, lanza `ImportError`
(como `ClassNotFoundException` en Java).

### El fix de UTF-8 para Windows

```python
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

Windows usa CP1252 por defecto en la consola, no UTF-8. Esto causa que los emojis
(como los checkmarks del script) fallen. Este fix fuerza UTF-8.

En Java, el equivalente seria:
```java
System.setOut(new PrintStream(System.out, true, "UTF-8"));
```

### El patron if __name__ == "__main__"

```python
if __name__ == "__main__":
    sys.exit(main())
```

En Java:

```java
public class TestSetup {
    public static void main(String[] args) {
        int exitCode = new TestSetup().run();
        System.exit(exitCode);
    }
}
```

`__name__ == "__main__"` verifica si el archivo se esta ejecutando directamente
(no importado como modulo). Es el equivalente al metodo `public static void main()`.

Si alguien hiciera `from test_setup import test_imports`, la funcion `main()` NO se
ejecutaria — solo se importaria la funcion individual.

---

## 14. .gitignore - Que ignorar en Python vs Java

### Diferencias clave

| Que se ignora | Java | Python |
|---------------|------|--------|
| Compilados | `target/`, `*.class` | `__pycache__/`, `*.pyc` |
| Dependencias | (van en Maven Central) | `venv/`, `env/` |
| IDE | `.idea/`, `*.iml` | `.idea/`, `.vscode/` |
| Build | `target/`, `build/` | `build/`, `dist/`, `*.egg-info/` |
| Secretos | `application-secret.properties` | `.env` |

### Nuestro .gitignore explicado

```gitignore
# Python — equivalente a target/ y *.class
__pycache__/        # ← Bytecode compilado (como .class)
*.py[cod]           # ← .pyc, .pyo, .pyd
*.so                # ← Librerias compiladas C (como .dll/.so en JNI)
venv/               # ← Entorno virtual (como node_modules, no .m2)

# Environment — secretos
.env                # ← Variables con secretos (como application-secret.properties)

# Data — archivos grandes generados
data/documents/*    # ← Documentos ingested
data/qdrant_storage/ # ← Base de datos vectorial
!data/documents/.gitkeep  # ← Pero SI mantener el .gitkeep (para que la carpeta exista)

# Testing
.pytest_cache/      # ← Cache de pytest (como .gradle/)
.coverage           # ← Reporte de cobertura
htmlcov/            # ← Reporte HTML de cobertura (como jacoco/)
```

### El truco de .gitkeep

```gitignore
data/documents/*           # ← Ignora TODO dentro de data/documents/
!data/documents/.gitkeep   # ← EXCEPTO .gitkeep
```

Git no trackea carpetas vacias. Si quieres que `data/documents/` exista en el repo
(pero vacia), creas un archivo `.gitkeep` dentro. El `!` en gitignore es una negacion:
"ignora todo EXCEPTO este archivo".

En Java normalmente no necesitas esto porque Maven crea las carpetas automaticamente.

---

## 15. Tabla resumen Java vs Python en M1

| Concepto | Java | Python (nuestro proyecto) |
|----------|------|---------------------------|
| Estructura de proyecto | Maven conventions (`src/main/java/`) | Libre (elegimos `src/`, `config/`, `tests/`) |
| Gestor de dependencias | Maven/Gradle (`pom.xml`) | pip (`requirements.txt`) |
| Repositorio de paquetes | Maven Central | PyPI |
| Aislamiento de proyecto | Classpath por proyecto | `venv` (entorno virtual) |
| Definir paquete | Carpeta + `package` statement | Carpeta + `__init__.py` |
| Configuracion | `application.properties` + `@ConfigurationProperties` | `.env` + `BaseSettings` |
| Validacion de config | Bean Validation (`@NotNull`, `@Pattern`) | Pydantic (`Literal[...]`, `Field()`) |
| Variables de entorno | `System.getenv()` o Spring profiles | Pydantic lee automaticamente |
| Singleton de config | `@Component` + DI de Spring | Instancia global `settings = Settings()` |
| Rutas de archivos | `java.nio.file.Path` + `Files` | `pathlib.Path` |
| Concatenar rutas | `path.resolve("sub")` | `path / "sub"` |
| Crear directorios | `Files.createDirectories(path)` | `path.mkdir(parents=True, exist_ok=True)` |
| Main method | `public static void main(String[])` | `if __name__ == "__main__":` |
| Import dinamico | `Class.forName("com.example.Foo")` | `__import__("package_name")` |
| Archivos compilados | `.class` en `target/` | `.pyc` en `__pycache__/` |
| Ignorar en git | `target/`, `.idea/` | `__pycache__/`, `venv/`, `.env` |

---

## Resumen final

M1 establecio la base del proyecto. No hay logica de negocio — es pura infraestructura:

1. **Estructura de carpetas** → Como `mvn archetype:generate`
2. **requirements.txt** → Como `pom.xml` (dependencias)
3. **venv/** → Como el classpath aislado de cada proyecto Java
4. **config/settings.py** → Como `application.properties` + `@ConfigurationProperties`
5. **.env** → Como las properties de Spring con secretos
6. **test_setup.py** → Como un `CommandLineRunner` de verificacion

Con esta base, los milestones siguientes solo necesitan:
- Crear una nueva carpeta en `src/` (como crear un nuevo paquete Java)
- Agregar campos a `Settings` (como agregar properties)
- Agregar tests en `tests/` (como agregar tests JUnit)

La fundacion esta hecha. A partir de M2, empezamos con logica real.
