# Guia Interna - Milestone 6: Flexible LLM Layer

> Guia escrita para desarrolladores con experiencia en Java.
> Cada concepto de Python se compara con su equivalente en Java.

---

## Indice

1. [Que hemos construido](#1-que-hemos-construido)
2. [Strategy Pattern - tercera vez en el proyecto](#2-strategy-pattern---tercera-vez-en-el-proyecto)
3. [config.py - Enum con str mixin](#3-configpy---enum-con-str-mixin)
4. [base_provider.py - ABC con metodos concretos](#4-base_providerpy---abc-con-metodos-concretos)
5. [OllamaProvider - el cliente local](#5-ollamaprovider---el-cliente-local)
6. [OpenAIProvider - Chat Completions API](#6-openaiprovider---chat-completions-api)
7. [AnthropicProvider - Messages API y context manager](#7-anthropicprovider---messages-api-y-context-manager)
8. [LLMProviderFactory - lazy imports](#8-llmproviderfactory---lazy-imports)
9. [Streaming con generators (yield)](#9-streaming-con-generators-yield)
10. [format_prompt_with_context - el template RAG](#10-format_prompt_with_context---el-template-rag)
11. [config/settings.py - campos LLM con Pydantic](#11-configsettingspy---campos-llm-con-pydantic)
12. [Tests con MagicMock y @patch](#12-tests-con-magicmock-y-patch)
13. [Por que lazy import en la factory](#13-por-que-lazy-import-en-la-factory)
14. [Patron completo: de .env a respuesta LLM](#14-patron-completo-de-env-a-respuesta-llm)
15. [Tabla resumen Java vs Python en M6](#15-tabla-resumen-java-vs-python-en-m6)

---

## 1. Que hemos construido

En milestones anteriores construimos:
- **M2**: Texto → vectores (EmbeddingService)
- **M3**: Almacenar y buscar vectores (QdrantDatabase)
- **M4**: Archivos → texto limpio (parsers)
- **M5**: Pipeline completo de ingestion

Pero faltaba la pieza que **genera respuestas**: el LLM (Large Language Model).
El problema es que hay muchos proveedores (Ollama, OpenAI, Anthropic) y cada uno
tiene una API diferente.

En M6 hemos creado una **capa de abstraccion** que permite cambiar de proveedor
cambiando una sola variable de entorno:

```
.env: LLM_PROVIDER=ollama_local    →  usa Ollama gratis en tu maquina
.env: LLM_PROVIDER=openai          →  usa GPT-4 (de pago)
.env: LLM_PROVIDER=anthropic       →  usa Claude (de pago)
```

El codigo que llama al LLM no cambia — solo la configuracion.

---

## 2. Strategy Pattern - tercera vez en el proyecto

Este es el tercer modulo que usa el Strategy Pattern:

| Milestone | Interfaz | Implementaciones | Factory |
|-----------|----------|------------------|---------|
| **M3** | `VectorDatabase` | `QdrantDatabase` | (directa) |
| **M4** | `DocumentParser` | `PDFParser`, `HTMLParser`, `MarkdownParser` | `ParserFactory` |
| **M6** | `LLMProvider` | `OllamaProvider`, `OpenAIProvider`, `AnthropicProvider` | `LLMProviderFactory` |

En Java, el Strategy Pattern se ve asi:

```java
// Java — Strategy Pattern
public interface LLMProvider {
    String generate(String prompt, String context, double temperature, int maxTokens);
    Iterator<String> generateStream(String prompt, String context, ...);
    Map<String, String> getModelInfo();
}

public class OllamaProvider implements LLMProvider { ... }
public class OpenAIProvider implements LLMProvider { ... }
public class AnthropicProvider implements LLMProvider { ... }

// Factory
public class LLMProviderFactory {
    public static LLMProvider createProvider(String providerType) { ... }
}
```

En Python es identico conceptualmente, pero con `ABC` en lugar de `interface`:

```python
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt, context, temperature, max_tokens) -> str: ...
    @abstractmethod
    def generate_stream(self, prompt, context, ...) -> Iterator[str]: ...
    @abstractmethod
    def get_model_info(self) -> dict[str, str]: ...

class OllamaProvider(LLMProvider): ...
class OpenAIProvider(LLMProvider): ...
class AnthropicProvider(LLMProvider): ...
```

La diferencia principal: en Python una `ABC` puede tener metodos concretos
(como `format_prompt_with_context`), mientras que en Java una `interface`
solo puede tener `default` methods desde Java 8.

---

## 3. config.py - Enum con str mixin

### LLMProviderType(str, Enum)

```python
class LLMProviderType(str, Enum):
    OLLAMA_LOCAL = "ollama_local"
    OLLAMA_SERVER = "ollama_server"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
```

El truco esta en `(str, Enum)`. Esto hace que cada miembro del enum sea
**tambien un string**:

```python
LLMProviderType.OPENAI == "openai"          # True (gracias a str mixin)
LLMProviderType("openai")                    # → LLMProviderType.OPENAI
LLMProviderType("nonexistent")               # → ValueError
```

Sin el `str` mixin, tendrias que usar `.value`:

```python
class BadEnum(Enum):
    OPENAI = "openai"

BadEnum.OPENAI == "openai"       # False — compara objeto vs string
BadEnum.OPENAI.value == "openai" # True — accede al valor
```

En Java:

```java
public enum LLMProviderType {
    OLLAMA_LOCAL("ollama_local"),
    OLLAMA_SERVER("ollama_server"),
    OPENAI("openai"),
    ANTHROPIC("anthropic");

    private final String value;

    LLMProviderType(String value) { this.value = value; }
    public String getValue() { return value; }

    // Necesitas un metodo manual para buscar por valor
    public static LLMProviderType fromValue(String value) {
        for (LLMProviderType type : values()) {
            if (type.value.equals(value)) return type;
        }
        throw new IllegalArgumentException("Unknown: " + value);
    }
}
```

En Python `LLMProviderType("openai")` hace la busqueda automaticamente.
En Java necesitas un metodo `fromValue()` manual.

### DEFAULT_MODELS como dict

```python
DEFAULT_MODELS = {
    LLMProviderType.OLLAMA_LOCAL: "llama3.2:3b",
    LLMProviderType.OLLAMA_SERVER: "llama3.2:3b",
    LLMProviderType.OPENAI: "gpt-4",
    LLMProviderType.ANTHROPIC: "claude-3-5-sonnet-20241022",
}
```

En Java:

```java
private static final Map<LLMProviderType, String> DEFAULT_MODELS = Map.of(
    LLMProviderType.OLLAMA_LOCAL, "llama3.2:3b",
    LLMProviderType.OLLAMA_SERVER, "llama3.2:3b",
    LLMProviderType.OPENAI, "gpt-4",
    LLMProviderType.ANTHROPIC, "claude-3-5-sonnet-20241022"
);
```

---

## 4. base_provider.py - ABC con metodos concretos

### La clase abstracta

```python
class LLMProvider(ABC):
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model

    @abstractmethod
    def generate(self, prompt, context=None, temperature=0.7, max_tokens=1024) -> str:
        pass

    @abstractmethod
    def generate_stream(self, prompt, context=None, ...) -> Iterator[str]:
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, str]:
        pass

    # Metodo CONCRETO — compartido por todos los providers
    def format_prompt_with_context(self, prompt, context=None) -> str:
        if not context:
            return prompt
        return f"Use the following context...\n\nContext:\n{context}\n\nQuestion: {prompt}\n\n..."
```

La mezcla de abstracto y concreto es comun en Python. En Java seria:

```java
public abstract class LLMProvider {
    protected String model;

    public LLMProvider(String model) { this.model = model; }

    // Metodos abstractos — las subclases los implementan
    public abstract String generate(String prompt, String context, double temperature, int maxTokens);
    public abstract Iterator<String> generateStream(String prompt, String context, ...);
    public abstract Map<String, String> getModelInfo();

    // Metodo concreto — compartido
    public String formatPromptWithContext(String prompt, String context) {
        if (context == null || context.isEmpty()) return prompt;
        return "Use the following context...\nContext:\n" + context + "\n\nQuestion: " + prompt + "\n\n...";
    }
}
```

### Optional[str] = None con `or`

```python
def __init__(self, model: Optional[str] = None) -> None:
    self.model = model
```

Y luego en las subclases:

```python
self.model = model or DEFAULT_MODELS[LLMProviderType.OPENAI]
```

El `or` funciona asi:
- Si `model` es `"gpt-4"` → usa `"gpt-4"`
- Si `model` es `None` → `None or DEFAULT_MODELS[...]` → usa el default

En Java:

```java
this.model = model != null ? model : DEFAULT_MODELS.get(LLMProviderType.OPENAI);
// O con Optional:
this.model = Optional.ofNullable(model).orElse(DEFAULT_MODELS.get(LLMProviderType.OPENAI));
```

---

## 5. OllamaProvider - el cliente local

### Constructor con branch

```python
class OllamaProvider(LLMProvider):
    def __init__(self, model=None, server_url=None):
        super().__init__(model)
        self.model = model or DEFAULT_MODELS[LLMProviderType.OLLAMA_LOCAL]
        self.server_url = server_url

        if server_url:
            self.client = ollama.Client(host=server_url)
        else:
            self.client = ollama.Client()  # localhost:11434
```

Ollama es el unico provider con dos modos: local (sin URL) y server (con URL).
Un solo `OllamaProvider` maneja ambos — la diferencia es solo si pasas `server_url`.

En Java:

```java
public class OllamaProvider extends LLMProvider {
    private final OllamaClient client;
    private final String serverUrl;

    public OllamaProvider(String model, String serverUrl) {
        super(model != null ? model : DEFAULT_MODELS.get(LLMProviderType.OLLAMA_LOCAL));
        this.serverUrl = serverUrl;

        if (serverUrl != null) {
            this.client = new OllamaClient(serverUrl);
        } else {
            this.client = new OllamaClient(); // localhost:11434
        }
    }
}
```

### generate() - llamada sync

```python
def generate(self, prompt, context=None, temperature=0.7, max_tokens=1024):
    full_prompt = self.format_prompt_with_context(prompt, context)

    try:
        response = self.client.generate(
            model=self.model,
            prompt=full_prompt,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response["response"]
    except Exception as e:
        raise RuntimeError(f"Failed to generate response: {e}") from e
```

Nota: Ollama devuelve un **dict** (`response["response"]`), no un objeto.
Esto es diferente a OpenAI/Anthropic que devuelven objetos con atributos.

### El patron try/except con re-raise

```python
try:
    response = self.client.generate(...)
    return response["response"]
except Exception as e:
    logger.error("Ollama generation failed: %s", e)
    raise RuntimeError(f"Failed to generate response: {e}") from e
```

El `from e` es importante: encadena la excepcion original (`ConnectionError`,
`TimeoutError`, etc.) con nuestra `RuntimeError`. Asi puedes ver ambas en el
traceback.

En Java:

```java
try {
    var response = client.generate(model, prompt, options);
    return response.get("response");
} catch (Exception e) {
    logger.error("Ollama generation failed: {}", e.getMessage());
    throw new RuntimeException("Failed to generate response: " + e.getMessage(), e);
    //                                                                         ↑ causa
}
```

---

## 6. OpenAIProvider - Chat Completions API

### La API de OpenAI es diferente

Ollama usa `client.generate(prompt=...)` — le pasas un string.
OpenAI usa `client.chat.completions.create(messages=[...])` — le pasas una lista de mensajes.

```python
response = self.client.chat.completions.create(
    model=self.model,
    messages=[{"role": "user", "content": full_prompt}],
    temperature=temperature,
    max_tokens=max_tokens,
)
return response.choices[0].message.content
```

La respuesta tiene una estructura anidada:
- `response` → objeto `ChatCompletion`
- `.choices` → lista de opciones
- `[0]` → primera opcion (normalmente solo hay una)
- `.message` → objeto `Message`
- `.content` → el texto generado

En Java (con la SDK oficial de OpenAI):

```java
ChatCompletionRequest request = ChatCompletionRequest.builder()
    .model(model)
    .messages(List.of(new ChatMessage("user", fullPrompt)))
    .temperature(temperature)
    .maxTokens(maxTokens)
    .build();

ChatCompletion response = client.createChatCompletion(request);
return response.getChoices().get(0).getMessage().getContent();
```

### Streaming en OpenAI

```python
stream = self.client.chat.completions.create(
    model=self.model,
    messages=[{"role": "user", "content": full_prompt}],
    stream=True,  # ← activa streaming
)

for chunk in stream:
    if chunk.choices[0].delta.content:  # ← delta, no message
        yield chunk.choices[0].delta.content
```

Cuando `stream=True`, la respuesta cambia:
- En vez de `response.choices[0].message.content` (completo)
- Obtienes `chunk.choices[0].delta.content` (fragmento)
- Algunos chunks tienen `delta.content = None` → los filtramos con `if`

---

## 7. AnthropicProvider - Messages API y context manager

### Diferencia clave: la API de Anthropic

```python
response = self.client.messages.create(
    model=self.model,
    max_tokens=max_tokens,      # Obligatorio en Anthropic (no en OpenAI)
    temperature=temperature,
    messages=[{"role": "user", "content": full_prompt}],
)
return response.content[0].text
```

La estructura es diferente a OpenAI:
- OpenAI: `response.choices[0].message.content`
- Anthropic: `response.content[0].text`

### Streaming con context manager (with)

```python
def generate_stream(self, prompt, context=None, temperature=0.7, max_tokens=1024):
    full_prompt = self.format_prompt_with_context(prompt, context)

    try:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": full_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        raise RuntimeError(f"Failed to stream response: {e}") from e
```

Anthropic usa un **context manager** (`with ... as stream`) para streaming.
Esto garantiza que el stream se cierra correctamente cuando terminas.

En Java esto equivale a un try-with-resources:

```java
try (var stream = client.messages().stream(request)) {
    stream.textStream().forEach(text -> System.out.print(text));
}
```

### Comparacion de las tres APIs

| Aspecto | Ollama | OpenAI | Anthropic |
|---------|--------|--------|-----------|
| Sync | `client.generate()` | `client.chat.completions.create()` | `client.messages.create()` |
| Respuesta | `response["response"]` (dict) | `response.choices[0].message.content` (obj) | `response.content[0].text` (obj) |
| Stream | `generate(stream=True)` | `create(stream=True)` | `messages.stream()` (context mgr) |
| Stream chunk | `chunk["response"]` | `chunk.choices[0].delta.content` | `stream.text_stream` |
| Auth | None (local) | API key | API key |
| Coste | $0 | ~$30/1M tokens | ~$15/1M tokens |

---

## 8. LLMProviderFactory - lazy imports

### El factory completo

```python
class LLMProviderFactory:
    @staticmethod
    def create_provider(provider_type=None, model=None, **kwargs):
        from config.settings import settings          # ← Lazy import #1

        provider_type = provider_type or settings.llm_provider
        model = model or settings.llm_model
        provider_enum = LLMProviderType(provider_type)

        if provider_enum == LLMProviderType.OLLAMA_LOCAL:
            from .ollama_provider import OllamaProvider   # ← Lazy import #2
            return OllamaProvider(model=model, server_url=None)

        elif provider_enum == LLMProviderType.OPENAI:
            from .openai_provider import OpenAIProvider   # ← Lazy import #3
            api_key = kwargs.get("api_key") or settings.openai_api_key
            return OpenAIProvider(model=model, api_key=api_key)

        elif provider_enum == LLMProviderType.ANTHROPIC:
            from .anthropic_provider import AnthropicProvider  # ← Lazy import #4
            api_key = kwargs.get("api_key") or settings.anthropic_api_key
            return AnthropicProvider(model=model, api_key=api_key)

        # ...
```

Hay **dos niveles** de lazy imports:

1. `from config.settings import settings` — carga settings solo cuando se llama al factory
2. `from .ollama_provider import OllamaProvider` — carga el SDK de Ollama solo si usas Ollama

Esto significa que si usas OpenAI, nunca se carga el SDK de Ollama ni el de Anthropic.

### kwargs.get("api_key") or settings.openai_api_key

```python
api_key = kwargs.get("api_key") or settings.openai_api_key
```

Cascada de prioridad:
1. `kwargs["api_key"]` — pasado explicitamente al factory
2. `settings.openai_api_key` — leido de `.env` via Pydantic
3. `None` — el SDK intenta leer `OPENAI_API_KEY` del entorno

En Java:

```java
String apiKey = kwargs.containsKey("api_key")
    ? kwargs.get("api_key")
    : settings.getOpenaiApiKey();
```

---

## 9. Streaming con generators (yield)

### Que es un generator?

En Python, una funcion con `yield` es un **generator** — produce valores uno
a uno, sin cargar todo en memoria.

```python
def generate_stream(self, prompt, ...):
    stream = self.client.chat.completions.create(stream=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content   # ← produce un valor
```

Cuando el llamador hace:

```python
for text in provider.generate_stream("Hello"):
    print(text, end="")
```

Cada `yield` "pausa" la funcion y devuelve un valor. El `for` resume la funcion
para obtener el siguiente. Es como un `Iterator<String>` en Java pero **mucho**
mas simple de escribir.

### En Java necesitas mas codigo

```java
// Java — hay que implementar Iterator manualmente
public class LLMStream implements Iterator<String> {
    private final Iterator<ChatCompletionChunk> chunks;

    public LLMStream(Stream<ChatCompletionChunk> stream) {
        this.chunks = stream.iterator();
    }

    @Override
    public boolean hasNext() {
        return chunks.hasNext();
    }

    @Override
    public String next() {
        ChatCompletionChunk chunk = chunks.next();
        return chunk.getChoices().get(0).getDelta().getContent();
    }
}
```

O con Java Streams:

```java
// Java 8+ — mas conciso con Stream
Stream<String> stream = client.createChatCompletionStream(request)
    .map(chunk -> chunk.getChoices().get(0).getDelta().getContent())
    .filter(Objects::nonNull);
```

### Type hint: Iterator[str]

```python
def generate_stream(self, ...) -> Iterator[str]:
```

`Iterator[str]` es el tipo correcto para un generator que produce strings.
Podrias tambien usar `Generator[str, None, None]` pero `Iterator` es mas simple.

En Java: `Iterator<String>` o `Stream<String>`.

---

## 10. format_prompt_with_context - el template RAG

### El metodo concreto en la base

```python
def format_prompt_with_context(self, prompt, context=None):
    if not context:
        return prompt

    return (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {prompt}\n\n"
        "Answer based on the context above. "
        "If the context doesn't contain enough information, say so."
    )
```

Este metodo es **concreto** (no abstracto) — todas las implementaciones lo heredan.
Pero un provider podria sobreescribirlo si necesita un formato diferente.

### if not context

```python
if not context:
    return prompt
```

`not context` es `True` cuando:
- `context` es `None`
- `context` es `""` (string vacio)
- `context` es `0` (cero)

En Java:

```java
if (context == null || context.isEmpty()) {
    return prompt;
}
```

### Concatenacion con parentesis

```python
return (
    "Use the following context...\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {prompt}\n\n"
    "Answer based on..."
)
```

En Python, strings literales adyacentes se concatenan automaticamente:

```python
"hello " "world"  # → "hello world"
```

Los parentesis `()` permiten repartir la concatenacion en varias lineas sin usar
`+`. El resultado es un solo string.

En Java:

```java
return "Use the following context...\n\n"
    + "Context:\n" + context + "\n\n"
    + "Question: " + prompt + "\n\n"
    + "Answer based on...";
```

O con text blocks (Java 15+):

```java
return """
    Use the following context to answer the question.

    Context:
    %s

    Question: %s

    Answer based on the context above. If the context doesn't contain enough information, say so.
    """.formatted(context, prompt);
```

---

## 11. config/settings.py - campos LLM con Pydantic

### Los 7 campos nuevos

```python
class DocVaultSettings(BaseSettings):
    # ... campos M1 existentes ...

    # LLM Configuration
    llm_provider: str = Field(default="ollama_local", description="LLM provider type")
    llm_model: Optional[str] = Field(default=None, description="LLM model name")
    llm_server_url: Optional[str] = Field(default=None, description="LLM server URL")
    llm_temperature: float = Field(default=0.7, description="Generation temperature")
    llm_max_tokens: int = Field(default=1024, description="Maximum response tokens")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
```

Pydantic lee automaticamente de las variables de entorno:
- `LLM_PROVIDER=openai` en `.env` → `settings.llm_provider == "openai"`
- `OPENAI_API_KEY=sk-...` en `.env` → `settings.openai_api_key == "sk-..."`

La convencion: Python usa `snake_case` (`llm_provider`), las variables de entorno
usan `UPPER_SNAKE_CASE` (`LLM_PROVIDER`). Pydantic convierte automaticamente.

En Java con Spring:

```java
@Configuration
@ConfigurationProperties(prefix = "llm")
public class LLMConfig {
    private String provider = "ollama_local";
    private String model;
    private String serverUrl;
    private double temperature = 0.7;
    private int maxTokens = 1024;
    // + getters/setters (o usar Lombok @Data)
}
```

### Optional[str] = Field(default=None)

```python
llm_model: Optional[str] = Field(default=None, description="LLM model name")
```

`Optional[str]` = puede ser `str` o `None`.
`Field(default=None)` = si no esta en `.env`, vale `None`.

En el factory: `model = model or settings.llm_model` — si el usuario no pasa modelo
Y no esta en `.env`, usa el default de `DEFAULT_MODELS[provider_type]`.

---

## 12. Tests con MagicMock y @patch

### Mockear un SDK completo

Los tests de LLM no pueden llamar a APIs reales (necesitarian API keys, Ollama corriendo, etc.).
Usamos `MagicMock` + `@patch` para simular las respuestas.

### Ejemplo: TestOpenAIProvider

```python
@patch("src.llm.openai_provider.OpenAI")       # ← Parchea la clase OpenAI
def test_generate_success(self, mock_openai_cls):
    # 1. Crear la cadena de mocks
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "The answer is 42."
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_cls.return_value = mock_client   # OpenAI() → mock_client

    # 2. Crear provider (usa el mock, no el SDK real)
    provider = OpenAIProvider(api_key="sk-test")

    # 3. Llamar generate
    result = provider.generate("What is the meaning of life?")

    # 4. Verificar
    assert result == "The answer is 42."
    mock_client.chat.completions.create.assert_called_once()
```

### La cadena de mocks — por que tantos?

OpenAI devuelve `response.choices[0].message.content`. Para mockear esto necesitas:

```
mock_response
    └── .choices = [mock_choice]
                        └── .message = mock_message
                                           └── .content = "The answer is 42."
```

En Java con Mockito seria igual de verboso:

```java
// Java — cadena de mocks
var mockMessage = mock(ChatMessage.class);
when(mockMessage.getContent()).thenReturn("The answer is 42.");

var mockChoice = mock(ChatCompletionChoice.class);
when(mockChoice.getMessage()).thenReturn(mockMessage);

var mockResponse = mock(ChatCompletion.class);
when(mockResponse.getChoices()).thenReturn(List.of(mockChoice));

var mockClient = mock(OpenAIClient.class);
when(mockClient.createChatCompletion(any())).thenReturn(mockResponse);
```

### @patch - donde parchear

```python
@patch("src.llm.openai_provider.OpenAI")
```

Esto **reemplaza** la clase `OpenAI` dentro del modulo `src.llm.openai_provider`.
Cuando `openai_provider.py` hace `from openai import OpenAI`, obtiene el mock.

Regla: parchea **donde se usa**, no donde se define.

- `src.llm.openai_provider.OpenAI` ← donde se usa (en el import del modulo)
- `openai.OpenAI` ← donde se define (modulo original)

Si parcharas `openai.OpenAI`, no funcionaria porque `openai_provider.py` ya tiene
su propia referencia a la clase original.

### Mockear streaming (context manager)

Para Anthropic, el streaming usa `with ... as stream:`:

```python
@patch("src.llm.anthropic_provider.Anthropic")
def test_generate_stream_success(self, mock_anthropic_cls):
    mock_client = MagicMock()

    # Crear mock de context manager
    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_ctx)
    mock_stream_ctx.__exit__ = MagicMock(return_value=False)
    mock_stream_ctx.text_stream = iter(["Hello", " from", " Claude"])

    mock_client.messages.stream.return_value = mock_stream_ctx
    mock_anthropic_cls.return_value = mock_client

    provider = AnthropicProvider(api_key="sk-ant-test")
    chunks = list(provider.generate_stream("Say hello"))

    assert chunks == ["Hello", " from", " Claude"]
```

Los `__enter__` y `__exit__` son los metodos que Python llama cuando usas `with`.

En Java, un context manager es un `AutoCloseable`:

```java
// Java — el equivalente de __enter__/__exit__
public interface AutoCloseable {
    void close() throws Exception;
}

// Y lo usas con try-with-resources
try (var stream = client.stream(request)) {
    // __enter__ = constructor / open
    stream.forEach(chunk -> ...);
} // __exit__ = close()
```

---

## 13. Por que lazy import en la factory

### El problema del import a nivel de modulo

Si la factory importara todo arriba:

```python
# provider_factory.py — imports normales (NO lo que hacemos)
from config.settings import settings
from .ollama_provider import OllamaProvider     # ← carga ollama SDK
from .openai_provider import OpenAIProvider     # ← carga openai SDK
from .anthropic_provider import AnthropicProvider  # ← carga anthropic SDK
```

Cualquier `from src.llm import LLMProviderFactory` cargaria los tres SDKs
y accederia a `settings`. Problemas:

1. Si no tienes `ollama` instalado → `ImportError` al importar la factory
2. Si no tienes `.env` → error al cargar settings
3. Los tests que solo prueban un provider cargan los tres SDKs

### La solucion: imports dentro del metodo

```python
@staticmethod
def create_provider(provider_type=None, model=None, **kwargs):
    from config.settings import settings  # Solo cuando se llama

    if provider_enum == LLMProviderType.OLLAMA_LOCAL:
        from .ollama_provider import OllamaProvider  # Solo si usas Ollama
        return OllamaProvider(...)

    elif provider_enum == LLMProviderType.OPENAI:
        from .openai_provider import OpenAIProvider  # Solo si usas OpenAI
        return OpenAIProvider(...)
```

### Impacto en tests

Sin lazy imports, el test del factory necesitaria parchear todos los SDKs:

```python
# Sin lazy imports — necesitas parchear todo
@patch("src.llm.provider_factory.Anthropic")
@patch("src.llm.provider_factory.OpenAI")
@patch("src.llm.provider_factory.ollama.Client")
@patch("src.llm.provider_factory.settings")
def test_create_openai(self, mock_settings, mock_ollama, mock_openai, mock_anthropic):
    ...
```

Con lazy imports, solo parcheas lo que usas:

```python
# Con lazy imports — solo parcheas lo necesario
@patch("src.llm.openai_provider.OpenAI")
@patch("config.settings.settings")
def test_create_openai(self, mock_settings, mock_openai_cls):
    ...
```

### Nota sobre @patch("config.settings.settings")

La factory hace `from config.settings import settings` (lazy). Pero
`settings` es un **objeto singleton** definido en `config.settings`.

Al parchear `config.settings.settings`, reemplazas el objeto en su modulo
de origen. Cuando el lazy import ejecuta `from config.settings import settings`,
obtiene el mock.

Si en cambio parcharas `src.llm.provider_factory.settings`, fallaria porque
ese nombre no existe a nivel de modulo — solo existe dentro del metodo.

---

## 14. Patron completo: de .env a respuesta LLM

```
     .env
      │
      │  LLM_PROVIDER=openai
      │  LLM_MODEL=gpt-4
      │  OPENAI_API_KEY=sk-...
      │
      ▼
  config/settings.py (Pydantic)
      │
      │  settings.llm_provider = "openai"
      │  settings.llm_model = "gpt-4"
      │  settings.openai_api_key = "sk-..."
      │
      ▼
  LLMProviderFactory.create_provider()
      │
      │  1. from config.settings import settings  (lazy)
      │  2. provider_type = "openai"
      │  3. LLMProviderType("openai") → OPENAI
      │  4. from .openai_provider import OpenAIProvider  (lazy)
      │  5. api_key = settings.openai_api_key
      │
      ▼
  OpenAIProvider(model="gpt-4", api_key="sk-...")
      │
      │  self.client = OpenAI(api_key="sk-...")
      │  self.model = "gpt-4"
      │
      ▼
  provider.generate(prompt="What is Python?", context="Python is...")
      │
      │  1. format_prompt_with_context(prompt, context)
      │     → "Use the following context...\nContext:\nPython is...\n\nQuestion: What is Python?"
      │
      │  2. client.chat.completions.create(
      │         model="gpt-4",
      │         messages=[{"role": "user", "content": full_prompt}],
      │         temperature=0.7,
      │         max_tokens=1024
      │     )
      │
      │  3. return response.choices[0].message.content
      │
      ▼
  "Python is a high-level programming language..."
```

### Cambiar de provider = solo cambiar .env

```
# Antes (OpenAI)              # Despues (Ollama local)
LLM_PROVIDER=openai     →     LLM_PROVIDER=ollama_local
LLM_MODEL=gpt-4         →     LLM_MODEL=llama3.2:3b
OPENAI_API_KEY=sk-...   →     (no necesita API key)
```

El codigo no cambia:
```python
provider = LLMProviderFactory.create_provider()
response = provider.generate(prompt="...", context="...")
```

---

## 15. Tabla resumen Java vs Python en M6

| Concepto | Java | Python (nuestro proyecto) |
|----------|------|---------------------------|
| Interfaz abstracta | `interface` / `abstract class` | `ABC` con `@abstractmethod` |
| Metodo concreto en interfaz | `default` method (Java 8+) | Metodo sin `@abstractmethod` en ABC |
| Enum con valor string | Enum con campo + `fromValue()` | `class Type(str, Enum)` (str mixin) |
| Buscar enum por valor | `fromValue("openai")` manual | `LLMProviderType("openai")` automatico |
| Modelo default | `Optional.ofNullable(model).orElse(...)` | `model or DEFAULT_MODELS[...]` |
| Factory | `static` method en clase | `@staticmethod` en clase |
| Lazy import | No existe (imports no ejecutan codigo) | `from x import y` dentro del metodo |
| Config desde env vars | Spring `@ConfigurationProperties` | Pydantic `BaseSettings` |
| Streaming | `Iterator<String>` / `Stream<String>` | `yield` (generator) → `Iterator[str]` |
| Context manager | `try-with-resources` + `AutoCloseable` | `with ... as stream:` + `__enter__/__exit__` |
| Mock SDK client | Mockito `mock(OpenAI.class)` | `MagicMock()` + `@patch("module.Class")` |
| Verificar llamada | `verify(mock, times(1)).method()` | `mock.method.assert_called_once()` |
| Mock context manager | No necesario (AutoCloseable es interfaz) | `mock.__enter__` + `mock.__exit__` |
| Exception chaining | `throw new RuntimeException("msg", cause)` | `raise RuntimeError("msg") from e` |
| String concatenation | `"str1" + var + "str2"` / text blocks | `f"str1{var}str2"` / parenthesized literals |
| Falsy check | `if (x == null \|\| x.isEmpty())` | `if not x:` (None, "", 0 son falsy) |
| API response access | `response.getChoices().get(0).getMessage()` | `response.choices[0].message.content` |
| Dict access | `map.get("key")` | `dict["key"]` o `dict.get("key")` |
| Logging format | SLF4J `{}` placeholders | `%s` formatting (no f-strings en logger) |

---

## Resumen final

M6 ha creado la capa de LLM que faltaba para completar el pipeline RAG:

1. **LLMProvider (ABC)** → Interfaz abstracta, como una `interface` en Java
2. **OllamaProvider** → Cliente local/remoto, respuestas como dict
3. **OpenAIProvider** → Chat Completions API, respuestas como objetos anidados
4. **AnthropicProvider** → Messages API, streaming con context manager
5. **LLMProviderFactory** → Lazy imports, lee config de Pydantic
6. **28 unit tests** → MagicMock para simular SDKs sin API keys

Con M6 completado, el sistema RAG tiene todas las piezas:
- **M2**: Texto → vectores
- **M3**: Guardar y buscar vectores
- **M4**: Archivos → texto limpio
- **M5**: Pipeline automatico de ingestion
- **M6**: Generar respuestas con cualquier LLM

Lo que falta (M7): **conectar todo** — el pipeline RAG completo que
toma una pregunta del usuario, busca documentos relevantes, y genera
una respuesta usando el LLM. Mas FastAPI y CLI.
