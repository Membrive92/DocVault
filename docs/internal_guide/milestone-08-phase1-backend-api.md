# Guia Interna - Fase 1: Backend API

> Conceptos de backend comparados con Java/Spring.

---

## CORS y nuevos endpoints

CORS (Cross-Origin Resource Sharing) es un mecanismo del navegador que bloquea peticiones
a un dominio distinto. En Java Spring usas `@CrossOrigin`. En FastAPI:

```python
# FastAPI CORS — equivalente a @CrossOrigin en Spring
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Esto solo es necesario cuando el frontend y backend corren en puertos distintos.
En produccion detras de un proxy (mismo dominio), CORS no es necesario.

---

## File upload con FastAPI

En Spring usas `@RequestParam MultipartFile file`. En FastAPI:

```java
// Spring
@PostMapping("/documents/upload")
public UploadResponse upload(@RequestParam("file") MultipartFile file) {
    Path dest = Paths.get("data/documents/" + file.getOriginalFilename());
    file.transferTo(dest.toFile());
    return new UploadResponse(file.getOriginalFilename(), file.getSize());
}
```

```python
# FastAPI
from fastapi import UploadFile

@app.post("/documents/upload")
async def upload_document(file: UploadFile):
    dest = settings.documents_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)
    return UploadResponse(filename=file.filename, size_bytes=dest.stat().st_size)
```

El patron es practicamente identico: recibir archivo → validar formato → guardar en disco.

**Diferencia clave:** FastAPI necesita `python-multipart` instalado para soportar
`UploadFile`. Spring lo trae de serie con `spring-boot-starter-web`.

---

## Tabla comparativa Fase 1

| Concepto | Java/Spring | FastAPI (Python) |
|----------|-------------|-----------------|
| CORS | `@CrossOrigin` o `WebMvcConfigurer` | `CORSMiddleware` |
| File upload | `@RequestParam MultipartFile` | `UploadFile` |
| Validar extension | Manual en controller | Manual, usando `SUPPORTED_EXTENSIONS` |
| Listar archivos | `Files.list(path)` | `Path.iterdir()` |
| Eliminar archivo | `Files.delete(path)` | `Path.unlink()` |
| Modelos request/response | `record` o `class` con Jackson | `BaseModel` con Pydantic |
| Dependencia multipart | Incluida en starter | `pip install python-multipart` |

---

**Siguiente:** [Fase 2 — Frontend Foundation](milestone-08-phase2-frontend-foundation.md)
