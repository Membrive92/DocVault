# Guia Interna - Milestone 8: Web Frontend

> Guia escrita para desarrolladores con experiencia en Java.
> Cada concepto de React/TypeScript se compara con su equivalente en Java.

---

## Indice por fases

| Fase | Documento | Conceptos Java comparados |
|------|-----------|---------------------------|
| **Fase 1: Backend API** | [milestone-08-phase1-backend-api.md](milestone-08-phase1-backend-api.md) | CORS vs `@CrossOrigin`, `UploadFile` vs `MultipartFile` |
| **Fase 2: Frontend Foundation** | [milestone-08-phase2-frontend-foundation.md](milestone-08-phase2-frontend-foundation.md) | TypeScript vs Java, Vite vs Maven, Tailwind, Proxy, Estructura proyecto, fetch vs RestTemplate, React Router vs @RequestMapping |
| **Fase 3: Functional Pages** | [milestone-08-phase3-functional-pages.md](milestone-08-phase3-functional-pages.md) | Componentes React vs clases Java, useState vs campos privados, useEffect vs @PostConstruct |

---

## Que vamos a construir

En M1-M7 construimos todo el backend: embeddings, Qdrant, parsers, ingestion, LLM y RAG pipeline
con API REST. Pero solo usuarios tecnicos pueden usarlo (curl, Python, CLI).

En M8 añadimos un **frontend web** para que cualquier persona abra un navegador y:
- Pregunte cosas sobre la documentacion
- Suba documentos nuevos
- Vea el estado del sistema

Es el equivalente a añadir un frontend Angular/React a tu API REST de Spring Boot.

```
Spring Boot + Thymeleaf  →  FastAPI + React (separados)
Monolito MVC              →  SPA + API REST
```

---

## Tabla resumen Java vs React/TypeScript en M8

| Concepto | Java/Spring | React/TypeScript |
|----------|-------------|-----------------|
| Componente UI | `class extends JPanel` | `function Component()` |
| Estado del componente | Campos privados | `useState()` hook |
| Inicializacion | `@PostConstruct` | `useEffect(() => {}, [])` |
| Cleanup | `@PreDestroy` | return function en useEffect |
| Routing | `@GetMapping("/path")` | `<Route path="/path" element={<Page />} />` |
| HTTP Client | `RestTemplate` / `WebClient` | `fetch()` nativo |
| Serializar JSON | Jackson automatico | `JSON.stringify()` / `.json()` |
| Tipos/Modelos | `class DTO` / `record` | `interface` TypeScript |
| Gestion deps | Maven/Gradle | npm (package.json) |
| Build tool | Maven/Gradle | Vite |
| Dev server | Spring Boot embedded Tomcat | Vite dev server |
| CSS | archivo `.css` separado | Tailwind clases utility |
| CORS | `@CrossOrigin` | `CORSMiddleware` en FastAPI |
| File upload | `MultipartFile` | `UploadFile` en FastAPI / `FormData` en JS |
| Proxy reverso | nginx / Spring Cloud Gateway | Vite proxy (dev) / nginx (prod) |
| Hot reload | JRebel (de pago) | HMR nativo (gratis, instantaneo) |
