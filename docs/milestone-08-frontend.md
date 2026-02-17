# Milestone 8: Web Frontend

**Status:** 🚧 In Progress (Phase 1 completed)
**Dependencies:** M7 (API), M5 (Ingestion)
**Goal:** Build a web UI for non-technical users to query, upload, and manage documentation

---

## Overview

This milestone adds a web frontend to DocVault so non-technical users (documentation writers, team leads, support staff) can interact with the RAG system through a browser instead of CLI or API calls.

**Key result:** Open a browser → upload documents → ask questions → get AI answers with cited sources.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Browser (React SPA)                   │
│                                                           │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │  QueryPage   │  │ DocumentsPage  │  │  AdminPage   │ │
│  │              │  │                │  │              │ │
│  │ Ask questions│  │ Upload & ingest│  │ Health &     │ │
│  │ See answers  │  │ Manage files   │  │ collection   │ │
│  │ View sources │  │ Trigger index  │  │ info         │ │
│  └──────┬───────┘  └───────┬────────┘  └──────┬───────┘ │
│         │                  │                   │         │
│         └──────────────────┼───────────────────┘         │
│                            │                             │
│                    ┌───────┴───────┐                     │
│                    │  API Client   │                     │
│                    │  (fetch)      │                     │
│                    └───────┬───────┘                     │
└────────────────────────────┼─────────────────────────────┘
                             │ HTTP (Vite proxy /api → :8000)
┌────────────────────────────┼─────────────────────────────┐
│                    FastAPI Backend                        │
│                                                           │
│  Existing (M7):              New (M8 Phase 1):           │
│  GET  /health                POST /documents/upload      │
│  POST /query                 GET  /documents             │
│  POST /query/stream          DELETE /documents/{name}    │
│  GET  /sources               POST /ingest                │
│                              GET  /ingest/status          │
│                              GET  /config                 │
└──────────────────────────────────────────────────────────┘
```

## Technology Stack

| Technology | Purpose |
|------------|---------|
| React 18 + TypeScript | UI framework with type safety |
| Vite | Fast build tool with HMR and dev proxy |
| Tailwind CSS | Utility-first styling |
| React Router v6 | Client-side page navigation |
| react-markdown + remark-gfm | Render LLM responses as formatted markdown |
| react-dropzone | Drag & drop file upload |
| lucide-react | Icon library |
| Native fetch | HTTP client (no axios) |

**Not using:** Redux, Zustand, axios, Material UI, Ant Design. State is simple (`useState` + `useEffect`).

## Phase Progress

| Phase | Status | Document |
|-------|--------|----------|
| **Phase 1: Backend API** | ✅ Done | [milestone-08-phase1-backend-api.md](milestone-08-phase1-backend-api.md) |
| **Phase 2: Frontend Foundation** | 🚧 Next | [milestone-08-phase2-frontend-foundation.md](milestone-08-phase2-frontend-foundation.md) |
| **Phase 3: Functional Pages** | ⏸️ Pending | [milestone-08-phase3-functional-pages.md](milestone-08-phase3-functional-pages.md) |
| **Phase 4: Polish & Documentation** | ⏸️ Pending | [milestone-08-phase4-polish.md](milestone-08-phase4-polish.md) |

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Native fetch vs axios | Fewer dependencies; fetch is sufficient for our use cases |
| useState vs Redux/Zustand | Simple state not shared between pages; no need for global store |
| Tailwind vs Material UI | Lighter, utility-first, doesn't impose component design |
| No streaming in v1 | Greatly simplifies frontend; can add later with EventSource |
| Vite proxy in dev | Avoids CORS issues in development without extra config |
| No authentication | Internal/educational project; can add later |
| Monorepo (frontend/) | Single repo keeps backend + frontend in sync; simpler CI/CD |

## Files Created/Modified (all phases)

### Backend (Phase 1 - done)

| File | Changes |
|------|---------|
| `src/api/server.py` | CORS middleware + 6 new endpoints + 5 Pydantic models |
| `tests/unit/test_api.py` | 15 new tests for document/ingestion/config endpoints |
| `requirements.txt` | Added `python-multipart==0.0.22` |

### Frontend (Phases 2-3)

| File | Purpose |
|------|---------|
| `frontend/package.json` | Dependencies (React, Vite, Tailwind, etc.) |
| `frontend/vite.config.ts` | Vite config + API proxy |
| `frontend/tsconfig.json` | TypeScript configuration |
| `frontend/index.html` | HTML entry point |
| `frontend/src/main.tsx` | React entry + Router setup |
| `frontend/src/App.tsx` | Layout wrapper |
| `frontend/src/styles/index.css` | Tailwind imports |
| `frontend/src/api/client.ts` | API client functions |
| `frontend/src/types/index.ts` | TypeScript interfaces |
| `frontend/src/pages/QueryPage.tsx` | Question & answer page |
| `frontend/src/pages/DocumentsPage.tsx` | Document management page |
| `frontend/src/pages/AdminPage.tsx` | System admin page |
| `frontend/src/components/Layout.tsx` | Sidebar + header layout |
| `frontend/src/components/SourceCard.tsx` | Source citation card |
| `frontend/src/components/FileUploader.tsx` | Drag & drop upload |
| `frontend/src/components/FileList.tsx` | Document list table |
| `frontend/src/components/HealthBadge.tsx` | Health status indicator |

## Full Verification (all phases complete)

```bash
# Backend tests
pytest tests/unit/test_api.py -v

# Start backend
python -m src.api.server

# Start frontend (separate terminal)
cd frontend && npm run dev

# Open http://localhost:5173
# 1. Documents → drag & drop a PDF/MD file
# 2. Documents → click "Ingest All"
# 3. Query → type a question → verify answer + sources
# 4. Admin → verify health green + collection info
```

---

**Related Files:**
- `src/api/server.py` — Extended FastAPI server (Phase 1)
- `frontend/` — React application (Phases 2-3)
- `tests/unit/test_api.py` — Backend API tests (Phase 1)
- `config/settings.py` — API host/port configuration
