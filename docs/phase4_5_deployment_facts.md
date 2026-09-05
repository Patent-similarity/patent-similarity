# Phase 4/5 - Deployment Facts for Phase 6

## Backend (Phase 4)

Startup command (from repo root):
uvicorn src.phase4_backend.main:app --host 0.0.0.0 --port 8000
(drop --host/--port for local defaults; --reload is dev-only)

Port: 8000

Required env var: GEMINI_API_KEY (loaded via python-dotenv)

API contract:
POST /search  { "query": "<text>" }  -> { job_id, status }
GET /search/{job_id}  -> job status + results (see src/phase4_backend/models.py)
GET /health  -> { status: ok }
Full OpenAPI schema at /openapi.json once running

Startup dependencies (must exist before boot):
embeddings/faiss_index/patent_similarity.faiss
embeddings/faiss_index/patent_similarity_metadata.json
data/patents.db

## Frontend (Phase 5)

Build/start command: none - static HTML/CSS/JS, no build step
Serve src/phase5_frontend/ as static files, or open index.html directly

Frontend env vars / backend URL config:
src/phase5_frontend/config.js sets BACKEND_URL
Edit that one file before deploying to point at the real backend

## Not yet done
CORS is currently allow_origins=[*] in main.py - needs restricting before production
