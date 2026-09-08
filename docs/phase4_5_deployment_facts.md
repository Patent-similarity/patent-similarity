# Phase 4/5 - Deployment Facts for Phase 6

## Purpose

This document records the verified deployment facts for the completed Phase 4 backend and Phase 5 frontend.

Phase 4 and Phase 5 are functionally integrated with the real Phase 2/3 pipeline and have been verified end-to-end using real API searches and real production artifacts.

---

## 1. Application Components

The application consists of:

### Phase 2 - Embedding + Retrieval

- Gemini embedding client
- FAISS similarity search
- FAISS metadata
- Real patent database
- Two-stage retrieval:
  - Stage 1: abstract similarity
  - Stage 2: claims-based reranking

### Phase 3 - Ranking + Synthesis

- Gemini LLM synthesis
- Candidate relevance scoring
- Verdict generation
- Overlap summary
- Key difference
- Supporting evidence

### Phase 4 - Backend

- FastAPI application
- Job-based asynchronous search API
- Search status polling
- CORS configuration
- Error handling
- Startup validation

### Phase 5 - Frontend

- Static HTML/CSS/JavaScript interface
- Patent search form
- Automatic job polling
- Elapsed-time display
- Synthesized result rendering
- Claims-matched candidate rendering
- Abstract-matched candidate rendering
- Individual synthesis error handling

---

## 2. Backend (Phase 4)

### Startup command

From the repository root:

```text
uvicorn src.phase4_backend.main:app --host 0.0.0.0 --port 8000