# Phase 4 - Backend (FastAPI)

Free-text patent similarity search API. Accepts an invention description and returns ranked similar patents with synthesized verdicts.

## Status

Backed by the real Phase 2/3 pipeline (real FAISS index, real Gemini synthesis). Verified end-to-end with real data.

## Running it

From the repo root:

    uvicorn src.phase4_backend.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs (Swagger UI).

## Architecture

Job-based async pattern. A search kicks off a background job immediately; poll for the result separately. This exists because a real search takes 75-130+ seconds (real Gemini calls), so it cannot be a single blocking HTTP request.

## Endpoints

- GET /health - liveness check
- POST /search - starts a search job, returns immediately
- GET /search/{job_id} - polls job status/result, never re-runs the pipeline

### POST /search request

    { "query": "A method for rendering 3D graphics using photon mapping techniques" }

Returns: { job_id, status: "pending" }

### GET /search/{job_id} response

status is one of: pending, running, complete, failed.

On complete, result contains three separate lists:
- stage1_candidates - all abstract-similarity hits (patent_id, title, score)
- stage2_candidates - all claims-reranked candidates (patent_id, title, score)
- results - the top synthesized candidates only, each with rank, patent_id, title, abstract_score, claim_score, final_score, and EITHER a synthesis object (verdict, overlap_summary, key_difference, supporting_evidence) OR an error string if that candidate's synthesis failed

verdict is one of: HIGH_RELEVANCE, POSSIBLE_RELEVANCE, LOW_RELEVANCE, INSUFFICIENT_EVIDENCE

Note: individual candidate synthesis failures (e.g. Gemini rate limits) do NOT fail the whole job - they show up as a per-candidate error field instead.

## Startup dependencies

Must exist before the server boots correctly:
- patent_similarity.faiss
- patent_similarity_metadata.json
- data/patents.db
- GEMINI_API_KEY set in .env

## Files

- main.py - FastAPI app, routes, CORS middleware, startup loading, error handling
- models.py - Pydantic request/response models (the API contract)
- job_store.py - in-memory job state (not persistent, not multi-process-safe - fine for this project's scope)
