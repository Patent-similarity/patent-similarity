# Phase 4 -- Backend (FastAPI)

Free-text patent similarity search API. Accepts an invention description
and returns ranked similar patents with synthesized verdicts.

## Status

Currently backed by mock data (mock_data.py), not the real Phase 2/3
pipeline. The response schema below is confirmed to match the partner's
actual Phase 2 (retrieval) and Phase 3 (synthesis) output, but the data
itself is hardcoded for now.

TODO: swap get_mock_search_response() in main.py for the real
pipeline call once Phase 2/3 exposes a callable interface.

## Running it

    uvicorn main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs
(Swagger UI) -- you can test endpoints directly from the browser there.

## Endpoints

- GET /health -- liveness check
- POST /search -- main search endpoint

### Request

    {
      "query_text": "A method for rendering 3D graphics using photon mapping techniques",
      "top_k": 5
    }

query_text must be at least 10 characters. top_k defaults to 5, max 20.

### Response

Results are split into two tiers:

- full_treatment_results -- top 5 (max), Stage-2-eligible results, each
  with a full synthesis: verdict (one of HIGH_RELEVANCE,
  POSSIBLE_RELEVANCE, LOW_RELEVANCE, INSUFFICIENT_EVIDENCE),
  overlap_summary, key_difference, and supporting_evidence
  (a verbatim quote plus its source: claim or abstract).
- additional_matches -- ranks 6-50, Stage-1-only, no synthesis --
  just patent_id, title, score.

## Files

- main.py -- FastAPI app, routes, CORS middleware, error handling
- models.py -- Pydantic request/response models (the API contract)
- mock_data.py -- placeholder data matching the confirmed real schema