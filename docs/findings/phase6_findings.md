# Phase 6 — Findings

The patent similarity search system was deployed and tested as an end-to-end web application, validating the full pipeline from public frontend to backend job processing.

## Deployment Findings

- The FastAPI backend was deployed on Render and is accessible via a public API endpoint.
- The frontend was deployed as a Render Static Site and successfully connected to the backend.
- The frontend was able to initiate patent similarity search jobs and display synthesized results directly in the UI.
- The backend's health and readiness endpoints (`/health`, `/ready`) consistently returned HTTP 200 during testing.

## Search Pipeline Findings

- The deployed system executed the two-stage patent retrieval pipeline: (1) abstract-based similarity retrieval, (2) claims-based re-ranking using claim embeddings.
- A completed test run — query: *"A system that detects images using camera"* — returned ranked candidate patents with abstract, claims, and final relevance scores, along with overlap and key-difference reasoning for each match.
- The frontend correctly polled the backend job endpoint (`GET /search/{job_id}`) throughout execution, confirming the asynchronous workflow (`pending → running → completed`) functions as designed.
- The system processed the real patent corpus and returned full synthesized results through the deployed application, including quoted claim excerpts supporting each match.

## Performance Findings

- The full search process takes noticeably longer on Render's free tier than in local testing.
- A completed run took approximately 10–15 minutes end-to-end.
- The bottleneck is the embedding/synthesis stage, not frontend or backend request handling.
- Frontend status messaging was updated to set user expectations for this longer processing window.

## Reliability Findings

- The backend remained available throughout testing, with `/health` and `/ready` returning 200 OK.
- Repeated polling of the job-status endpoint also returned 200 OK without errors.
- In the observed run, two candidate patents failed individual Gemini synthesis requests ("Synthesis failed for this candidate"), while the pipeline continued and successfully returned synthesized results for the remaining candidates — confirming that isolated synthesis failures do not halt the overall search job.

## Overall Finding

Phase 6 demonstrates that the patent similarity system operates correctly as a publicly accessible, end-to-end application. The retrieval, re-ranking, and synthesis pipeline functions properly in the deployed environment and degrades gracefully on individual candidate failures. The main limitation identified is the ~10–15 minute processing time on free-tier infrastructure.
