# Phase 5 - Frontend

Static HTML/CSS/JS UI for the patent similarity search API. No framework, no build step.

## Running it

Just open index.html directly in a browser, or serve the folder as static files with any web server.

Requires the Phase 4 backend running and reachable (see config.js).

## Configuration

config.js sets BACKEND_URL - edit this one file to point at a different backend (e.g. before deploying).

## Files

- index.html - page structure, search form, results layout
- style.css - styling
- config.js - BACKEND_URL setting (loaded before app.js)
- app.js - search submission, polling GET /search/{job_id}, results rendering, error handling

## Behavior notes

- A search takes 75-130+ seconds (real Gemini calls) - the UI polls automatically and shows an elapsed-time indicator while waiting.
- Results render in three sections: Synthesized results (full verdict analysis, top candidates), Additional claims-matched candidates (scored, not fully analyzed), Additional abstract-matched candidates (broadest match, abstract only).
- Individual candidate synthesis failures (e.g. Gemini rate limits) show as a per-candidate error message, not a broken page.
