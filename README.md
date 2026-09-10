# Patent Similarity Search

[**Live Demo →**](https://patentsimilarity.onrender.com/)

> **Demo note:** The deployed demo may take several minutes to complete a search because embedding generation and LLM synthesis run on the free-tier backend.

A full-stack system that takes a plain-language description of an invention, retrieves technically related US patents from a real corpus, and produces an LLM-synthesized relevance judgment for each candidate — deployed end-to-end as a public web application.

**Status:** **Phases 1–6 complete.** The core retrieval and synthesis pipeline is frozen, validated on real data, deployed, and confirmed working end-to-end in production on Render.

---

## 1. What it does

1. A user enters a free-text invention description.
2. The system embeds the query and retrieves technically related patents from an initial corpus of **6,980 US patents**, filtered to **6,779 patents** for the retrieval pipeline (CPC subclass **G06T** — image data processing / computer vision, filed 2015–2025).
3. Candidates are reranked using their claims text for stronger, invention-specific evidence.
4. The top candidates are passed to an LLM, which returns a structured verdict (`HIGH_RELEVANCE` / `POSSIBLE_RELEVANCE` / `LOW_RELEVANCE` / `INSUFFICIENT_EVIDENCE`), an overlap summary, a key difference, and a verbatim, mechanically verified supporting quote from the patent's own text.
5. Results are served asynchronously through a job-based API and rendered in a web frontend that polls for completion.

---

## 2. Architecture

```text
User query (free text)
        │
        ▼
Stage 1 — Abstract retrieval (Phase 2)
  Gemini embeddings + FAISS (IndexFlatIP, cosine)
  Searches all patents' abstracts → top-K candidates
        │
        ▼
Stage 2 — Claims-aware rerank (Phase 2)
  Embeds claims for the top-K candidates only
  final_score = 0.3 × abstract_score + 0.7 × claim_score
        │
        ▼
Phase 3 — LLM synthesis
  Gemini Flash, up to 5 candidates, run concurrently
  verdict / overlap / key difference / verified quote
        │
        ▼
Phase 4 — FastAPI backend (Render)
  POST /search → job_id
  GET /search/{job_id} → pending → running → complete
        │
        ▼
Phase 5 — Static frontend (Render)
  Submits query, polls job status, renders synthesized results
```

The layered design is deliberate: embedding similarity is treated as a **retrieval and ranking signal**, not a calibrated match probability. The final relevance judgment is delegated to the evidence-backed synthesis layer rather than a single cosine-similarity threshold (see Phase 2 findings, §15–17).

---

## Team Contributions

The project was developed collaboratively, with responsibilities divided across the system's development phases.

### Swayam Surve

* **Phase 2 — Embedding + Retrieval:** Implemented the Gemini embedding pipeline, FAISS-based retrieval, and claims-aware reranking. Conducted retrieval evaluation and finalized the retrieval configuration.
* **Phase 3 — Ranking + Synthesis:** Implemented the LLM-based structured synthesis layer, relevance verdicts, evidence extraction/validation, and candidate-level failure handling.
* **Phase 6 — Deployment:** Deployed the backend/frontend workflow to Render and performed end-to-end production verification.

### Ayush Shirke

* **Phase 1 — Data Pipeline:** Built the patent corpus pipeline, including data collection, cleaning, structuring, provenance, and claims/abstract availability tracking.
* **Phase 4 — Backend/API:** Implemented the FastAPI service, asynchronous job lifecycle, API endpoints, health/readiness checks, and backend integration.
* **Phase 5 — Frontend/Interface:** Implemented the web interface for submitting searches, polling job status, and displaying ranked and synthesized results.

---

## 3. Phase-by-phase summary

Full detail for each phase is documented in the corresponding findings document. This section summarizes the main deliverables and final status.

### Phase 1 — Corpus construction
**[phase1_findings.md](./docs/findings/phase1_findings.md)**

- Built a **6,980-patent** corpus (CPC subclass G06T, filed 2015–2025) from USPTO Bulk Data after the PatentsView API became inaccessible without ID.me verification.
- Used stratified sampling by `(year, CPC subgroup)` across 369 distinct subgroups. The final corpus size of 6,980 rather than the original 10,000 target was a deliberate tradeoff to preserve representation across subgroups rather than allow common subgroups to dominate.
- Claims text was found for 64.2% of patents; **2,501 records lack claims**, concentrated in 2015–2021 due to a documented gap in USPTO bulk long-text data for older grants.
- **201 patents (2.88%) lack an abstract.**
- Delivered `patents.db` (SQLite) with `claims_status` and provenance tracking rather than silently dropping incomplete records.

### Phase 2 — Two-stage retrieval
**[phase2_findings.md](./docs/findings/phase2_findings.md)**

- Implemented Stage 1 abstract retrieval using FAISS and Stage 2 claims-aware reranking using a **30/70 abstract/claims weighting**.
- Validated the retrieval implementation first on toy data (**7/7 correct top-1 retrieval**) and then on the full real corpus.
- The primary full-corpus evaluation used **16 real cases**: 3 genuine-positive pairs, 3 designated-wrong tests, and 10 constructed no-match queries.
- All 3 known genuine partners were retrieved within the **top 0.18% of the corpus**.
- Stage 2 reranking preserved or improved the rankings of the known positive partners.
- **Key finding:** cosine similarity does not cleanly separate genuine matches from no-match queries at the tail. The lowest genuine-positive score was **0.784457**, while the highest no-match score was **0.779555**, a gap of only ~0.0049.
- Consequently, no production-grade similarity threshold is used for the final relevance decision; relevance is deferred to Phase 3 synthesis.
- **Phase 2 status: COMPLETE / FROZEN.**

### Phase 3 — LLM-based synthesis
**[phase3_findings.md](./docs/findings/phase3_findings.md)**

- For each of Phase 2's top candidates, the LLM generates a four-field structured judgment: verdict, overlap summary, key difference, and supporting quote.
- **Evidence integrity is mechanically enforced:** every returned quote is checked against the actual source text and rejected if it is not a genuine verbatim substring.
- The evidence validation mechanism was tested across all four verdict categories with **zero validation failures**.
- Candidate synthesis runs concurrently rather than sequentially, while preserving candidate rank order.
- Individual synthesis failures are isolated so that one failed candidate does not prevent the remaining candidates from being returned.
- Full-scale local runs with `max_results=5` measured approximately **130 seconds for 5/5 successful candidates** and **75 seconds for 4/5 candidates with one forced failure**, demonstrating genuine parallel execution and motivating the asynchronous job architecture in Phase 4.
- **Phase 3 status: COMPLETE / FROZEN.**

### Phase 4 — Backend
**[phase4_5_deployment_facts.md](./docs/findings/phase4_5_deployment_facts.md)**

- Implemented a FastAPI backend exposing a job-based asynchronous search API.
- `POST /search` creates a job and returns a `job_id`.
- `GET /search/{job_id}` exposes the job lifecycle:
  `pending → running → complete`.
- Includes CORS configuration, error handling, and startup/readiness validation.
- The asynchronous architecture avoids keeping a client HTTP request open during the multi-minute retrieval and synthesis process.
- **Phase 4 status: COMPLETE.**

### Phase 5 — Frontend
**[phase4_5_deployment_facts.md](./docs/findings/phase4_5_deployment_facts.md)**

- Implemented a static HTML/CSS/JavaScript frontend.
- Provides a search form, automatic job polling, elapsed-time display, and synthesized result rendering.
- Displays verdict, overlap, key difference, and quoted evidence for each candidate.
- Supports graceful per-candidate error display when an individual synthesis fails.
- **Phase 5 status: COMPLETE.**

### Phase 6 — Deployment & end-to-end verification
**[phase6_findings.md](./docs/findings/phase6_findings.md)**

- Deployed the FastAPI backend on Render and the frontend as a Render Static Site.
- Connected the frontend to the live backend and verified the complete production workflow.
- A real invention query submitted through the deployed application completed successfully and returned ranked candidates, synthesized relevance judgments, overlap/key-difference reasoning, and verified patent-text quotes.
- Confirmed the asynchronous job lifecycle:
  `pending → running → complete`.
- Graceful per-candidate failure handling was validated during controlled Phase 3 testing; the same implementation is deployed in Phase 6.
- **Phase 6 status: COMPLETE.**
- **Deployment limitation:** on the free-tier Render infrastructure, a full search takes approximately **10–15 minutes end-to-end**, primarily due to embedding and LLM processing rather than frontend or HTTP request handling.

---

## 4. Evaluation and cross-phase findings

**[PROJECT_FINDINGS.md](./docs/PROJECT_FINDINGS.md)** contains the detailed cross-phase audit, evaluation caveats, and future-improvement notes.

### Evaluation methodology

Two evaluation sets were used during development:

- **Phase 1:** preliminary exploratory evaluation using 3 known similar pairs.
- **Phase 2:** primary evaluation using the expanded 16-case real-corpus evaluation described above.

The evaluation pairs are corpus-derived rather than independently labeled ground truth such as examiner-cited prior art or citation relationships. Therefore, the results demonstrate retrieval behavior and engineering validation, but should not be interpreted as a formal benchmark of patent-similarity accuracy.

### Current cross-phase limitations

- **Missing abstracts:** 201 patents (2.88%) lack an abstract and are currently excluded from Stage 1 retrieval.
- **Missing claims:** 2,501 patents (~35.8%) lack claims text and therefore cannot participate in claims-aware Stage 2 reranking.
- **No-match classification:** absolute similarity score and margin were investigated, but the evaluation did not establish a sufficiently reliable threshold for automatic no-match classification.
- **Embedding batching:** request pacing is implemented; true batch embedding (`batchEmbedContents`) is not currently used and would become more relevant if the corpus were scaled substantially.
- **Historical coverage:** claims bulk files for 2011–2014 were unavailable during corpus construction, so earlier coverage is incomplete.
- **Evidence validation:** quote verification guarantees literal presence in the source text, but does not by itself establish that the quoted passage is the most semantically informative evidence.
- **Production latency:** the current free-tier deployment takes approximately 10–15 minutes for a full search.

These are documented limitations of the current system rather than blockers to the completed core pipeline.

---

## 5. Running the project

### Backend

From the repository root:

```bash
uvicorn src.phase4_backend.main:app --host 0.0.0.0 --port 8000
```

### API

| Endpoint | Method | Description |
|---|---|---|
| `/search` | `POST` | Submit an invention description; returns a `job_id` |
| `/search/{job_id}` | `GET` | Poll job status: `pending` → `running` → `complete`; `result` is populated on completion |
| `/health` | `GET` | Liveness check |
| `/ready` | `GET` | Readiness check for required runtime components |

### Frontend

The frontend is a static HTML/CSS/JavaScript site and can be deployed to a static host. The deployed version is hosted on Render and configured to call the public backend URL.

---

## 6. Known limitations

The following limitations are intentionally retained rather than hidden because they describe the actual behavior and scope of the system:

- Similarity scores are a **ranking signal, not a calibrated match probability**.
- Approximately **35.8% of the corpus lacks claims text** and is therefore excluded from claims-aware reranking.
- Approximately **2.9% of the corpus lacks an abstract** and is currently excluded from retrieval.
- Full search latency on the current free-tier deployment is approximately **10–15 minutes**.
- Individual LLM synthesis calls can fail; the system is designed to degrade gracefully and return partial results rather than fail the entire job.
- The evaluation is based on corpus-derived test cases rather than an independent benchmark of patent prior-art relevance.

---

## 7. Future improvements

The following are potential extensions rather than unfinished core phases:

- Add title-only fallback retrieval for patents without abstracts.
- Improve no-match detection using a larger independently labeled evaluation set.
- Reconcile the preliminary Phase 1 evaluation with the primary Phase 2 evaluation into a single benchmark.
- Add true embedding batching for larger corpora or higher API throughput.
- Improve semantic evaluation of generated evidence in addition to literal quote validation.
- Extend corpus coverage to earlier filing years where source data becomes available.
- Replace the current in-memory job store with durable distributed job infrastructure if the application is scaled beyond the current deployment model.

---

## 8. Repository structure

```text
src/phase2_embedding_retrieval/embedding_pipeline.py   # Stage 1 + Stage 2 retrieval
src/phase3_llm_synthesis/phase3.py                     # LLM synthesis + evidence validation
src/phase4_backend/main.py                             # FastAPI app, job API
data/real_corpus.py                                    # Real corpus loader
test/                                                   # Toy-data, smoke, and real-evaluation tests
docs/
├── PROJECT_FINDINGS.md
└── findings/
    ├── phase1_findings.md
    ├── phase2_findings.md
    ├── phase3_findings.md
    ├── phase4_5_deployment_facts.md
    └── phase6_findings.md
```

---

## 9. Project status

**Core project: COMPLETE and FROZEN.**

Phases 1–6 have been implemented, evaluated, integrated, deployed, and verified end-to-end on real data. The remaining items documented above are known limitations and potential future improvements, not prerequisites for the current system to function.

The project demonstrates a complete pipeline from **real patent corpus construction → semantic retrieval → claims-aware reranking → evidence-grounded LLM synthesis → asynchronous API → deployed web frontend**.
