# Project Findings — Patent Similarity Search

## 1. Executive Summary

This project implements an end-to-end patent similarity search system that accepts a plain-language invention description and returns semantically similar patents with claims-aware ranking and structured LLM-generated relevance judgments.

The complete pipeline consists of:

1. Patent corpus construction and validation
2. Embedding-based similarity retrieval
3. Claims-aware reranking
4. LLM-based structured synthesis
5. Asynchronous FastAPI backend
6. Static web frontend and Render deployment

The core pipeline is **complete, integrated, evaluated, deployed, and frozen**.

Evaluation was performed on a real corpus of **6,980 US patents** in CPC subclass **G06T**, covering filings from **2015–2025**. The retrieval pipeline successfully recovered known similar patent partners near the top of the ranked results, while the LLM synthesis layer produced structured relevance judgments with mechanically validated evidence quotes.

The deployed application was also verified through a successful end-to-end production run on Render.

---

# 2. System Overview

The system follows a staged retrieval-and-synthesis architecture:

```text
Plain-language invention description
              │
              ▼
      Gemini Embedding Model
              │
              ▼
       FAISS Vector Search
       Stage 1 Retrieval
              │
              ▼
       Claims-aware Reranking
       0.3 × Abstract + 0.7 × Claims
              │
              ▼
       Top-ranked candidates
              │
              ▼
       Gemini Flash Synthesis
              │
              ▼
     Structured relevance results
              │
              ▼
        FastAPI Job API
              │
              ▼
       Static Web Frontend
```

The system separates high-recall vector retrieval from more expensive LLM reasoning. This allows the LLM to operate only on a small number of top candidates rather than the entire corpus.

---

# 3. Phase 1 — Corpus Construction

## Objective

Construct a reproducible patent corpus suitable for semantic similarity search while preserving provenance and claim availability information.

## Dataset

The final corpus contains:

* **6,980 US patents**
* CPC subclass: **G06T**
* Filing years: **2015–2025**
* **369 CPC subgroups** represented

Sampling was stratified by:

```text
(year, CPC subgroup)
```

This was used to avoid constructing a corpus dominated by only a small number of years or technology subgroups.

## Claims Coverage

Claims were successfully obtained for approximately:

* **64.2%** of patents

Approximately:

* **2,501 patents** had missing claims.

The project therefore records claim availability explicitly rather than silently treating missing claims as valid evidence.

## Abstract Coverage

Approximately:

* **201 patents (2.88%)** lack abstracts.

These records remain identifiable in the corpus and are documented as a known data limitation.

## Data Storage

The corpus and metadata were organized using SQLite, including claim-status and provenance information.

The corpus construction process was designed to be reproducible from the available USPTO bulk-data sources after direct PatentsView API access became impractical.

## Finding

The final corpus is sufficiently large and internally consistent for evaluating the retrieval pipeline. However, incomplete claims and a small number of missing abstracts introduce unavoidable data-quality limitations.

**Phase 1 status: COMPLETE.**

---

# 4. Phase 2 — Embedding Retrieval and Claims-Aware Reranking

## Objective

Retrieve semantically similar patents using embeddings and improve ranking by incorporating patent claims.

## Stage 1 — Abstract Retrieval

Patent abstracts were embedded using Gemini embeddings and indexed in FAISS.

The FAISS index uses:

```text
IndexFlatIP
```

with normalized embeddings, allowing inner-product similarity to correspond to cosine similarity.

Stage 1 therefore provides a high-recall semantic candidate set.

## Stage 2 — Claims-Aware Reranking

The retrieved candidates were reranked using:

```text
Final Score =
    0.3 × Abstract Similarity
  + 0.7 × Claim Similarity
```

Claims receive greater weight because they more directly represent the protected technical subject matter of a patent.

## Evaluation

The expanded primary evaluation used **16 real-corpus cases**, consisting of:

* 3 known genuine-positive similarity pairs
* 3 designated-wrong tests
* 10 constructed no-match queries

The three known genuine partners were all retrieved within the **top 0.18% of the corpus**.

The lowest observed genuine-positive cosine similarity was:

```text
0.784457
```

while the highest observed no-match similarity was:

```text
0.779555
```

This produced an observed gap of approximately:

```text
0.0049
```

## Threshold Finding

The observed score distributions overlap too closely to justify a production binary similarity threshold.

Therefore, the system does **not** classify patents as similar or dissimilar using an arbitrary cosine cutoff.

Instead, similarity scores are used for ranking, followed by claims-aware reranking and LLM-based analysis.

## Important Evaluation Note

The evaluation cases were derived from the project corpus and known relationships rather than from a large independently labeled benchmark dataset.

Therefore, these results demonstrate that the pipeline can recover known similarities and behave sensibly on constructed tests, but they should not be interpreted as a formal benchmark of patent-search recall or precision.

**Phase 2 status: COMPLETE AND FROZEN.**

---

# 5. Phase 3 — LLM Structured Synthesis

## Objective

Convert retrieved patent candidates into structured, interpretable relevance judgments.

For each candidate, the LLM produces:

* `HIGH_RELEVANCE`
* `POSSIBLE_RELEVANCE`
* `LOW_RELEVANCE`
* `INSUFFICIENT_EVIDENCE`

along with:

* overlap summary
* key difference
* supporting evidence quote

The synthesis stage operates only on the top-ranked retrieval candidates.

## Evidence Integrity

A key requirement was that evidence quoted by the LLM must actually occur in the supplied patent text.

The implementation therefore performs exact substring validation on the returned supporting quote.

Invalid quotes are rejected rather than being silently accepted as evidence.

## Validation

All four verdict categories were exercised during controlled validation.

The structured synthesis tests produced:

```text
Quote validation failures: 0
```

The synthesis layer also supports concurrent candidate processing while preserving candidate ranking.

Controlled failure testing confirmed that an individual candidate-synthesis failure does not require the entire search result to fail.

## Failure Handling

Per-candidate failures are represented explicitly, allowing successful candidates to remain available even if another synthesis request fails.

This behavior was validated during controlled Phase 3 testing.

The same implementation is deployed in the production application.

## Finding

The LLM layer adds interpretability to the numerical retrieval results while maintaining an explicit evidence-integrity check.

The evidence validator is intentionally conservative: it verifies literal textual presence rather than attempting to determine whether a quote is semantically representative.

**Phase 3 status: COMPLETE AND FROZEN.**

---

# 6. Phase 4 — FastAPI Backend

## Objective

Expose the search pipeline through an asynchronous HTTP API.

The backend provides:

```text
POST /search
GET  /search/{job_id}
GET  /health
GET  /ready
```

## Job Lifecycle

A search request creates a job and returns a job ID.

The job progresses through:

```text
pending → running → complete
```

The frontend polls the job endpoint until processing finishes.

## Backend Features

The backend includes:

* asynchronous background processing
* job status tracking
* structured search results
* per-candidate error reporting
* CORS configuration
* health endpoint
* readiness endpoint
* HTTP error handling

The API separates liveness from readiness:

* `/health` indicates that the service is alive.
* `/ready` indicates that required application initialization is ready.

## Finding

The API successfully integrates the retrieval and synthesis pipeline without requiring the frontend to maintain a long-running HTTP request.

**Phase 4 status: COMPLETE.**

---

# 7. Phase 5 — Frontend

## Objective

Provide a simple interface for submitting invention descriptions and viewing the resulting similarity analysis.

The frontend provides:

* invention-description input
* search submission
* job-status polling
* elapsed-time display
* ranked patent results
* synthesis verdicts
* overlap summaries
* key differences
* supporting evidence quotes
* per-candidate error display

The frontend communicates with the asynchronous FastAPI job API rather than attempting to perform the computationally expensive search directly in the browser.

## Finding

The frontend provides a functional presentation layer for the complete retrieval and synthesis pipeline and correctly reflects the asynchronous backend workflow.

**Phase 5 status: COMPLETE.**

---

# 8. Phase 6 — Deployment and Production Verification

## Objective

Deploy the complete application and verify that the system operates outside the local development environment.

## Deployment

The project was deployed using:

* Render backend service
* Render Static Site frontend

The frontend communicates with the deployed FastAPI backend.

## Production Verification

A real invention query was submitted through the deployed application.

The production run successfully demonstrated:

1. Frontend submission
2. Backend job creation
3. Job status progression
4. Retrieval of ranked patent candidates
5. LLM-generated relevance judgments
6. Overlap summaries
7. Key differences
8. Supporting claim/evidence quotes
9. Completion of the end-to-end search workflow

The observed production lifecycle was:

```text
pending → running → complete
```

This confirms that the major application components work together in the deployed environment.

## Controlled vs Production Validation

The following distinction is important:

| Capability                  | Validation                 |
| --------------------------- | -------------------------- |
| Frontend → deployed backend | Production                 |
| `/search` job creation      | Production                 |
| Job lifecycle               | Production                 |
| Ranked retrieval results    | Production                 |
| LLM synthesis results       | Production                 |
| Evidence quotes             | Production                 |
| Concurrent synthesis        | Controlled Phase 3 testing |
| Forced candidate failure    | Controlled Phase 3 testing |
| Full synthesis timing       | Controlled Phase 3 testing |

Graceful per-candidate failure handling was validated during controlled Phase 3 testing. The production deployment uses the same implementation, but an artificial synthesis failure was not required to establish successful production deployment.

## Production Performance

The primary production limitation is runtime.

A complete Render free-tier search can take approximately:

```text
10–15 minutes
```

The latency is primarily associated with embedding and LLM processing rather than the frontend or HTTP layer.

This is acceptable for demonstrating the asynchronous architecture, because the frontend does not need to keep a single synchronous request open during the entire computation.

## Finding

Phase 6 demonstrates that the complete system is deployable and operational outside the local development environment.

The successful production run provides evidence that the retrieval, reranking, synthesis, backend, and frontend layers are correctly integrated.

**Phase 6 status: COMPLETE.**

---

# 9. Cross-Phase Findings

## 9.1 Retrieval and LLM Reasoning Serve Different Roles

The system benefits from separating retrieval from reasoning.

FAISS provides efficient corpus-scale candidate retrieval, while the LLM provides deeper comparison only on a small candidate set.

This reduces the computational cost of applying LLM reasoning to the entire patent corpus.

---

## 9.2 Claims Improve the Search Objective

Abstract similarity alone is not necessarily sufficient for patent comparison.

The claims-aware second stage explicitly shifts ranking toward protected technical subject matter:

```text
30% abstract similarity
70% claim similarity
```

The retrieval architecture therefore reflects the distinction between general document similarity and claim-level technical overlap.

---

## 9.3 Similarity Scores Should Be Used for Ranking, Not Absolute Classification

The evaluation showed that genuine-positive and constructed no-match examples can have relatively close cosine scores.

Therefore:

```text
cosine similarity ≠ definitive legal similarity
```

The score is best treated as a ranking signal.

The LLM synthesis layer then provides additional structured analysis rather than relying on a hard numerical cutoff.

---

## 9.4 Evidence Validation Improves Trustworthiness

LLM-generated explanations can otherwise contain unsupported or hallucinated quotations.

Exact substring validation provides a simple mechanical safeguard:

```text
LLM quote
     │
     ▼
Exact substring check
     │
 ┌───┴────┐
 │        │
Valid    Invalid
 │        │
Accept   Reject
```

This does not prove that the quote is the best evidence, but it does ensure that accepted quotes originate from the supplied source text.

---

## 9.5 Asynchronous Processing Is Appropriate for the Workload

Because embedding generation and LLM synthesis can take substantial time, a synchronous API would provide a poor user experience.

The asynchronous job model allows:

```text
POST /search
      ↓
   job_id
      ↓
poll status
      ↓
retrieve results
```

This architecture also makes the long production runtime manageable.

---

# 10. Known Limitations

The following are known limitations of the current implementation.

## 10.1 Missing Claims

Approximately 35.8% of the corpus does not have available claims.

This limits claims-aware comparison for those records.

---

## 10.2 Missing Abstracts

Approximately 2.88% of patents lack abstracts.

The current system does not implement a separate title-only or alternative embedding fallback for these records.

---

## 10.3 No-Match Classification

The evaluation does not establish a reliable universal cosine threshold for declaring that no similar patent exists.

The system therefore returns ranked candidates rather than making a definitive absence-of-prior-art claim.

---

## 10.4 Evaluation Scale

The primary evaluation contains 16 cases.

Although it includes genuine-positive, designated-wrong, and no-match cases, it is not large enough to establish general statistical performance across the patent domain.

---

## 10.5 Corpus-Derived Ground Truth

The evaluation examples were constructed from the project corpus and known relationships rather than a large independently labeled benchmark.

The results should therefore be interpreted as validation of system behavior rather than as a formal industry benchmark.

---

## 10.6 Literal Evidence Validation

The quote validator checks whether an LLM-provided quote occurs literally in the supplied text.

It does not evaluate:

* whether the quote is the strongest evidence
* whether the quote fully supports the LLM's conclusion
* whether the legal interpretation is correct

Human or semantic evaluation would be required for those questions.

---

## 10.7 Historical Coverage

Some earlier patent records have incomplete claim availability in the available bulk-data sources.

This creates an additional limitation when comparing the current corpus against older prior art.

---

## 10.8 Embedding and LLM Latency

The deployed free-tier configuration can require approximately 10–15 minutes for a complete search.

The architecture is asynchronous specifically to accommodate this workload.

---

## 10.9 In-Memory Job State

The FastAPI implementation currently stores job state in memory.

Consequently, restarting the backend can remove active job information, and the design is not intended for multi-instance production scaling.

A durable job store would be appropriate for a larger deployment.

---

# 11. Future Improvements

The following are potential extensions rather than prerequisites for the completed project.

### Data

* Add a title-based fallback for patents without abstracts.
* Improve historical claims coverage.
* Expand the corpus beyond the current G06T scope.

### Evaluation

* Build a larger independently labeled evaluation set.
* Measure precision, recall, MRR, and/or Recall@K.
* Perform a formal semantic evaluation of evidence quality.
* Expand the no-match test set.

### Retrieval

* Experiment with additional weighting strategies.
* Evaluate alternative embedding models.
* Investigate batch embedding to reduce latency.
* Compare additional vector-index configurations.

### Production Infrastructure

* Replace in-memory job storage with a durable job queue/store.
* Support multiple backend workers.
* Add authentication and rate limiting if exposed to unrestricted public traffic.
* Improve caching and request batching.

These improvements are intentionally outside the frozen core implementation.

---

# 12. Final Project Assessment

The project successfully demonstrates a complete patent similarity-search workflow:

```text
Patent corpus
     ↓
Data validation
     ↓
Gemini embeddings
     ↓
FAISS retrieval
     ↓
Claims-aware reranking
     ↓
LLM structured synthesis
     ↓
Evidence validation
     ↓
FastAPI asynchronous API
     ↓
Web frontend
     ↓
Render deployment
     ↓
Production verification
```

The main technical findings are:

1. **Semantic retrieval successfully identifies known similar patents near the top of the corpus.**
2. **Claims-aware reranking provides a deliberate mechanism for emphasizing protected technical subject matter.**
3. **A universal cosine-similarity threshold is not supported by the current evaluation.**
4. **LLM synthesis provides interpretable relevance judgments and explanations.**
5. **Exact quote validation provides a mechanical safeguard against unsupported evidence quotations.**
6. **The asynchronous job architecture is appropriate for the computational cost of embedding and LLM processing.**
7. **The complete application has been deployed and successfully verified through a real production search.**

## Final Status

**Core project: COMPLETE AND FROZEN**

All six phases have been implemented, evaluated, integrated, deployed, and verified.

The remaining limitations and future improvements documented above do not represent unfinished core phases. They describe opportunities for expanding evaluation coverage, improving data completeness, reducing latency, and preparing the system for larger-scale production use.
