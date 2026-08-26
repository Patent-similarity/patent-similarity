# Phase 3 Findings — LLM-Based Patent Similarity Synthesis

**Status:** Core synthesis logic validated. Full-scale `(max_results=5)` timing and return shape confirmed with real live runs — see Section 7. Genuinely open: partial-results collection has been validated with an injected worker failure but not against a deliberately reproduced API-side failure; the evidence-integrity check is proven to catch non-substrings but not yet to reject meaningless-but-verbatim ones. The earlier `429` anomaly has been investigated and explained as a lower-level test bypassing the production wrapper. See Sections 3.1 and 8.
**Deliverable:** `src/phase3_synthesis/synthesis.py`

## 1. What Phase 3 does

Phase 3 sits on top of Phase 2's retrieval output. Instead of returning
raw similarity scores, it takes each of Phase 2's top Stage-2-eligible
candidates (up to 5, per `select_full_treatment`) and uses
`gemini-3.6-flash` to generate a structured, evidence-backed judgment
of how the candidate relates to the user's query.

## 2. Output schema

Each synthesized candidate returns exactly four fields:

- **verdict** — one of exactly four fixed categories, not free text:
  `HIGH_RELEVANCE`, `POSSIBLE_RELEVANCE`, `LOW_RELEVANCE`,
  `INSUFFICIENT_EVIDENCE`
- **overlap_summary** — the main technical overlap between query and
  candidate
- **key_difference** — the most important technical distinction
- **supporting_evidence** — `{quote, source}`, where `quote` must be an
  exact, contiguous, verbatim substring of the candidate's abstract or
  claims text, and `source` identifies which

**Evidence integrity is mechanically enforced, not just instructed.**
Every returned quote is checked in code (`validate_output`) against the
actual source text fed to the model — if the quote isn't a real
substring, the result is rejected. This was validated live across all
four verdict categories (see Section 3) with zero failures: every
returned quote verified as genuine.

`INSUFFICIENT_EVIDENCE` specifically means the candidate's text is too
generic/vague to support a reliable judgment — not that text is
missing (claims-missing candidates are already excluded upstream by
Phase 2's Stage 2 design, so every candidate reaching Phase 3 has both
abstract and claims text).

## 3. Live prompt validation across all four verdict categories

Before building the concurrency layer, the prompt itself was tested
live against real Gemini output on hand-picked toy-patent pairs
designed to stress each category:

- **HIGH_RELEVANCE** — near-identical query/patent pair. Correct
  verdict, quote verified.
- **LOW_RELEVANCE** — cross-domain pair (vision vs. NLP). Correct
  verdict; model correctly avoided treating shared buzzwords ("neural
  network") as meaningful overlap.
- **POSSIBLE_RELEVANCE** — partial domain overlap (object detection vs.
  medical imaging, shared mechanism "extracting visual features using a
  neural network"). Correct verdict, sitting properly between the other
  two on a real gradient.
- **INSUFFICIENT_EVIDENCE** — synthetic patent with deliberately
  generic, boilerplate claims language. Model correctly identified it as
  too vague rather than forcing a LOW_RELEVANCE verdict.

All four runs returned schema-valid JSON with genuinely verifiable
evidence quotes.

### 3.1 Corrupted-input test — reveals a validator gap, not a pass

During live concurrency testing (Section 7), one candidate's claims
text was deliberately corrupted, intending to force a genuine
evidence-validation failure (a quote that fails the substring check).
That did not happen. Instead, the model returned `INSUFFICIENT_EVIDENCE`
with the corrupted marker text itself as the "quote" — which is a
real, mechanically-verifiable substring of the (corrupted) source, so
`validate_output` correctly passed it.

**What this actually shows:** the substring check in Section 2 only
confirms a quote is verbatim-present in the source text — it says
nothing about whether that text is meaningful patent language. Feeding
it garbage and getting a technically-valid-but-content-meaningless
quote back is not evidence of graceful degradation; it's evidence that
the validator cannot distinguish real technical language from planted
nonsense, as long as it's copied verbatim. The original goal — forcing
a genuine validation *rejection* — still has not been achieved and
remains untested.

## 4. Concurrent synthesis

Candidates are synthesized independently and concurrently
(`ThreadPoolExecutor`), rather than one large combined request, so that:

- one candidate's failure doesn't block or invalidate the others
- results can be produced faster than strictly sequential calls
- the system doesn't assume a fixed candidate count (Phase 2 may return
  fewer than 5 Stage-2-eligible candidates by design — no backfill)

**Rank order is explicitly preserved** regardless of which worker
finishes first, by mapping each future back to its original position
(dict keyed by future → rank/index) rather than appending results in
completion order.

## 5. Failure-handling policy

**Decision:** if one candidate's synthesis fails (rate limit, timeout,
malformed response), `run_patent_similarity` returns **partial
results** — the other candidates' real syntheses, plus an explicit
error marker (`{"rank", "patent_id", "title", "error"}`) for the failed
one. The whole call never fails outright because one candidate errored.

## 6. Mock-based concurrency validation (0 API calls)

Before spending any live quota, the concurrency mechanism was validated
against a mock synthesis function covering:

- normal case (3 candidates)
- fewer than 5 candidates (tested at both 2 and 4)
- zero candidates (returns `[]`, no exception, no worker submitted)
- deliberate single-candidate failure (error captured correctly,
  successful candidates preserved, rank order intact)

All cases passed. No Gemini calls were made during this phase of
testing.

## 7. Live concurrency and timing

An initial live test was run against real Gemini calls at small scale:

- **1 candidate:** 41.56s
- **2 candidates concurrently (`max_workers=2`):** 45.80s

Since 2 concurrent candidates cost only ~4s more than 1 (not ~2x), this
confirmed **genuine parallel execution**, not silent serialization, and
gave an extrapolated estimate of 90–130s for a full 5-candidate result.

**That estimate has since been confirmed with two real full-scale
(`max_results=5`) runs, not just extrapolated:**

- **5/5 candidates succeeding:** ~130s
- **4/5 succeeding + 1 deliberately forced failure (see Section 8):**
  ~74.67s

Both runs used `max_workers=2`. This is now a measured result, not a
projection, and is the basis for recommending a **background-task
architecture** for the `/search` endpoint rather than a synchronous
blocking call — a multi-minute open HTTP connection is poor UX and
risks proxy/client-side timeouts regardless of what FastAPI itself
permits.

**Quota constraint on this model:** `gemini-3.6-flash` (synthesis) has
a separate, much tighter quota than the embedding model —
observed live at **5 RPM / 20 RPD** on the free tier. This is why
live synthesis testing is rationed carefully; a single full 5-candidate
run consumes a quarter of the daily budget.

## 8. Open items — not yet resolved

- **Partial-results collection mechanism confirmed live for a
  locally-raised exception; genuine API-side failure not deliberately
  reproduced.** A deterministic live test was run with one candidate's
  worker forced to raise (via a test-only wrapper that fails before
  any Gemini call, at zero extra quota cost) while the other four went
  through real synthesis. Result: 4 real syntheses + 1 correctly-shaped
  error marker `({"rank", "patent_id", "title", "error"})`, rank order
  preserved, confirmed with the deliberately hardest case (the forced
  failure landed on the rank-1 candidate). This validates that the
  collection/assembly logic correctly builds partial results when a
  worker raises. A genuine Gemini-side failure (malformed JSON, a real
  429, a timeout) has not been deliberately reproduced, so that specific
  failure mode remains untested.
  The earlier live test reported a `429 RESOURCE_EXHAUSTED` after one
  candidate completed in approximately 12.90s. Investigation showed
  that this test directly invoked the lower-level
  `synthesize_patent()` function rather than the production
  `run_patent_similarity()` wrapper. The lower-level function is
  intentionally designed to raise when its Gemini request fails, so
  the exception propagated to the test's main() function. The
  production wrapper separately catches completed worker exceptions,
  retains successful candidates, inserts an explicit error marker for
  the failed candidate, and restores the original Phase 2 ranking
  order. Therefore the 12.90s/429 result does not demonstrate a failure
  of the production partial-results mechanism.
- ~~Full `max_results=5` has not been run.~~ **Resolved** — see
  Section 7; run twice live (all-succeed and forced-partial-failure).
- ~~`stage1_candidates`/`stage2_candidates` were bare integer
  indices.~~ **Resolved and confirmed live** — both full-scale runs in
  Section 7 returned real `{patent_id, title, score}` objects.
- ~~Sync/async recommendation (Section 7) — communication status to
confirm~~. Resolved — the background-task recommendation was
incorporated into the Phase 4 job-based async search implementation.
The /search endpoint creates a job and the client can retrieve the
result through GET /search/{job_id}, avoiding a multi-minute
blocking HTTP request.

## 9. Explicitly out of scope for this document

Retrieval-quality evaluation (the S1–S3/D1–D3 known-similar/dissimilar
pair tests) exercises Phase 2's Stage 1/Stage 2 ranking, not Phase 3
synthesis, and is documented in `phase2_findings.md` /
`PROJECT_FINDINGS.md` instead of here.
