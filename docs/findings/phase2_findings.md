# Phase 2 Findings — Two-Stage Retrieval Validation

**Status:** Validated on toy data; real-data validation in progress
(loader, restructuring, and rate limiting confirmed working; live
retrieval quality testing partially complete, blocked on daily API
quota, not on any unresolved design or code issue).
**Deliverable:** `src/phase2_embedding_retrieval/embedding_pipeline.py`,
`data/real_corpus.py`, `test/phase2-test.ipynb`,
`test/test_real_smoke.py`, `test/test_eval_pairs_real.py`

## What this validates

Phase 2 owns the retrieval design: Stage 1 embeds the query and every
patent's **abstract only** (Gemini `gemini-embedding-001`), searches
with FAISS `IndexFlatIP` (cosine via normalized inner product), and
takes the top-K candidates forward. Stage 2 embeds **claims** for those
candidates only and reranks with
`final_score = 0.3 * abstract_score + 0.7 * claim_score`.

Before real data was available, a 5-patent toy corpus (CNN image
classifier, tumor detection, autonomous-vehicle object detection, NLP
summarization, facial recognition access control) validated the
mechanics of the two-stage design — not real patent recall. The goal
was to catch design and implementation bugs early, since the retrieval
logic doesn't depend on which corpus it runs against; only the
data-loading step changes.

## A bug found and fixed early

The first version of `patent_to_text` concatenated title + abstract +
claims for Stage 1. That's wrong — Stage 1 is abstract-only, with
claims held back for Stage 2. Caught by inspecting the embedding input
directly, fixed before any evaluation ran. The corrected function has
an inline comment explaining why it's abstract-only, so it can't
regress silently.

## Toy-data evaluation method

8 queries, expected results locked before running anything: 5 clean
single-match queries, 1 deliberately ambiguous query, 1 no-match query,
1 near-duplicate phrasing check.

## Toy-data results

7 of 7 in-domain queries retrieved the correct patent at rank 1, at
both stages. Claims reranking never flipped a correct Stage 1 result
into an incorrect one.

Example (Q8, phrased close to P001's own claim text): Stage 1 top
score 0.8060 (correct) → Stage 2 top score 0.9240 (correct, reinforced).

**Ambiguity stress test (Q6):** query deliberately overlaps P003 and
P005. Stage 1 alone gave a thin margin (0.8039 vs 0.7395, gap 0.0644).
Claims reranking widened that gap to 0.1307 — more than double —
because claims text makes the application explicit in a way the
abstract alone doesn't. Concrete evidence Stage 2 does real
disambiguating work, not just noise.

**No-match query (Q7):** absolute scores stayed flat and low (top
score 0.6631) relative to genuine matches (0.8060+). That gap is the
signal a no-match detector should key off — see limitations.

## Missing-claims exclusion — validated

Patents with `claims: None` are excluded from Stage 2 entirely (never
embedded, never scored, never reranked), while remaining eligible for
Stage 1. `select_full_treatment(stage2_indices, max_results=5)` selects
up to 5 Stage-2-eligible candidates, **with no backfilling** from
Stage-1-only candidates if fewer than 5 are eligible.

Toy corpus expanded from 5 to 8 patents (3 synthetic claims-missing
patents added) to test this. Full 8-query regression held with all
original results unchanged. Cap behavior verified at both 5-eligible
and 4-eligible counts — the no-backfill rule holds rather than padding.

**Observation:** claims-missing patents can outrank claims-eligible
ones in Stage 1's abstract-only view. With the real corpus's ~35.8%
claims-missing rate (see below — this was an early ~29% estimate,
now confirmed against real data), Stage 1's top-K will likely include
a non-trivial share of claims-ineligible candidates, meaning the
number of Stage-2-eligible candidates per query could land well below
5 in practice. The no-backfill policy is designed for exactly this.

## Stage 2 restructured for real-data efficiency

The original design precomputed claims embeddings for every patent in
`build_indexes`, unconditionally — fine at 5–8 toy patents, wasteful
and wrong at real scale (would mean ~7,000 claims embeddings per corpus
build, most never used). Restructured so claims are embedded only
inside `evaluate_query`, scoped to that query's top-K Stage 1 survivors,
after excluding claims-missing candidates — never more than K claims
embeddings per query, never for the whole corpus.

Validated: full 8-query toy regression reproduced identical results to
the pre-restructuring version (same rankings, same scores), confirming
the refactor changed only internals, not behavior. Also validated
against a 20-patent real-data slice — correct exclusion of
claims-missing candidates, correct reranking of eligible ones.

## Real-data loader

`data/real_corpus.py` reads `patents.db`, excludes rows with missing
abstracts, preserves `claims: None` for missing claims so the
pipeline's exclusion logic handles them correctly.

**Bug found and fixed:** the loader originally filtered on an
`abstract_status` column that does not exist in the rebuilt corpus
schema (Phase 1's subgroup-stratification fix changed the schema).
Fixed to filter directly on the `abstract` field
(`IS NOT NULL AND TRIM(abstract) != ''`). Verified: 6,980 total rows,
6,779 loaded after exclusion — exactly matching Phase 1's documented
201 missing-abstract count.

## Rate limiting — real constraint discovered and fixed

Initial real-data runs hit `429 RESOURCE_EXHAUSTED` errors. Root cause:
`gemini-embedding-001`'s free tier enforces a **100 requests/minute**
limit (confirmed directly via Google AI Studio's rate-limit dashboard),
not just the 1,000/day limit assumed in earlier project planning.
Neither `build_indexes` nor `evaluate_query` had any request pacing.

Fixed by adding `time.sleep(delay)` (default 0.7s) after each embedding
call in both functions. This is a stopgap for moderate volume, not a
substitute for real batching (`batchEmbedContents`), which is still
needed before running the full 6,779-patent corpus in one pass.

## Live real-data eval-pair testing — in progress

Six known-similar/known-dissimilar pairs were selected independently
from the real corpus (full abstract and claims text read for every
candidate before inclusion, not just titles — one pair's misleadingly
generic title was specifically checked against real text before being
kept; every candidate's `claims_status` confirmed `found`). See
`PROJECT_FINDINGS.md` for the full pair list and an important open
item: a second, independently-selected eval-pair set also exists from
Phase 1, not yet reconciled with this one.

**S1 result** (query: `11615557`, partner: `11620767`, both point-cloud
octree compression patents, tested against a 90-patent slice of the
real corpus): partner ranked **position 2 of 90 in both Stage 1 and
Stage 2**. First evidence this pipeline works on real, independently-
sourced patent text, not just hand-written toy data.

The remaining five pairs (S2, S3, D1, D2, D3) are queued, blocked only
on daily API quota — the rate-limiting fix is in place and correct,
this is purely a "wait for tomorrow's quota reset" blocker.

## Limitations — read before further real-data work

- **No-match detection is designed, not calibrated.** AND-gate on
  absolute top score + top-2 margin, checked against a single toy
  no-match case. Needs real score distributions from the full real
  corpus before it's trustworthy.
- **Missing-abstract fallback not implemented.** 201 real patents are
  currently excluded entirely rather than falling back to title-only
  embedding, despite Phase 1's findings recommending that fallback.
- **Real batching not implemented.** Current pacing (`time.sleep`)
  works for moderate test volumes but is not a substitute for batched
  requests at full-corpus scale.
- **Eval-pair reconciliation pending** — see `PROJECT_FINDINGS.md`.

## Next steps

- Finish the remaining 5 eval-pair retrieval tests once daily quota
  resets.
- Implement title-only fallback for missing-abstract patents.
- Recalibrate no-match thresholds against real score distributions once
  full-corpus retrieval data exists.
- Implement real batching (`batchEmbedContents`) before running the
  full 6,779-patent corpus in one pass.
- Reconcile the two independently-selected eval-pair sets with the
  Phase 1 owner.