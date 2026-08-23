# Project Findings — Patent Similarity Search

This document ties together Phase 1 (corpus construction) and Phase 2
(retrieval pipeline) findings. Full detail lives in the per-phase docs
linked below; this page connects what each phase produced to what the
next phase actually needed.

## Phase 1 → Phase 2 handoff

Phase 1 delivered a 6,980-patent G06T corpus, stratified by
(year, CPC subgroup), stored in `patents.db`. Two things about that
handoff are worth stating plainly here, since they affected Phase 2
directly:

- **Corpus size (6,980, not the original 10,000 target) is a deliberate
  tradeoff, not a shortfall.** Chasing 10,000 would have meant loosening
  per-cell sampling caps, letting common CPC subgroups absorb the extra
  headroom while rare subgroups stayed thin — reintroducing the exact
  imbalance the subgroup-stratification fix existed to solve. See
  [phase1_findings.md, Section 5.5] for the full investigation.
- **The corpus schema does not include an `abstract_status` column**,
  despite earlier project discussion assuming one would exist. Phase 2's
  real-data loader (`data/real_corpus.py`) originally filtered on this
  column and would have failed immediately against the real schema; this
  was caught and fixed by filtering directly on the `abstract` field
  instead (`IS NOT NULL AND TRIM(abstract) != ''`). 201 patents are
  excluded this way, matching Phase 1's documented missing-abstract count
  exactly (6,980 → 6,779 loaded).

Phase 2 has not yet implemented the title-only fallback Phase 1's
findings recommend for these 201 patents (see "Open items" below) —
they are currently excluded from retrieval entirely, not degraded
gracefully.

## Phase 2 status

Two-stage retrieval (Stage 1: abstract-only FAISS search; Stage 2:
claims-weighted rerank on Stage 1's top-K, 30/70 weighting) is fully
implemented and validated in stages:

1. **Toy-data validation** (5→8 hand-written patents): 7/7 correct
   top-1 retrieval, a documented and fixed abstract/claims concatenation
   bug, missing-claims exclusion (Stage 2 never scores a claims-missing
   candidate), and a no-backfill cap (`select_full_treatment` returns
   fewer than 5 results rather than padding from Stage 1 when fewer than
   5 candidates are Stage-2-eligible). Full detail in
   [phase2_findings.md].
2. **Real-data validation, in progress.** The real corpus loader was
   tested against the actual `patents.db` (6,779 patents loaded,
   arithmetic cross-checked against Phase 1's documented counts). Stage
   2 was restructured to embed claims only for a query's top-K Stage 1
   candidates, not the whole corpus upfront, cutting embedding cost from
   ~7,000 calls per corpus build to ≤50 calls per query. A live retrieval
   test against one of the six independently-verified eval pairs (see
   below) returned the known-similar partner patent at **position 2 of
   90** in both Stage 1 and Stage 2 — the first evidence this pipeline
   works on real, independently-sourced patent text, not just
   hand-written toy data. The remaining five pairs are queued, blocked
   only on daily API quota, not on any unresolved design or code issue.
3. **Phase 3 prompt design (four-field synthesis: verdict, overlap,
   key difference, evidence)** has been validated live against the
   Gemini API across all four verdict categories
   (`HIGH_RELEVANCE`, `POSSIBLE_RELEVANCE`, `LOW_RELEVANCE`,
   `INSUFFICIENT_EVIDENCE`), with every returned evidence quote
   programmatically confirmed to be an exact, verbatim substring of the
   source text — the core anti-hallucination guarantee the schema was
   designed around actually holds under live model output.

## ⚠️ Two independent eval-pair sets currently exist — not yet reconciled

Phase 1 and Phase 2 each independently selected known-similar/
known-dissimilar patent pairs from the final corpus, without
coordinating:

- **Phase 1's set** (3 pairs, `phase1_findings.md` Section 6/7):
  selected by browsing the corpus directly; the doc itself flags this
  as a known limitation — pairs are corpus-derived, not independently
  sourced.
- **Phase 2's set** (6 pairs — S1–S3 similar, D1–D3 dissimilar,
  octree/point-cloud compression and MIP-map compression topics):
  each candidate's full abstract and claims text was read before
  finalizing (not just titles — one pair's misleadingly generic title
  was caught and verified against real text before inclusion), and
  every pair's `claims_status` was confirmed as `found` before locking.

Both sets share the same underlying limitation Phase 1's doc names
honestly: presence in the corpus is guaranteed by construction, not
independent sourcing. Neither set is a substitute for independently-
sourced pairs.

**This needs a decision, not a silent merge:** which set is canonical
for future retrieval-quality claims, or are both kept as a triangulating
pair of checks? Flagged for direct discussion between contributors
before either set is treated as final.

## Open items across both phases

- **Missing-abstract fallback** (Phase 1 §8, Phase 2 handoff above):
  title-only embedding for the 201 abstract-missing patents, not yet
  implemented — currently full exclusion.
- **No-match / low-confidence detection**: designed (AND-gate on
  absolute score + top-2 margin) but calibrated only against a single
  toy-data case. Needs real score distributions from the full real
  corpus before it's trustworthy.
- **Eval-pair reconciliation** (above).
- **2011–2014 date-range gap**: claims bulk files for these years were
  unavailable; corpus starts at 2015. Revisit if earlier coverage
  becomes necessary.
- **Batching/rate-limiting**: request pacing (`time.sleep` between
  calls) is in place and functional for moderate volume, but real
  batching (`batchEmbedContents`) has not been implemented. Needed
  before running the full 6,779-patent corpus in one pass, given the
  free-tier 100 RPM / 1,000 RPD ceiling.

[phase1_findings.md]: ./phase1_findings.md
[phase2_findings.md]: ./phase2_findings.md
