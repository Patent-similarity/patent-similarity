# Phase 2 Findings — Two-Stage Retrieval Validation

**Status:** Complete on toy data. Not yet run on the real G06T corpus.
**Deliverable:** `src/embedding_pipeline.py`, `test/phase2-test.ipynb`

## What this validates

Phase 2 owns the retrieval design: Stage 1 embeds the query and every
patent's **abstract only** (Gemini `gemini-embedding-001`), searches with
FAISS `IndexFlatIP` (cosine via normalized inner product), and takes the
top candidates forward. Stage 2 embeds **claims** for those candidates
and reranks with `final_score = 0.3 * abstract_score + 0.7 * claim_score`.

Before real data was available, I built a 5-patent toy corpus (CNN image
classifier, tumor detection, autonomous-vehicle object detection, NLP
summarization, facial recognition access control) to validate the
*mechanics* of the two-stage design — not to prove anything about real
patent recall. The goal was to catch design and implementation bugs early,
since the retrieval logic itself doesn't depend on which corpus it runs
against; only the data-loading step changes later.

## A bug I found and fixed

The first version of `patent_to_text` concatenated title + abstract +
claims for Stage 1. That's wrong — Stage 1 is supposed to be
abstract-only, with claims held back for Stage 2. I caught this by
inspecting the embedding input directly, not by noticing bad output, and
fixed it before running any evaluation. The corrected function is in
`embedding_pipeline.py` with a comment explaining why it's abstract-only,
so it doesn't regress silently later.

## Evaluation method

8 queries, with expected results **locked before running anything**:

- 5 clean single-match queries (one per toy patent)
- 1 deliberately ambiguous query (overlaps two patents by design)
- 1 no-match query (audio processing — no toy patent should match)
- 1 near-duplicate phrasing check of an existing patent's claim text

## Results

**7 of 7 in-domain queries retrieved the correct patent at rank 1**, at
both Stage 1 and Stage 2. Claims reranking never flipped a correct
Stage 1 result into an incorrect one anywhere in this test set.

Example (Q8, phrased close to P001's own claim text):

| Stage | Top score | Patent |
|---|---|---|
| Stage 1 (abstract) | 0.8060 | P001 (correct) |
| Stage 2 (reranked) | 0.9240 | P001 (correct, score increased) |

**Ambiguity stress test (Q6):** query deliberately overlaps P003
(autonomous-vehicle object detection) and P005 (facial recognition
access control) — both involve detecting/identifying entities from
camera input. Stage 1 alone gave a thin margin between them
(0.8039 vs 0.7395, a gap of 0.0644). Claims reranking widened that
gap to roughly 0.1307 — more than double — because the claims text
makes the vehicle-control application explicit in a way the abstract
alone doesn't. This is the concrete evidence that Stage 2 is doing
real disambiguating work, not just adding noise.

**No-match query (Q7):** all five toy patents still returned a
"top" result, since nearest-neighbor search always returns *something*.
But the absolute scores were flat and low relative to the in-domain
queries — top score 0.6631, versus 0.8060+ for a genuine match (Q8).
That gap is the signal a no-match detector should key off, but there's
no calibrated threshold yet — see limitations below.

## Limitations — read before reusing this on real data

- **No-match detection is designed, not calibrated.** The plan is an
  AND-gate (low absolute top score + low top-2 margin), conceptually
  checked against this single toy no-match case. One data point does
  not calibrate a threshold. This needs real score distributions from
  the actual G06T corpus before it's trustworthy.
- **This is 5 patents, not 10,000.** 7/7 correct top-1 on a toy corpus
  demonstrates the pipeline mechanics work as designed; it says nothing
  about retrieval quality or accuracy at real corpus scale, where
  confusable near-duplicates are far more likely.
- **Claims coverage gap not yet handled.** Phase 1's corpus has
  `claims_status: missing` for ~29% of patents. Every toy patent has
  complete claims, so this pipeline has never been exercised against a
  patent with no claims text. Stage 2 needs an explicit path for that
  case (skip rerank and fall back to abstract-only score, most likely)
  before this runs on the real corpus.

## Addendum — missing-claims exclusion validated

Before adapting the pipeline to the real corpus, the missing-claims
policy from the limitations section above was implemented and tested:
patents with `claims: None` are now excluded from Stage 2 entirely
(never embedded, never scored, never reranked) while remaining fully
eligible for the Stage 1 abstract-only ranking. A new
`select_full_treatment(stage2_indices, max_results=5)` function selects
up to 5 Stage-2-eligible candidates for full synthesis, with **no
backfilling** from Stage-1-only candidates if fewer than 5 are eligible.

**Toy corpus expanded from 5 to 8 patents** to test this: the original
5 plus 3 synthetic claims-missing patents (medical-imaging themed,
deliberately overlapping P002's abstract to create realistic
competition, not easy-to-spot noise).

**Regression check:** the original 8-query locked eval set was rerun
against the expanded corpus. All results held — Q1–Q5 still retrieve
their expected patent at rank 1 in both stages; Q6's ambiguity test
still shows the claims-rerank margin widening from 0.0644 to 0.1307
between P003 and P005; Q7's no-match scores remain flat and low
relative to genuine matches; Q8's reinforcement (0.8060 → 0.9240)
is unchanged. Adding claims-missing patents did not interfere with
any previously-validated behavior.

**Cap behavior verified at two eligibility counts:** with 5 eligible
candidates, `select_full_treatment` returns all 5. With one additional
patent's claims temporarily set to `None` (4 eligible), it returns
exactly 4 — confirming the no-backfill rule holds rather than silently
padding to 5 from Stage-1-only candidates.

**New observation — claims-missing patents can crowd out eligible ones
in Stage 1.** In several queries (Q2, Q3, Q7, Q8), the synthetic
claims-missing patents ranked *above* real, claims-eligible patents in
the abstract-only Stage 1 view. This wasn't designed to happen — it's
a byproduct of writing realistic missing-claims patents — but it's a
useful early signal: with the real corpus's ~29% claims-missing rate,
Stage 1's top-50 will likely include a non-trivial share of
claims-ineligible candidates, meaning the number of Stage-2-eligible
candidates per query could sometimes land well below 5. The no-backfill
policy above is designed for exactly this case, and this is toy-scale
evidence that it's a realistic scenario, not just a hypothetical edge
case.

## Next steps

- Adapt the data-loading step to read from the real `patents.db` schema,
  respecting `claims_status` / `abstract_status` instead of assuming
  every field is populated.
- Recalibrate the no-match AND-gate thresholds against real score
  distributions once the real corpus is embedded.
- Re-run the same 8-query-style evaluation (expanded with real eval
  pairs from Phase 1) against the real corpus, predictions locked
  before running, same as this round.
