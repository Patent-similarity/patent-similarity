# Phase 2 Findings — Two-Stage Retrieval Validation

**Status:** Validated on toy data and full real corpus; Stage 1 and Stage 2 retrieval behavior validated against 16 real evaluation cases; no-match calibration investigated; Phase 2 ready for freeze/sign-off.

**Deliverable:** `src/phase2_embedding_retrieval/embedding_pipeline.py`, `data/real_corpus.py`, `test/phase2-test.ipynb`, `test/test_real_smoke.py`, `test/test_eval_pairs_real.py`, `test/evaluate_stage2.py`

---

## What this validates

Phase 2 owns the retrieval design: Stage 1 embeds the query and every patent's **abstract only** (Gemini `gemini-embedding-001`), searches with FAISS `IndexFlatIP` (cosine via normalized inner product), and takes the top-K candidates forward.

Stage 2 embeds **claims** for those candidates only and reranks with:

`final_score = 0.3 * abstract_score + 0.7 * claim_score`

The production-scale retrieval corpus contains 6,980 patent records, of which 6,779 have usable abstracts and are represented in the FAISS index.

The final Stage 2 evaluation uses the corrected `stage2_chunked_claims_v2` pipeline, which safely handles oversized claims by splitting them into token-safe chunks before embedding and aggregating each patent's claim similarity using the maximum chunk-level similarity.

---

# What was validated before full real-data evaluation

## A bug found and fixed early

The first version of `patent_to_text` concatenated title + abstract + claims for Stage 1. That's wrong — Stage 1 is abstract-only, with claims held back for Stage 2.

The issue was caught by inspecting the embedding input directly and was fixed before real evaluation. The corrected function has an inline comment explaining why it is abstract-only, preventing silent regression.

---

## Toy-data evaluation

An 8-patent toy corpus was used to validate the retrieval mechanics before full real-data testing.

The evaluation contained 8 queries with expected results locked before execution:

* 5 clean single-match queries
* 1 deliberately ambiguous query
* 1 no-match query
* 1 near-duplicate phrasing check

### Toy-data results

7 of 7 in-domain queries retrieved the correct patent at rank 1 at both stages. Claims reranking never flipped a correct Stage 1 result into an incorrect one.

Example (Q8, phrased close to P001's own claim text):

* Stage 1 top score: `0.8060`
* Stage 2 top score: `0.9240`
* Correct patent remained rank 1

### Ambiguity stress test

Q6 deliberately overlapped P003 and P005.

* Stage 1 margin: `0.0644`
* Stage 2 margin: `0.1307`

Claims reranking more than doubled the separation between the intended candidate and the competing candidate, demonstrating that claims can provide meaningful disambiguating evidence rather than simply adding noise.

### No-match toy query

Q7 produced a top score of `0.6631`, below the genuine-match scores observed in the toy corpus.

This established the initial no-match hypothesis, but it was explicitly treated as uncalibrated until real-corpus testing.

---

# Missing-claims exclusion — validated

Patents with `claims: None` are excluded from Stage 2 entirely. They remain eligible for Stage 1 but are never embedded, scored, or reranked at Stage 2.

`select_full_treatment(stage2_indices, max_results=5)` selects up to 5 Stage-2-eligible candidates, with **no backfilling** from Stage-1-only candidates if fewer than 5 are eligible.

The expanded toy corpus and regression tests confirmed:

* claims-missing patents remain excluded from Stage 2
* remaining eligible patents are reranked normally
* the maximum Stage 2 treatment count is respected
* fewer than 5 eligible candidates are not artificially padded

The real corpus contains approximately 35.8% claims-missing records, making this exclusion behavior operationally important.

---

# Stage 2 restructured for real-data efficiency

The original implementation precomputed claim embeddings for every patent during `build_indexes`. That approach was acceptable for a small toy corpus but wasteful at real scale.

The implementation was restructured so that claims are embedded only for the current query's Stage 1 survivors, after excluding claims-missing candidates.

This means Stage 2 processing is query-scoped rather than corpus-wide.

The restructuring was validated through toy regression and real-data smoke testing.

---

# Real-data loader

`data/real_corpus.py` reads `patents.db`, excludes rows with missing abstracts, and preserves `claims: None` for patents whose claims are unavailable.

A loader bug was found when the implementation attempted to filter using an `abstract_status` column that did not exist in the rebuilt Phase 1 schema.

The loader was corrected to filter directly on:

`abstract IS NOT NULL AND TRIM(abstract) != ''`

Verification:

* Total database rows: **6,980**
* Patents with usable abstracts: **6,779**
* Missing abstracts: **201**

These counts match the documented Phase 1 corpus.

---

# Rate limiting and embedding constraints

Initial real-data runs encountered Gemini `429 RESOURCE_EXHAUSTED` errors.

The observed free-tier limits included request-per-minute and daily request constraints. The implementation therefore added pacing and token-aware batching.

The current implementation:

* tracks request usage
* tracks estimated token usage
* waits when the token-per-minute budget is reached
* splits embedding work into token-safe batches
* retries transient rate-limit failures
* safely resumes completed evaluation cases

A later full evaluation run demonstrated the daily free-tier request quota could be exhausted during Stage 2 processing. This was an API quota limitation rather than a retrieval or implementation defect.

The final evaluation was subsequently completed using the corrected chunked-claims pipeline.

---

# Full-corpus evaluation

The final evaluation uses the complete **6,779-patent FAISS corpus** and 16 evaluation cases:

* 3 verified genuine-positive pairs: S1, S2, S3
* 3 designated-wrong candidate tests: D1, D2, D3
* 10 deliberately constructed no-match queries: NM1–NM10

The existing FAISS index was reused. The corpus was not rebuilt or re-embedded for the no-match experiments.

The final Stage 2 evaluation uses pipeline version:

`stage2_chunked_claims_v2`

---

# Findings from the Full 6,779-Patent Evaluation

## 1. The retrieval pipeline is deterministic

Queries using the same query patent produce identical retrieval results.

For example:

* S2 and D1 both use query `11676310`
* S1 and D2 both use query `11615557`

Their retrieval results are identical when the query is identical.

**Finding:** The FAISS retrieval pipeline is deterministic and stable rather than producing flaky or inconsistent rankings.

---

## 2. Genuine similar pairs retrieve highly in the full corpus

The three verified genuine-positive pairs produced:

| Pair | Partner Stage 1 score | Full-corpus rank |
| ---- | --------------------: | ---------------: |
| S1   |              0.847694 |       #3 / 6,779 |
| S2   |              0.790738 |      #12 / 6,779 |
| S3   |              0.869376 |       #2 / 6,779 |

All three genuine partners were therefore retrieved within approximately the top 0.18% of the full corpus.

**Finding:** Stage 1 successfully retrieves all three known genuine-positive partners from the complete 6,779-patent corpus.

---

## 3. S2 demonstrates a real top-5 recall limitation

S2's genuine partner is:

* Query: `11676310`
* Partner: `11790567`
* Stage 1 score: `0.790738`
* Stage 1 rank: **#12 / 6,779**
* Stage 2 rank: **#11**
* Stage 2 final score: `0.784457`

The current full-treatment/synthesis cutoff is `max_results=5`.

Therefore, a genuine partner can be successfully retrieved by Stage 1 and retained within the Stage 2 candidate pool while still being excluded from the final top-5 full-treatment set.

**Finding:** The current top-5 synthesis cutoff does not guarantee recall of all genuine matches.

The existing `top_k=50` should therefore be retained. No evidence currently justifies reducing the retrieval depth.

---

## 4. Stage 2 preserves or improves the known-positive rankings

The corrected `stage2_chunked_claims_v2` evaluation produced:

| Pair | Stage 1 rank | Stage 2 rank | Stage 2 final score |
| ---- | -----------: | -----------: | ------------------: |
| S1   |           #3 |           #3 |            0.850839 |
| S2   |          #12 |          #11 |            0.784457 |
| S3   |           #2 |           #2 |            0.827281 |

All three genuine-positive partners remained successfully retrieved after claim-level reranking.

S2 improved from rank #12 to #11.

S1 and S3 retained their original positions.

**Finding:** The corrected Stage 2 claim-processing implementation does not damage the known-positive retrieval results and can improve candidate ordering.

---

## 5. D1, D2, and D3 designated-wrong targets remain outside the Stage 1 candidate set

Under the final `stage2_chunked_claims_v2` evaluation:

* D1 target: not in top 50
* D2 target: not in top 50
* D3 target: not in top 50

The previously observed full-corpus scores for the designated wrong candidates were also far below the genuine-positive region.

**Finding:** None of the three designated-wrong targets is retrieved into the Stage 1 top-50 candidate pool.

This supports the ability of the embedding layer to reject these specific wrong candidates.

---

## 6. D1/D2/D3 are designated-wrong tests, not genuine no-match ground truth

D1/D2/D3 test:

> "Does this particular wrong patent score low against this query?"

They do not test:

> "Does this query have no appropriate match anywhere in the corpus?"

Therefore, these cases should not be interpreted as a complete no-match evaluation.

The separate NM1–NM10 experiment provides the actual no-match/hard-negative evidence.

---

# Findings from the Corrected Stage 2 Evaluation

## 7. Stage 2 claim chunking successfully handles oversized claims

The corrected implementation detects claims whose estimated token count is too large for a single embedding request.

Such claims are split into multiple token-safe chunks before embedding.

For each original patent, the claim score is then aggregated using the maximum similarity across its claim chunks.

The final score remains:

`final_score = 0.3 * abstract_score + 0.7 * claim_score`

The full evaluation successfully processed oversized claims without the previous failure mode.

**Finding:** The chunked-claims implementation resolves the oversized-claim constraint without changing the overall two-stage scoring architecture.

---

## 8. Stage 2 claim evidence can alter rankings without destroying Stage 1 retrieval

For the three known positives:

### S1

* Stage 1 rank: #3
* Stage 2 rank: #3
* Partner claim score: `0.852187`
* Partner final score: `0.850839`

### S2

* Stage 1 rank: #12
* Stage 2 rank: #11
* Partner claim score: `0.781765`
* Partner final score: `0.784457`

### S3

* Stage 1 rank: #2
* Stage 2 rank: #2
* Partner claim score: `0.809240`
* Partner final score: `0.827281`

**Finding:** Claims-based reranking provides additional evidence while preserving all three known-positive retrievals. It should therefore remain a second-stage reranking mechanism rather than replacing Stage 1.

---

# Findings from the Attractor Investigation

## 9. Patent 12477156 is a strong semantic attractor

Patent `12477156`, **"Method for encoding and decoding, encoder, and decoder,"** has a highly specific point-cloud/octree compression focus.

Its abstract and claims contain concepts including:

* point-cloud compression
* octree geometry
* parent/child nodes
* occupancy patterns
* neighboring nodes
* coding context
* entropy encoding/decoding

It appears highly for multiple related queries.

For example:

| Query           | 12477156 score |
| --------------- | -------------: |
| S1 / `11615557` |       0.868016 |
| S2 / `11676310` |       0.839566 |
| D1 / `11676310` |       0.839566 |

For S1, it even outranks the hand-verified partner.

**Finding:** A high Stage 1 score can represent membership in a close technical neighborhood without necessarily identifying the specific patent considered the correct match by the evaluation ground truth.

---

## 10. Stage 1 measures semantic relatedness rather than final patent equivalence

D1 makes the attractor behavior particularly clear.

For query `11676310`:

* Designated wrong partner: approximately `0.576`
* Patent `12477156`: `0.839566`

The embedding is therefore identifying a technically meaningful alternative patent rather than simply producing an arbitrary false result.

**Finding:** Stage 1 should be interpreted as a **semantic candidate-retrieval layer**, not as a final patent-equivalence classifier.

This supports retaining Stage 2 claim-level evaluation and downstream synthesis.

---

# Findings from the Final No-Match Evaluation

The final no-match results were evaluated using the corrected `stage2_chunked_claims_v2` pipeline.

Unlike the earlier Stage 1-only no-match analysis, these results include the Stage 2 claim score and the final weighted score.

## 11. Stage 2 no-match scores can remain relatively high

Final Stage 2 scores:

| Query | Abstract |   Claims |        Final |
| ----- | -------: | -------: | -----------: |
| NM1   | 0.613017 | 0.621864 |     0.619210 |
| NM2   | 0.690602 | 0.701243 |     0.698051 |
| NM3   | 0.721483 | 0.721097 |     0.721213 |
| NM4   | 0.739242 | 0.727066 |     0.730719 |
| NM5   | 0.771276 | 0.783103 | **0.779555** |
| NM6   | 0.722833 | 0.730890 |     0.728473 |
| NM7   | 0.741519 | 0.741284 |     0.741355 |
| NM8   | 0.731818 | 0.725550 |     0.727430 |
| NM9   | 0.735034 | 0.747125 |     0.743497 |
| NM10  | 0.752067 | 0.738363 |     0.742474 |

The highest final no-match score is:

**NM5 = 0.779555**

The lowest genuine-positive Stage 2 final score is:

**S2 = 0.784457**

The observed gap is therefore only approximately:

**0.004902**

**Finding:** Stage 2 claim-aware scoring does not create a clean absolute-score separation between genuine positives and deliberately constructed no-match queries.

This does not indicate a broken retrieval system. It demonstrates that semantic similarity can remain high for technically related but intentionally non-matching inventions.

---

## 12. NM5 demonstrates that a high Stage 2 score does not establish a match

NM5 produced:

* Abstract score: `0.771276`
* Claim score: `0.783103`
* Final score: `0.779555`

The retrieved patent concerns vehicle brake-wear monitoring and prediction.

The query was intentionally constructed as a no-match case.

The relatively high claim score demonstrates that claim-level semantic similarity can still occur between technically related concepts without establishing that the retrieved patent is the intended invention.

**Finding:** Stage 2 improves evidence quality but does not turn cosine similarity into a calibrated probability of patent-level equivalence.

---

## 13. NM10 remains a useful hard-negative example

NM10 produced:

* Abstract score: `0.752067`
* Claim score: `0.738363`
* Final score: `0.742474`

The retrieved patent concerns contactless assessment of hemodynamic parameters and vital signs.

The query is intentionally treated as a no-match case despite the strong semantic relationship.

**Finding:** A score in the mid-0.70s can correspond to substantial technical/domain overlap without representing an appropriate patent match.

---

## 14. No-match behavior is heterogeneous

The final no-match set demonstrates several different patterns:

### Type A — Weak semantic neighborhood

Examples:

* NM1
* NM2
* NM6
* NM9

These generally have moderate scores and relatively weak candidate separation.

### Type B — Strong domain-related candidate

Examples:

* NM5
* NM7
* NM8

These produce meaningful semantic similarity despite being labeled no-match.

### Type C — Strong semantic overlap but wrong invention

Examples:

* NM3
* NM10

These demonstrate that even relatively strong semantic relationships can correspond to inventions that are materially different.

**Finding:** No-match queries do not have a single characteristic score or ranking pattern. The retrieval system is correctly behaving as a semantic search system, while the application-level definition of "match" is stricter.

---

# Updated Threshold Findings

## 15. A single cosine threshold should not be used as the final patent-match decision

The corrected Stage 2 evaluation shows:

* Lowest genuine-positive final score: **0.784457**
* Highest no-match final score: **0.779555**
* Observed gap: approximately **0.004902**

This overlap is too small to justify a production binary threshold from the current evaluation.

A threshold around `0.78` would already be extremely sensitive to small changes in the evaluation set.

**Finding:** The current evidence does not support using Stage 2 final cosine similarity as a standalone match/no-match classifier.

---

## 16. Margin should not be used as a hard rejection rule

The earlier candidate-specific margin analysis remains relevant.

For genuine positives:

| Positive | Partner score | Next candidate | Candidate margin |
| -------- | ------------: | -------------: | ---------------: |
| S1       |      0.847694 |       0.834315 |         0.013378 |
| S2       |      0.790738 |       0.788676 |         0.002061 |
| S3       |      0.869376 |       0.783486 |         0.085891 |

S2 demonstrates that a genuine partner can have a very small candidate-specific margin.

At the same time, no-match cases can produce substantial margins.

For example, NM5 previously produced a Stage 1 margin of `0.055323` while still being deliberately labeled no-match.

**Finding:** Margin is useful as a diagnostic or supporting feature but should not be treated as a hard match/rejection gate.

---

## 17. The previously observed 0.77 Stage 1 threshold should remain provisional

The earlier Stage 1-only analysis found that a threshold of `0.77` retained all 3 known positives while producing zero false positives among the 10 no-match queries.

However, that threshold was based on Stage 1 scores only.

The corrected Stage 2 results demonstrate that final claim-aware scores overlap more closely with the no-match distribution.

Therefore, `0.77` should **not** be described as a production match threshold.

It may remain a provisional **Stage 1 retrieval/triage threshold** if needed, but the current architecture should not use it as a definitive patent-match decision.

**Finding:** No production-grade cosine threshold is selected at Phase 2 freeze.

---

# Architectural Takeaway

The full evaluation strengthens the original two-stage architecture rather than invalidating it.

The evidence supports:

**Stage 1 — Broad semantic retrieval**

Use abstract embeddings and FAISS to identify technically related candidates.

↓

**Stage 2 — Claim-aware reranking**

Use candidate claims to add invention-specific evidence and improve candidate ordering.

↓

**Phase 3 — LLM synthesis**

Use the retrieved candidates and claim/technical evidence to produce the application's final relevance assessment.

The evaluation demonstrates why this layered architecture is preferable to treating a single embedding score as the final answer.

---

# Final Phase 2 Conclusions

The final evaluation establishes the following:

1. The 6,779-patent FAISS retrieval pipeline is deterministic and stable.
2. All 3 verified genuine-positive partners were retrieved from the full corpus.
3. The genuine partners ranked #3, #12, and #2 respectively.
4. Stage 2 preserved the correct positive retrievals and improved S2 from #12 to #11.
5. All 3 designated-wrong targets remained outside the Stage 1 top-50 candidate set.
6. The corrected chunked-claims implementation successfully handles oversized claims.
7. The strongest known positive has a relatively small candidate-specific margin, so margin cannot be a hard gate.
8. Multiple patents can act as strong semantic attractors because they belong to the same technical neighborhood.
9. No-match queries can produce relatively high Stage 2 final scores.
10. The highest observed no-match final score (`0.779555`) is close to the lowest genuine-positive final score (`0.784457`).
11. Therefore, cosine similarity should be interpreted as a retrieval/ranking signal, not as a calibrated probability of patent-level similarity.
12. No production-grade binary similarity threshold is claimed from the current 16-case evaluation.
13. `top_k=50` should be retained because S2's genuine partner occurs at rank #12.
14. The current top-5 synthesis cutoff is not a guaranteed-recall cutoff.
15. The two-stage retrieval architecture is supported by the evaluation and should remain unchanged for the current project.

---

# Phase 2 Freeze Decision

The current evidence is sufficient to freeze the Phase 2 retrieval implementation.

No further corpus rebuild, FAISS rebuild, or embedding generation is required for Phase 2 sign-off.

The remaining Phase 2 action is documentation and handoff of the validated behavior and limitations.

The system should be handed forward with the following interpretation:

> **Phase 2 provides deterministic semantic candidate retrieval and claim-aware reranking over the 6,779-patent corpus. Similarity scores identify technically related patents but are not calibrated patent-match probabilities. Final patent-level relevance should therefore be determined using the downstream claim/evidence synthesis layer rather than a single cosine threshold.**

The verified FAISS and metadata artifacts have already been handed off for integration.

**Phase 2 status: COMPLETE / FROZEN.**
