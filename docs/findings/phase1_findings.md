# Phase 1 Findings - Data Pipeline

**Patent Similarity Search - Corpus Construction**
Owner: Ayush Shirke | Status: Complete | Last updated: this session

---

## 1. Summary

This document covers the construction of the patent corpus used for retrieval and embedding in later phases: data source selection, pipeline design, data quality issues discovered and resolved, and the final evaluation dataset. The corpus is a stratified sample of **6,980 US patents** classified under CPC subclass **G06T** (image data processing / computer vision), filed between 2015 and 2025, stored in SQLite with title, abstract, independent claims text, and CPC subgroup metadata.

Four significant issues were found and resolved during construction, documented in full in Section 5. Each is presented with root cause, evidence gathered, and resolution - not just the fix, since the investigation process is as relevant to the project's data-quality posture as the final numbers.

---

## 2. Locked specification (Phase 0)

| Parameter | Value |
|---|---|
| Input format | Free text (user-submitted invention description) |
| Similarity target | Dual-vector - abstract embedding (30% weight) + independent-claims embedding (70% weight), fused via weighted average |
| Corpus scope | CPC subclass G06T, filed 2015-2025 |
| Success criteria | Top-5 ranked results with synthesized explanation, validated against known-similar/known-dissimilar patent pairs plus manual inspection |

---

## 3. Data source

**Attempted first:** PatentsView API (`search.patentsview.org`). Blocked - USPTO's migration to the Open Data Portal now requires ID.me identity verification to obtain an API key, which introduced a verification-queue dependency incompatible with the project timeline.

**Used instead:** USPTO Bulk Data Directory (`data.uspto.gov/bulkdata`), which offers direct TSV downloads with no authentication wall. Products used:

| Product | File | Fields used |
|---|---|---|
| PVGPATDIS (bibliographic) | `g_patent.tsv` | patent_id, patent_date, patent_title |
| PVGPATDIS (bibliographic) | `g_patent_abstract.tsv` | patent_id, patent_abstract |
| PVGPATDIS (bibliographic) | `g_cpc_current.tsv` | patent_id, cpc_subclass, cpc_group |
| PVGPATTXT (long text) | `g_claims_2015.tsv` … `g_claims_2025.tsv` (11 files) | patent_id, claim_text, dependent |

---

## 4. Pipeline architecture

`src/phase1_data_pipeline/build_corpus.py`, executed as a single sequential script:

1. **Load & filter patents** - `g_patent.tsv`, restricted to filing dates 2015-2025.
2. **Load & filter CPC classifications** - `g_cpc_current.tsv`, restricted to `cpc_subclass == "G06T"`, deduplicated to one row per patent (first-occurrence `cpc_group` kept as each patent's representative subgroup).
3. **Join** patents ↔ CPC (inner join - only classified G06T patents survive).
4. **Join** in abstract text (left join - patents without an abstract are retained, flagged).
5. **Stratified sampling** by `(year, cpc_group)` - see Section 5.3 for why this axis was added and its effect on corpus size.
6. **Claims extraction** - for each of the 11 yearly claims files, filter to patents already selected in the corpus, retain independent claims only (`dependent` column null), concatenate multiple independent claims per patent into one field.
7. **Status tracking** - every patent carries a `claims_status` (`found`/`missing`) flag rather than silently dropping incomplete records.
8. **Persist** to SQLite (`patents.db`), `patent_id` as primary key.

**Final schema:**
```
patent_id (PK), title, abstract, cpc_subclass, cpc_group,
claims_text, claims_status, filing_date
```

---

## 5. Issues found and resolved

### 5.1 PatentsView API inaccessible (ID.me verification wall)

**Symptom:** API key request page redirected to ID.me identity verification, introduced as part of USPTO's migration to the Open Data Portal (legacy PatentsView developer hub decommissioned).

**Investigation:** Confirmed via USPTO's own transition-guide documentation that the legacy API was fully retired and ODP access now requires a verified USPTO.gov account with ID.me linkage - a multi-step process with unpredictable turnaround.

**Resolution:** Pivoted to the Bulk Data Directory, which provides the same underlying data (and, as it turned out, a cleaner join structure - bibliographic and long-text data as separate flat files rather than paginated API responses) with no authentication barrier.

**Impact:** No data loss; access method changed, not data availability.

---

### 5.2 Independent vs. dependent claim misclassification risk

**Symptom:** The `dependent` column's semantics were not self-evident from the column name or a first look at its values (a mix of null and strings like `"claim 1"`).

**Investigation:** Rather than assume a polarity, sample claim text was pulled for both cases:
- `dependent = null` → text began *"1. A computer system comprising..."* - a standalone claim.
- `dependent = "claim 1"` → text began *"2. The computer system of claim 1, wherein..."* - explicitly referencing a parent claim.

**Resolution:** Confirmed `dependent.isna()` correctly identifies independent claims. This filter is what feeds the `claims_text` field used for the 70%-weighted claims embedding - an incorrect polarity here would have silently fed dependent (narrower, less representative) claims into every patent's similarity vector.

**Impact:** Verified correct before use; no rework required, but this was treated as a hard gate - the pipeline was not run against real claims data until this was confirmed with evidence, not assumption.

---

### 5.3 Claims completeness gap, concentrated in 2015-2021

**Symptom:** An early full-corpus run showed ~43% of patents missing claims text overall - high enough to investigate rather than accept.

**Investigation (hypothesis elimination, in order):**
1. *Are claims files missing entirely for some years?* - No; all 11 yearly files (2015-2025) confirmed present on disk.
2. *Is there a `patent_id` dtype mismatch between files causing silent join failures?* - No; confirmed both `g_patent.tsv` and yearly claims files use `int64` for `patent_id`.
3. *Is the independent/dependent filter excluding valid rows due to a formatting quirk in older files (e.g. empty string vs. true null)?* - No; a specific "missing" patent was checked and had zero rows in its expected year's claims file at all - not a filter miss.
4. *Is the patent's claims data misfiled under a different year than its patent_date implies?* - No; the patent was searched across all 11 years' claims files and found in none.

**Conclusion:** A genuine gap in USPTO's bulk long-text dataset for older grants, not a pipeline defect. Missingness by year (sample check): 2015-2021 ranged from ~50% to ~92% missing (worst in 2020-2021); 2022-2025 was ~98-100% complete.

**Resolution:** Corpus sampling weights recent years (2022-2025) more heavily than older years, since claims data is reliably present there. `claims_status` flag preserves the distinction rather than silently dropping incomplete records - final corpus: **4,479/6,980 (64.2%) claims found**, with the gap fully explained rather than papered over.

---

### 5.4 Missing abstracts

**Symptom:** 201/6,980 patents (2.88%) had no abstract text.

**Investigation:** Checked whether "missing" meant a mix of true `NULL`, empty string, and whitespace-only values (which would suggest a parsing bug) or a clean, consistent absence.

**Resolution:** Confirmed all 201 cases were clean `NULL` - no empty strings, no whitespace-only values. No data quality ambiguity; these patents genuinely lack an abstract in the source data. Retained in the corpus rather than excluded; Phase 2 embedding should fall back to title-only embedding for these specific records.

---

### 5.5 CPC subgroup stratification gap (found during peer review, most significant fix)

**Symptom:** Corpus review by a project collaborator raised a design concern: the sampling strategy's stated justification ("broad, representative coverage of the CPC subgroup space") depended on stratifying by CPC subgroup, but the actual sampling code only grouped by `year`. If true, the corpus could be dominated by whichever G06T subgroups happened to be most common in the raw data, with representativeness across subtypes (e.g. object tracking vs. image enhancement vs. 3D rendering) never actually verified.

**Investigation:**
- Confirmed the concern was valid, not hypothetical: `g_cpc_current.tsv` does contain a `cpc_group` column (real values, e.g. `G06T7/00`) - subgroup-level classification genuinely exists in the source data.
- Traced the root cause precisely: `pd.read_csv(..., usecols=["patent_id", "cpc_subclass"])` excluded `cpc_group` at the very first read of the file - it was never dropped downstream, it never entered the pipeline at all.
- Classified this as a fixable pipeline defect, not a data availability limitation.

**Resolution - first attempt:** Modified the pipeline to read `cpc_group` and stratify sampling by `(year, cpc_group)` via `groupby(["year", "cpc_group"]).apply(...)`. Hit a pandas 3.x behavior change where multi-column `groupby().apply()` silently drops the grouping columns from the result, causing a `KeyError` downstream. Diagnosed via traceback inspection and confirmed by testing column presence post-groupby directly.

**Resolution - final:** Rewrote the stratification step to build a single combined string key (`year + "|" + cpc_group`) and group by that one column instead of two - sidesteps the pandas multi-column-drop behavior entirely, since single-column groupby reliably preserves its key. Verified working end-to-end with no crash.

**Consequence - corpus size dropped from a 10,000-patent target to 6,980.** This was investigated as a real design tradeoff, not treated as a bug to explain away:
- **369 distinct G06T subgroups** exist in the filtered date range.
- With even per-cell sampling caps applied across all 369 subgroups × 11 years (4,059 possible cells), many rare (subgroup, year) combinations simply do not contain enough patents to fill their target - the sum of achievable cells falls short of 10,000.
- This is a direct, checkable mathematical consequence of strict even stratification against real-world subgroup sparsity - not a data shortage. Total available G06T patents in the date range is 134,850; raw supply was never the constraint.
- The original 10,000 figure was a safety ceiling derived from Gemini embedding quota and FAISS query latency headroom, not a hard requirement - prior scaling discussion indicated even 100K patents would likely be workable pending benchmarking. 6,980 sits comfortably under that ceiling.

**Decision:** Accepted 6,980 as final rather than loosening per-cell caps to chase 10,000, since doing so would let common subgroups absorb the extra headroom while rare subgroups stayed thin - reintroducing the exact imbalance this fix was meant to resolve. A properly-stratified smaller corpus was judged preferable to a larger, unevenly-stratified one.

---

## 6. Evaluation pairs

**Update (post-initial-review):** the original pairs (documented in earlier commits) were selected by browsing patents already inside the built corpus -- presence was guaranteed by construction, not independent evidence of retrieval quality. Replaced with pairs selected using a concept-first process: candidate technique pairs were named before querying the database, then matched against real corpus entries by keyword search, with the specific patent chosen by reading titles/abstracts rather than accepting the first match.

**Final pairs, concept-first selection:**

| Pair | Patents | Relationship |
|---|---|---|
| Similar A | `8983133` (multi-view object detection via appearance model transfer) vs. `9014421` (drift-corrected planar tracking, Lucas-Kanade optical flow) | Object detection and object tracking -- tracking commonly builds on detection output |
| Similar B | `9070197` (object-based segmentation) vs. `9025861` (floorplan reconstruction and 3D modeling) | Image segmentation and surface/structure reconstruction -- segmentation frequently feeds reconstruction pipelines |
| Dissimilar | `9105091` (watermark detection via propagation map) vs. `11714487` (gaze/foveal adjustment tracking) | Unrelated CV subdomains -- embedded signal recovery vs. eye-movement tracking, no meaningful technical overlap |

Verified present in the 6,980-patent corpus: 6/6.

**Remaining limitation:** while concept selection preceded the database query (an improvement over the original corpus-first method), the specific patent chosen to represent each concept was still picked from corpus search results, not sourced from an external reference set. Fully independent validation would use patent pairs whose relationship is established outside this project (e.g. citation relationships, examiner-cited prior art, or existing patent-similarity benchmarks) rather than pairs selected via keyword search against the same corpus being evaluated.

---

## 7. Final corpus statistics

| Metric | Value |
|---|---|
| Total patents | 6,980 |
| CPC scope | G06T, stratified across 369 distinct subgroups |
| Date range | 2015-01 to 2025-12 |
| Claims found | 4,479 (64.2%) |
| Claims missing | 2,501 (35.8%) |
| Abstract missing | 201 (2.88%) |
| Claims text length (found, chars) | mean 3,651 · median 3,203 · p25 2,203 · p75 4,520 · max 33,344 |

---

## 8. Open items for downstream phases

- **Missing-abstract fallback:** Phase 2 embedding should implement title-only embedding for the 201 patents lacking abstract text, rather than excluding them from retrieval entirely.
- **Eval pair strength:** current pairs are corpus-derived; consider independently-sourced pairs before final retrieval-quality claims.
- **2011-2014 gap:** claims bulk files for these years repeatedly failed to download; corpus date range starts at 2015 as a result. Revisit if earlier coverage becomes necessary.