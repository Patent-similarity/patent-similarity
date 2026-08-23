
---

## 6. Issues found and resolved

### 6.1 PatentsView API inaccessible (ID.me verification wall)

**Symptom:** API key request page redirected to ID.me identity verification, introduced as part of USPTO's migration to the Open Data Portal (legacy PatentsView developer hub decommissioned).

**Investigation:** Confirmed via USPTO's own transition-guide documentation that the legacy API was fully retired and ODP access now requires a verified USPTO.gov account with ID.me linkage - a multi-step process with unpredictable turnaround.

**Resolution:** Pivoted to the Bulk Data Directory, which provides the same underlying data (and, as it turned out, a cleaner join structure - bibliographic and long-text data as separate flat files rather than paginated API responses) with no authentication barrier.

**Impact:** No data loss; access method changed, not data availability.

---

### 6.2 Independent vs. dependent claim misclassification risk

**Symptom:** The `dependent` column's semantics were not self-evident from the column name or a first look at its values (a mix of null and strings like `"claim 1"`).

**Investigation:** Rather than assume a polarity, sample claim text was pulled for both cases:
- `dependent = null` -> text began *"1. A computer system comprising..."* - a standalone claim.
- `dependent = "claim 1"` -> text began *"2. The computer system of claim 1, wherein..."* - explicitly referencing a parent claim.

**Resolution:** Confirmed `dependent.isna()` correctly identifies independent claims. This filter is what feeds the `claims_text` field used for the 70%-weighted claims embedding - an incorrect polarity here would have silently fed dependent (narrower, less representative) claims into every patent's similarity vector.

**Impact:** Verified correct before use; no rework required, but this was treated as a hard gate - the pipeline was not run against real claims data until this was confirmed with evidence, not assumption.

---

### 6.3 Claims completeness gap, concentrated in 2015-2021

**Symptom:** An early full-corpus run showed ~43% of patents missing claims text overall - high enough to investigate rather than accept.

**Investigation (hypothesis elimination, in order):**
1. *Are claims files missing entirely for some years?* - No; all 11 yearly files (2015-2025) confirmed present on disk.
2. *Is there a `patent_id` dtype mismatch between files causing silent join failures?* - No; confirmed both `g_patent.tsv` and yearly claims files use consistent typing for `patent_id`.
3. *Is the independent/dependent filter excluding valid rows due to a formatting quirk in older files (e.g. empty string vs. true null)?* - No; a specific "missing" patent was checked and had zero rows in its expected year's claims file at all - not a filter miss.
4. *Is the patent's claims data misfiled under a different year than its patent_date implies?* - No; the patent was searched across all 11 years' claims files and found in none.

**Conclusion:** A genuine gap in USPTO's bulk long-text dataset for older grants, not a pipeline defect. Missingness by year (sample check): 2015-2021 ranged from ~50% to ~92% missing (worst in 2020-2021); 2022-2025 was ~98-100% complete.

**Resolution:** Corpus sampling weights recent years (2022-2025) more heavily than older years, since claims data is reliably present there. `claims_status` flag preserves the distinction rather than silently dropping incomplete records - final corpus: **4,479/6,980 (64.2%) claims found**, with the gap fully explained rather than papered over.

---

### 6.4 Missing abstracts

**Symptom:** 201/6,980 patents (2.88%) had no abstract text.

**Investigation:** Checked whether "missing" meant a mix of true `NULL`, empty string, and whitespace-only values (which would suggest a parsing bug) or a clean, consistent absence.

**Resolution:** Confirmed all 201 cases were clean `NULL` - no empty strings, no whitespace-only values. No data quality ambiguity; these patents genuinely lack an abstract in the source data. Retained in the corpus rather than excluded, with an `abstract_status` flag; Phase 2 embedding should fall back to title-only embedding for these specific records.

---

### 6.5 CPC subgroup stratification gap (found during peer review, most significant fix)

**Symptom:** Corpus review by a project collaborator raised a design concern: the sampling strategy's stated justification ("broad, representative coverage of the CPC subgroup space") depended on stratifying by CPC subgroup, but the actual sampling code only grouped by `year`. If true, the corpus could be dominated by whichever G06T subgroups happened to be most common in the raw data, with representativeness across subtypes (e.g. object tracking vs. image enhancement vs. 3D rendering) never actually verified.

**Investigation:**
- Confirmed the concern was valid, not hypothetical: `g_cpc_current.tsv` does contain a `cpc_group` column (real values, e.g. `G06T7/00`) - subgroup-level classification genuinely exists in the source data.
- Traced the root cause precisely: `pd.read_csv(..., usecols=["patent_id", "cpc_subclass"])` excluded `cpc_group` at the very first read of the file - it was never dropped downstream, it never entered the pipeline at all.
- Classified this as a fixable pipeline defect, not a data availability limitation.

**Resolution - first attempt:** Modified the pipeline to read `cpc_group` and stratify sampling by `(year, cpc_group)` via `groupby(["year", "cpc_group"]).apply(...)`. Hit a pandas 3.x behavior change where multi-column `groupby().apply()` drops the grouping columns from the result. Diagnosed via traceback inspection and confirmed by testing column presence post-groupby directly.

**Resolution - final:** Rewrote the stratification step to build a single combined string key (`year + "|" + cpc_group`) and group by that one column instead of two - sidesteps the pandas multi-column-drop behavior entirely, since single-column groupby reliably preserves its key. Verified working end-to-end with no crash.

**Consequence - corpus size dropped from a 10,000-patent target to 6,980.** This was investigated as a real design tradeoff, not treated as a bug to explain away:
- **369 distinct G06T subgroups** exist in the filtered date range.
- With even per-cell sampling caps applied across all 369 subgroups x 11 years (4,059 possible cells), many rare (subgroup, year) combinations simply do not contain enough patents to fill their target - the sum of achievable cells falls short of 10,000.
- This is a direct, checkable mathematical consequence of strict even stratification against real-world subgroup sparsity - not a data shortage. Total available G06T patents in the date range is 134,850; raw supply was never the constraint.
- The original 10,000 figure was a safety ceiling derived from Gemini embedding quota and FAISS query latency headroom, not a hard requirement. 6,980 sits comfortably under that ceiling.

**Decision:** Accepted 6,980 as final rather than loosening per-cell caps to chase 10,000, since doing so would let common subgroups absorb the extra headroom while rare subgroups stayed thin - reintroducing the exact imbalance this fix was meant to resolve. A properly-stratified smaller corpus was judged preferable to a larger, unevenly-stratified one. Optional future work: revisit raised per-cell caps (e.g. recent=6/older=3) to test whether corpus size can grow closer to 10,000 without reintroducing imbalance.

---

## 7. Evaluation pairs

Because the corpus was rebuilt twice after the initial version (once for the subgroup fix, once more for the pandas workaround), evaluation anchors were **deliberately re-selected from the final corpus each time**, rather than force-included from an earlier sample - force-including specific known IDs after the fact would reintroduce the exact cherry-picking bias the selection process is meant to avoid.

**Final pairs, selected from the completed 6,980-patent corpus:**

| Pair | Patents | Relationship |
|---|---|---|
| Similar 1 | `8928658` (photon mapping via kd-trees) vs. `8933933` (early Z-mode rendering pipeline) | Both GPU rendering pipeline optimization techniques |
| Similar 2 | `8934666` (surrounding object/scene analysis, class segmenting) vs. `8929621` (segmentation and surface matching) | Both scene/object segmentation approaches |
| Dissimilar | `8929589` (high-resolution gaze tracking) vs. `8930719` (data protection encoding/decoding) | Unrelated CV problem domains; second patent is only nominally G06T-classified |

Verified present in final corpus: 6/6.

**Known limitation (see Section 3, decision 2):** these pairs were selected by browsing patents already inside the built corpus, not chosen independently beforehand. Their presence is guaranteed by construction rather than evidence of retrieval quality. Useful as an initial smoke test; independently-sourced pairs should be used before final retrieval-quality claims are made.

---

## 8. Final corpus statistics

| Metric | Value |
|---|---|
| Total patents | 6,980 |
| CPC scope | G06T, stratified across 369 distinct subgroups |
| Date range | 2015-01 to 2025-12 |
| Claims found | 4,479 (64.2%) |
| Claims missing | 2,501 (35.8%) |
| Abstract found | 6,779 (97.12%) |
| Abstract missing | 201 (2.88%) |
| Claims text length (found, chars) | mean 3,651 - median 3,203 - p25 2,203 - p75 4,520 - max 33,344 |

---

## 9. Open items for downstream phases

- **Missing-abstract fallback:** Phase 2 embedding should implement title-only embedding for the 201 patents lacking abstract text, rather than excluding them from retrieval entirely.
- **Eval pair strength:** current pairs are corpus-derived (see Section 3, decision 2); independently-sourced pairs should be added before final retrieval-quality claims.
- **2011-2014 gap:** claims bulk files for these years repeatedly failed to download; corpus date range starts at 2015 as a result (see Section 3, decision 1). Revisit if earlier coverage becomes necessary.
- **Optional stretch:** test raised per-cell stratification caps to see if corpus size can approach 10,000 without reintroducing subgroup imbalance (see Section 6.5).