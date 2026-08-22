# Phase 1 Findings — Data Pipeline

## Spec (locked)

- **Input format:** free text (user-submitted invention description)
- **Similarity target:** dual-vector — abstract embedding (30% weight) + independent-claims embedding (70% weight), fused via weighted average
- **Corpus scope:** CPC subclass **G06T** (image data processing / computer vision), filed **2015–2025**, target size **10,000 patents**
- **Success criteria:** top-5 ranked results with synthesized explanation, validated against known-similar/known-dissimilar patent pairs plus manual inspection

## Data source

USPTO Bulk Data Directory (`data.uspto.gov`) — the PatentsView API was attempted first but blocked by USPTO's ID.me identity verification requirement (introduced as part of PatentsView's migration to the Open Data Portal). Pivoted to direct bulk TSV downloads instead.

Files used, product `PVGPATDIS` (bibliographic) + `PVGPATTXT` (long text):
- `g_patent.tsv` — patent_id, patent_date, patent_title
- `g_patent_abstract.tsv` — patent_id, patent_abstract
- `g_cpc_current.tsv` — patent_id, cpc_subclass (filtered to G06T)
- `g_claims_2015.tsv` through `g_claims_2025.tsv` — patent_id, claim_text, dependent (11 yearly files)

## Pipeline (`src/phase1_data_pipeline/build_corpus.py`)

1. Load `g_patent.tsv`, filter to filing date 2015–2025
2. Load `g_cpc_current.tsv`, filter to `cpc_subclass == "G06T"`, deduplicate by `patent_id` (a patent can have multiple G06T-prefixed CPC codes; we keep first-occurrence as a membership flag only, not the full code list)
3. Inner join patent + CPC → G06T-only corpus in date range
4. Join in abstract text from `g_patent_abstract.tsv`
5. Weighted sampling to reach target corpus size — years 2022–2025 weighted higher than 2015–2021 (see data quality section below for why)
6. For each yearly claims file, filter to patents in our corpus and to **independent claims only** (`dependent` column is null — verified against real claim text, see below)
7. Concatenate independent claims per patent into one field; track `claims_status` (`found`/`missing`) per patent
8. Save to SQLite (`patents.db`), schema: `patent_id, title, abstract, claims_text, claims_status, abstract_status, filing_date`

## Data quality investigation

### Independent vs. dependent claims

The `dependent` column in the claims files is not boolean — it holds the parent claim label (e.g. `"claim 1"`) for dependent claims, and is `NaN` for independent claims. Verified by inspecting real claim text: rows with `dependent = NaN` were standalone claims; rows with `dependent = "claim 1"` explicitly referenced claim 1 in their text (e.g. *"The computer system of claim 1, wherein..."*). Confirmed via `src/phase1_data_pipeline/verify_claims_polarity.py`.

### Claims completeness gap (2015–2021)

Initial corpus build showed 43% of patents missing claims text overall. Investigation (documented via hypothesis elimination, not assumption):
1. Ruled out missing claims files — all 11 years (2015–2025) confirmed present
2. Ruled out `patent_id` dtype mismatch between files — both `int64`
3. Ruled out filter logic on the `dependent` column — polarity verified correct
4. Confirmed via cross-year search — a sample "missing" patent was checked against all 11 years' claims files and found in none

**Conclusion:** genuine gap in USPTO's bulk long-text dataset for older grants, not a pipeline bug. Missingness is heavily concentrated: 2015–2021 ranges from ~50% to ~92% missing per year (worst in 2020–2021), while 2022–2025 is ~98–100% complete. This is why the corpus sampling weights recent years more heavily — final corpus claims completeness after weighting: **7,063/10,000 (70.6%) found**.

### Missing abstracts

245/10,000 patents (2.45%) have no abstract text. Confirmed via `verify_abstract_nulls.py` that all cases are clean `NULL` values (0 empty strings, 0 whitespace-only) — no mixed data-quality issue. Added `abstract_status` column (`found`/`missing`) via `add_abstract.py`, matching the `claims_status` pattern. Consistency between the status flag and actual data verified via `verify_abstract_status_consistency.py`: 0 mismatches.

## Evaluation pairs

6 patent IDs selected as known-similar/known-dissimilar pairs for retrieval validation:

- **Similar pair 1:** `8929608` (3D position/orientation recognition) vs. `8933993` (pose determination for mobile device)
- **Similar pair 2:** `8937592` (3D content rendering on handheld device) vs. `8938093` (3D graphics + face-tracking interaction)
- **Dissimilar pair:** `8929636` (graph-based image segmentation) vs. `8930846` (app repositioning/UI management — only nominally G06T)

**Important caveat:** these pairs were selected by browsing patents *already inside* the built 10,000-patent corpus, not chosen independently beforehand and then checked for presence. All 6 were confirmed present (6/6), but this was close to guaranteed by construction — it is not strong evidence that the retrieval system can find genuinely known-similar patents from an independent source. If a stronger validation signal is needed, eval pairs should be re-selected from an external source, prior to and independent of corpus inspection.

Key findings of Phase 1 
Corpus: 10,000 G06T (computer vision) patents, 2015–2025, from USPTO Bulk Data Directory (PatentsView API blocked by ID.me verification, pivoted to bulk TSV downloads)
Schema: patent_id, title, abstract, cpc_subclass, claims_text, claims_status, abstract_status, filing_date in SQLite
Claims completeness: 7,063/10,000 (70.6%) — root-caused via hypothesis elimination to a genuine USPTO bulk-data gap concentrated in 2015–2021 (50–92% missing per year), while 2022–2025 is ~98–100% complete; corpus sampling weighted toward recent years accordingly
Abstract completeness: 9,755/10,000 (97.55%) — 245 confirmed clean NULLs, no data quality ambiguity
Independent vs. dependent claims: correctly distinguished via the dependent column (NaN = independent), verified against real claim text, not assumed
Known limitation (unresolved): subgroup-level CPC stratification not yet implemented — corpus is representative across years but not verified across G06T subtypes


##Decision Made 

Keep the current 6 pairs as an initial smoke test, but flag them as weak validation — re-select independently-sourced pairs before final validation if stronger evidence is needed.
Reason: The 6 pairs were selected by browsing patents already inside the built corpus, not chosen independently beforehand. Their presence in the corpus (6/6) was close to guaranteed, not a real test of retrieval quality.

Missing abstracts (245/10,000)

Answer: Keep them in the corpus with abstract_status = 'missing', and let Phase 2 fall back to title-only embedding for those specific patents in Stage 1.
Reason: Excluding them shrinks the corpus for no real gain — 245 patents (2.45%) is small, and title-only embedding is a legitimate degraded-but-usable fallback rather than losing the patents entirely.

CPC subgroup stratification (the new one your partner raised)

Answer: Not yet fixed. Pipeline currently stratifies by year only; cpc_group exists in the raw data but isn't used. A fix exists but hasn't been run.
Reason: This is genuinely unresolved — don't write it as decided.

## Open questions for partner sign-off

1. **Date range narrowing (2011–2026 → 2015–2025):** 2011–2014 claims files repeatedly failed to download from USPTO's bulk directory. Proceeded without them rather than continuing to retry — this was a unilateral call, not a joint decision. Needs explicit sign-off, or a decision to retry those years.
2. **Eval pair selection order:** see caveat above — needs a decision on whether the current pairs are sufficient or should be replaced with independently-sourced pairs.
3. **Missing-abstract handling for Phase 2:** exclude the 245 patents with missing abstracts from the corpus entirely, or keep them with the `abstract_status` flag and fall back to title-only embedding for those specific patents in Stage 1 retrieval.

## Final corpus stats

- Total patents: 10,000
- CPC scope: G06T only
- Date range: 2015-01 to 2025-12
- Claims found: 7,063 (70.6%) | missing: 2,937
- Abstract found: 9,755 (97.55%) | missing: 245
- Claims text length (found rows): mean 3,666 chars, median 3,213, 25th/75th percentile 2,268/4,512, max 48,049
