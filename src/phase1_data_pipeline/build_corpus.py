"""
Phase 1 Pipeline — Build G06T patent corpus into SQLite.

Steps (matches the reasoning we worked through together):
1. Load g_patent.tsv, filter to date range.
2. Load g_cpc_current.tsv, filter to cpc_subclass == "G06T", dedupe by patent_id.
3. Join patent + cpc (inner join -> only G06T patents survive).
4. Load g_patent_abstract.tsv, join in abstract text.
5. Cap corpus size (~1000 patents) for MVP scope.
6. For each year's claims file, filter to patents in our corpus + independent claims only
   (dependent.isna()), concatenate multiple independent claims per patent into one string.
7. Track claims_status per patent: 'found' or 'missing' (idempotency/failure-tracking design).
8. Save everything into SQLite with patent_id as PRIMARY KEY.
"""

import pandas as pd
import sqlite3
import os

DATASETS_DIR = "datasets"
DB_PATH = "patents.db"

# ---- CONFIG: adjust these based on what you actually have downloaded ----
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
TARGET_CPC_SUBCLASSES = ["G06T"]  # locked Phase 0 spec -- G06T only
CORPUS_CAP = 1000
CLAIMS_YEARS = list(range(2015, 2026))  # matches your downloaded years

# ---------------------------------------------------------------------
# STEP 1: Load main patent records, filter by date
# ---------------------------------------------------------------------
print("Step 1: Loading g_patent.tsv ...")
patents = pd.read_csv(
    os.path.join(DATASETS_DIR, "g_patent.tsv"),
    sep="\t",
    usecols=["patent_id", "patent_date", "patent_title"]
)
patents = patents[
    (patents["patent_date"] >= START_DATE) & (patents["patent_date"] <= END_DATE)
]
print(f"  Patents in date range: {len(patents)}")

# ---------------------------------------------------------------------
# STEP 2: Load CPC file, filter to G06T, dedupe (per our earlier reasoning)
# ---------------------------------------------------------------------
print("Step 2: Loading g_cpc_current.tsv ...")
cpc = pd.read_csv(
    os.path.join(DATASETS_DIR, "g_cpc_current.tsv"),
    sep="\t",
    usecols=["patent_id", "cpc_subclass"]
)
g06_target = cpc[cpc["cpc_subclass"].isin(TARGET_CPC_SUBCLASSES)]
# A patent could technically have codes in both classes -- keep first match,
# so each patent is counted once, under one class, not double-counted.
g06_target = g06_target.drop_duplicates(subset="patent_id")
print(f"  Unique G06T+G06N patents (any date): {len(g06_target)}")
print(g06_target["cpc_subclass"].value_counts())

# ---------------------------------------------------------------------
# STEP 3: Join patent + cpc -> only G06T patents in our date range survive
# ---------------------------------------------------------------------
print("Step 3: Joining patent + CPC ...")
corpus = patents.merge(g06_target[["patent_id", "cpc_subclass"]], on="patent_id", how="inner")
print(f"  G06T+G06N patents in date range: {len(corpus)}")

# ---------------------------------------------------------------------
# STEP 4: Bring in abstract text
# ---------------------------------------------------------------------
print("Step 4: Loading g_patent_abstract.tsv ...")
abstracts = pd.read_csv(
    os.path.join(DATASETS_DIR, "g_patent_abstract.tsv"),
    sep="\t"
)
corpus = corpus.merge(abstracts, on="patent_id", how="left")
missing_abstract = corpus["patent_abstract"].isna().sum()
print(f"  Patents missing abstract: {missing_abstract}")

# ---------------------------------------------------------------------
# STEP 5: Weighted stratified sampling toward ~5000 total.
# Recent years (2022-2025) get more weight -- we confirmed via
# check_missing.py that claims coverage there is ~98-100% complete.
# Older years (2015-2021) get less weight since 50-92% of claims are
# missing there -- still included, for date-range coverage, just fewer.
# ---------------------------------------------------------------------
corpus["year"] = corpus["patent_date"].str[:4]

# ---------------------------------------------------------------------
# STEP 5: Weighted sampling toward 10,000 total (confirmed with partner
# based on Gemini quota + FAISS latency, not guessed). G06T only.
# Recent years (2022-2025) weighted higher -- claims coverage there is
# ~98-100% complete, vs 50-92% missing for 2015-2021 (verified earlier).
# ---------------------------------------------------------------------
corpus["year"] = corpus["patent_date"].str[:4]

RECENT_YEARS = {"2022", "2023", "2024", "2025"}
RECENT_PER_YEAR = 1450   # 4 years x 1450 = 5800
OLDER_PER_YEAR = 600     # 7 years x 600 = 4200  (total = 10,000)

def sample_year_group(group):
    year = group.name
    target = RECENT_PER_YEAR if year in RECENT_YEARS else OLDER_PER_YEAR
    n = min(target, len(group))
    return group.sample(n=n, random_state=42)

corpus = corpus.groupby("year", group_keys=False).apply(sample_year_group)
corpus = corpus.drop(columns=["year"], errors="ignore")
print(f"Step 5: Corpus size after weighted sampling: {len(corpus)}")

corpus_ids = set(corpus["patent_id"])

# ---------------------------------------------------------------------
# STEP 6: Load claims year by year, filter to our corpus + independent claims only
# ---------------------------------------------------------------------
print("Step 6: Loading claims files ...")
claims_by_patent = {}  # patent_id -> list of independent claim texts

for year in CLAIMS_YEARS:
    fname = os.path.join(DATASETS_DIR, f"g_claims_{year}.tsv")
    if not os.path.exists(fname):
        print(f"  [SKIP] {fname} not found on disk")
        continue

    print(f"  Reading {fname} ...")
    yearly = pd.read_csv(
        fname, sep="\t",
        usecols=["patent_id", "claim_text", "dependent"]
    )

    # keep only claims belonging to patents in our corpus (saves memory)
    yearly = yearly[yearly["patent_id"].isin(corpus_ids)]

    # independent claims only -> dependent is NaN (verified earlier against real text)
    yearly = yearly[yearly["dependent"].isna()]

    for pid, group in yearly.groupby("patent_id"):
        text = " ".join(group["claim_text"].astype(str).tolist())
        claims_by_patent.setdefault(pid, []).append(text)

# collapse to one string per patent, track status
def get_claims_and_status(pid):
    texts = claims_by_patent.get(pid)
    if texts:
        return " ".join(texts), "found"
    return None, "missing"

claims_col = corpus["patent_id"].apply(get_claims_and_status)
corpus["claims_text"] = claims_col.apply(lambda x: x[0])
corpus["claims_status"] = claims_col.apply(lambda x: x[1])

found = (corpus["claims_status"] == "found").sum()
missing = (corpus["claims_status"] == "missing").sum()
print(f"  Claims found: {found} | Claims missing: {missing}")

# ---------------------------------------------------------------------
# STEP 7: Save to SQLite
# ---------------------------------------------------------------------
print(f"Step 7: Writing to {DB_PATH} ...")
corpus_final = corpus.rename(columns={
    "patent_title": "title",
    "patent_date": "filing_date",
    "patent_abstract": "abstract",
})[["patent_id", "title", "abstract", "cpc_subclass", "claims_text", "claims_status", "filing_date"]]

conn = sqlite3.connect(DB_PATH)
corpus_final.to_sql("patents", conn, if_exists="replace", index=False,
                     dtype={"patent_id": "INTEGER PRIMARY KEY"})
conn.close()

print("Done. Corpus saved.")
print(f"Total patents: {len(corpus_final)}")
print(f"With claims: {found} | Missing claims: {missing}")

# ---- Additional diagnostics for partner's cap calculation ----
corpus_final["claims_len_chars"] = corpus_final["claims_text"].fillna("").str.len()
print("\n--- Claims text length stats (characters, 'found' rows only) ---")
print(corpus_final[corpus_final["claims_status"] == "found"]["claims_len_chars"].describe())

print("\n--- Missing abstract count ---")
print((corpus_final["abstract"].isna()).sum())