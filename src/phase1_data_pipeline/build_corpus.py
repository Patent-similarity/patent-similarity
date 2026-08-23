"""
Phase 1 Pipeline — Build G06T patent corpus into SQLite.
Stratified by (year, cpc_group) so the corpus is representative both
across time and across G06T subgroups (e.g. object tracking, image
enhancement, 3D rendering), not just across years.

Steps:
1. Load g_patent.tsv, filter to date range.
2. Load g_cpc_current.tsv, filter to cpc_subclass == "G06T", dedupe by
   patent_id, keep cpc_group (subgroup-level code).
3. Join patent + cpc.
4. Join in abstract text.
5. Stratified sampling by (year, cpc_group) toward CORPUS_CAP total.
   Uses a combined string key for grouping instead of a multi-column
   groupby, to avoid a pandas 3.x quirk where groupby(["a","b"]).apply()
   can drop the grouping columns from the result.
6. Load claims year by year, keep independent claims only, attach to corpus.
7. Track claims_status per patent: 'found' or 'missing'.
8. Save to SQLite, patent_id as PRIMARY KEY.
"""

import pandas as pd
import sqlite3
import os

DATASETS_DIR = "datasets"
DB_PATH = "patents.db"

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
TARGET_CPC_SUBCLASSES = ["G06T"]
CORPUS_CAP = 10000
CLAIMS_YEARS = list(range(2015, 2026))
RECENT_YEARS = {"2022", "2023", "2024", "2025"}

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
# STEP 2: Load CPC file, filter to G06T, keep cpc_group
# ---------------------------------------------------------------------
print("Step 2: Loading g_cpc_current.tsv ...")
cpc = pd.read_csv(
    os.path.join(DATASETS_DIR, "g_cpc_current.tsv"),
    sep="\t",
    usecols=["patent_id", "cpc_subclass", "cpc_group"]
)
g06_target = cpc[cpc["cpc_subclass"].isin(TARGET_CPC_SUBCLASSES)]
g06_target = g06_target.drop_duplicates(subset="patent_id")
print(f"  Unique G06T patents (any date): {len(g06_target)}")
print(f"  Unique cpc_group values found: {g06_target['cpc_group'].nunique()}")

# ---------------------------------------------------------------------
# STEP 3: Join patent + cpc
# ---------------------------------------------------------------------
print("Step 3: Joining patent + CPC ...")
corpus = patents.merge(
    g06_target[["patent_id", "cpc_subclass", "cpc_group"]], on="patent_id", how="inner"
)
print(f"  G06T patents in date range: {len(corpus)}")

# ---------------------------------------------------------------------
# STEP 4: Bring in abstract text
# ---------------------------------------------------------------------
print("Step 4: Loading g_patent_abstract.tsv ...")
abstracts = pd.read_csv(os.path.join(DATASETS_DIR, "g_patent_abstract.tsv"), sep="\t")
corpus = corpus.merge(abstracts, on="patent_id", how="left")
print(f"  Patents missing abstract: {corpus['patent_abstract'].isna().sum()}")

# ---------------------------------------------------------------------
# STEP 5: Stratified sampling by (year, cpc_group), using a combined
# key column instead of multi-column groupby to sidestep the pandas
# column-dropping issue.
# ---------------------------------------------------------------------
corpus["year"] = corpus["patent_date"].str[:4]
corpus["_strata_key"] = corpus["year"] + "|" + corpus["cpc_group"].astype(str)

num_groups = corpus["cpc_group"].nunique()
num_years = corpus["year"].nunique()
print(f"Step 5: Stratifying across {num_groups} cpc_group values x {num_years} years")

total_cells_recent = len(RECENT_YEARS) * num_groups
total_cells_older = (num_years - len(RECENT_YEARS)) * num_groups
recent_weight, older_weight = 2.4, 1.0
denom = (total_cells_recent * recent_weight) + (total_cells_older * older_weight)
recent_per_cell = max(1, round((CORPUS_CAP * recent_weight) / denom))
older_per_cell = max(1, round((CORPUS_CAP * older_weight) / denom))
print(f"  Target per (year, cpc_group) cell: recent={recent_per_cell}, older={older_per_cell}")

def sample_group(group):
    year = group["year"].iloc[0]
    target = recent_per_cell if year in RECENT_YEARS else older_per_cell
    n = min(target, len(group))
    return group.sample(n=n, random_state=42)

# groupby on the single combined key column -- this one survives .apply()
sampled_parts = [sample_group(g) for _, g in corpus.groupby("_strata_key")]
corpus = pd.concat(sampled_parts, ignore_index=True)
corpus = corpus.drop(columns=["_strata_key", "year"], errors="ignore")

print(f"Step 5: Corpus size after stratified sampling: {len(corpus)}")
print("  Subgroup distribution in final corpus (top 20):")
print(corpus["cpc_group"].value_counts().head(20))

corpus_ids = set(corpus["patent_id"])

# ---------------------------------------------------------------------
# STEP 6: Load claims year by year, independent claims only
# ---------------------------------------------------------------------
print("Step 6: Loading claims files ...")
claims_by_patent = {}
for year in CLAIMS_YEARS:
    fname = os.path.join(DATASETS_DIR, f"g_claims_{year}.tsv")
    if not os.path.exists(fname):
        print(f"  [SKIP] {fname} not found on disk")
        continue
    print(f"  Reading {fname} ...")
    yearly = pd.read_csv(fname, sep="\t", usecols=["patent_id", "claim_text", "dependent"])
    yearly = yearly[yearly["patent_id"].isin(corpus_ids)]
    yearly = yearly[yearly["dependent"].isna()]
    for pid, group in yearly.groupby("patent_id"):
        text = " ".join(group["claim_text"].astype(str).tolist())
        claims_by_patent.setdefault(pid, []).append(text)

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
# STEP 7: Save to SQLite, cpc_group included
# ---------------------------------------------------------------------
print(f"Step 7: Writing to {DB_PATH} ...")
corpus_final = corpus.rename(columns={
    "patent_title": "title",
    "patent_date": "filing_date",
    "patent_abstract": "abstract",
})[["patent_id", "title", "abstract", "cpc_subclass", "cpc_group",
    "claims_text", "claims_status", "filing_date"]]

conn = sqlite3.connect(DB_PATH)
corpus_final.to_sql("patents", conn, if_exists="replace", index=False,
                     dtype={"patent_id": "INTEGER PRIMARY KEY"})
conn.close()

print("Done. Corpus saved.")
print(f"Total patents: {len(corpus_final)}")
print(f"With claims: {found} | Missing claims: {missing}")

corpus_final["claims_len_chars"] = corpus_final["claims_text"].fillna("").str.len()
print("\n--- Claims text length stats (found rows only) ---")
print(corpus_final[corpus_final["claims_status"] == "found"]["claims_len_chars"].describe())
print("\n--- Missing abstract count ---")
print((corpus_final["abstract"].isna()).sum())