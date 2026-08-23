import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import random

from data.real_corpus import load_real_corpus
from src.phase2_embedding_retrieval.embedding_pipeline import (
    get_client,
    build_indexes,
    evaluate_query,
)


PAIRS = {
    "S1": ("11615557", "11620767"),
    "S2": ("11676310", "11790567"),
    "S3": ("10674162", "11818368"),
    "D1": ("11676310", "11869140"),
    "D2": ("11615557", "11776314"),
    "D3": ("12307167", "10229528"),
}

client = get_client()

patents = load_real_corpus()
print(f"Loaded real corpus: {len(patents)} patents")

random.seed(42)
needed_ids = {pid for pair in PAIRS.values() for pid in pair}
must_include = [p for p in patents if p["id"] in needed_ids]
remaining = [p for p in patents if p["id"] not in needed_ids]
sample = random.sample(remaining, 80)

patents = must_include + sample
print(f"Sliced corpus for this test: {len(patents)} patents")

index_data = build_indexes(client, patents)
print("Indexes built once.")

patent_by_id = {str(patent["id"]): patent for patent in patents}

for pair_name, (query_id, partner_id) in PAIRS.items():
    query_id = str(query_id)
    partner_id = str(partner_id)

    if query_id not in patent_by_id:
        print(f"{pair_name}: ERROR — query {query_id} not in corpus")
        continue
    if partner_id not in patent_by_id:
        print(f"{pair_name}: ERROR — partner {partner_id} not in corpus")
        continue

    query_text = patent_by_id[query_id]["abstract"]
    results = evaluate_query(client, index_data, query_text, top_k=50)

    stage1_ids = [str(patents[idx]["id"]) for idx in results["stage1_indices"]]
    stage2_ids = [str(patents[idx]["id"]) for idx in results["stage2_indices"]]

    stage1_rank = stage1_ids.index(partner_id) + 1 if partner_id in stage1_ids else None
    stage2_rank = stage2_ids.index(partner_id) + 1 if partner_id in stage2_ids else None

    print(f"\n===== {pair_name} =====")
    print(f"Query:   {query_id}")
    print(f"Partner: {partner_id}")
    print(f"Stage 1: {'position ' + str(stage1_rank) if stage1_rank else 'NOT in top-50'}")
    print(f"Stage 2: {'position ' + str(stage2_rank) if stage2_rank else 'NOT in eligible candidates'}")