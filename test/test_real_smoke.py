from data.real_corpus import load_real_corpus
from src.phase2_embedding_retrieval.embedding_pipeline import get_client, build_indexes, evaluate_query, print_ranking, select_full_treatment


client = get_client()

# Only 20 real patents — keeps API usage small.
all_patents = load_real_corpus()

with_claims = [p for p in all_patents if p["claims"] is not None][:10]
without_claims = [p for p in all_patents if p["claims"] is None][:10]

patents = with_claims + without_claims

print("Loaded real-data slice:", len(patents))

index_data = build_indexes(client, patents)

print("Abstract index size:", index_data["abstract_index"].ntotal)

results = evaluate_query(
    client,
    index_data,
    "a system for detecting objects in images using a neural network",
)

print("\n===== STAGE 1 =====")
print_ranking(
    patents,
    results["stage1_indices"],
    results["abstract_scores"],
)

print("\n===== STAGE 2 =====")
print_ranking(
    patents,
    results["stage2_indices"],
    results["final_scores"],
)

full_treatment = select_full_treatment(
    results["stage2_indices"]
)

print("\nFull treatment candidates:", len(full_treatment))
print("Stage 2 eligible:", len(results["stage2_indices"]))