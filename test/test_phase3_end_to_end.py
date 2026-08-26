import json
import time

from data.toy_patents import TOY_PATENTS

from src.phase2_embedding_retrieval.embedding_pipeline import (
    get_client,
    build_indexes,
    evaluate_query,
    select_full_treatment,
)

from src.phase3_llm_synthesis.phase3 import (
    synthesize_patent,
)


def main():

    # =====================================================
    # Configuration
    # =====================================================

    query = (
        "A deep learning system analyzes camera images "
        "to identify and classify objects in real time."
    )

    print()
    print("=" * 60)
    print("PHASE 3 TIMING BREAKDOWN TEST")
    print("=" * 60)

    print()
    print("Query:")
    print(query)

    total_start = time.perf_counter()

    # =====================================================
    # Gemini client
    # =====================================================

    client = get_client()

    # =====================================================
    # Phase 2 — build index
    # =====================================================

    print()
    print("=" * 60)
    print("BUILDING PHASE 2 INDEX")
    print("=" * 60)

    index_start = time.perf_counter()

    index_data = build_indexes(
        client=client,
        patents=TOY_PATENTS,
        batch_size=100,
        delay=0,
        max_retries=5,
    )

    index_end = time.perf_counter()

    index_time = (
        index_end - index_start
    )

    print()
    print(
        f"Index build time: "
        f"{index_time:.2f} seconds"
    )

    # =====================================================
    # Phase 2 — Stage 1 + Stage 2
    # =====================================================

    print()
    print("=" * 60)
    print("RUNNING PHASE 2")
    print("=" * 60)

    phase2_start = time.perf_counter()

    retrieval = evaluate_query(
        client=client,
        index_data=index_data,
        query_text=query,
        top_k=4,
        claims_batch_size=100,
        claims_delay=0,
        max_retries=5,
    )

    phase2_end = time.perf_counter()

    phase2_time = (
        phase2_end - phase2_start
    )

    print()
    print(
        f"Phase 2 time: "
        f"{phase2_time:.2f} seconds"
    )

    # =====================================================
    # Select Stage 2 candidates
    # =====================================================

    selected_indices = select_full_treatment(
        retrieval["stage2_indices"],
        max_results=5,
    )

    print()
    print(
        f"Stage 1 candidates: "
        f"{len(retrieval['stage1_indices'])}"
    )

    print(
        f"Stage 2 candidates: "
        f"{len(retrieval['stage2_indices'])}"
    )

    print(
        f"Phase 3 candidates: "
        f"{len(selected_indices)}"
    )

    # =====================================================
    # Phase 3 — individual timing
    # =====================================================

    print()
    print("=" * 60)
    print("RUNNING PHASE 3")
    print("=" * 60)

    phase3_start = time.perf_counter()

    results = []

    for rank, idx in enumerate(
        selected_indices,
        start=1,
    ):

        patent = index_data["patents"][idx]

        print()
        print(
            f"Phase 3 candidate {rank}: "
            f"{patent['id']} — "
            f"{patent['title']}"
        )

        candidate_start = time.perf_counter()

        synthesis = synthesize_patent(
            client=client,
            query=query,
            patent=patent,
        )

        candidate_end = time.perf_counter()

        candidate_time = (
            candidate_end - candidate_start
        )

        print(
            f"Synthesis time: "
            f"{candidate_time:.2f} seconds"
        )

        results.append(
            {
                "rank": rank,
                "patent_id": patent["id"],
                "title": patent["title"],
                "abstract_score": float(
                    retrieval["abstract_scores"][idx]
                ),
                "claim_score": float(
                    retrieval["claim_scores"][idx]
                ),
                "final_score": float(
                    retrieval["final_scores"][idx]
                ),
                "synthesis": synthesis,
                "synthesis_time": candidate_time,
            }
        )

    phase3_end = time.perf_counter()

    phase3_time = (
        phase3_end - phase3_start
    )

    # =====================================================
    # Print final results
    # =====================================================

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for item in results:

        print()
        print(
            f"Rank {item['rank']}"
        )

        print(
            f"Patent ID: "
            f"{item['patent_id']}"
        )

        print(
            f"Title: "
            f"{item['title']}"
        )

        print(
            f"Abstract score: "
            f"{item['abstract_score']:.4f}"
        )

        print(
            f"Claim score: "
            f"{item['claim_score']:.4f}"
        )

        print(
            f"Final score: "
            f"{item['final_score']:.4f}"
        )

        print(
            f"Synthesis time: "
            f"{item['synthesis_time']:.2f} seconds"
        )

        print(
            json.dumps(
                item["synthesis"],
                indent=2,
            )
        )

        print("-" * 60)

    # =====================================================
    # Timing summary
    # =====================================================

    total_end = time.perf_counter()

    total_time = (
        total_end - total_start
    )

    print()
    print("=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)

    print(
        f"Index build:       "
        f"{index_time:.2f} sec"
    )

    print(
        f"Phase 2:           "
        f"{phase2_time:.2f} sec"
    )

    print(
        f"Phase 3:           "
        f"{phase3_time:.2f} sec"
    )

    print(
        f"Total:             "
        f"{total_time:.2f} sec"
    )

    # =====================================================
    # Phase 3 breakdown
    # =====================================================

    print()
    print("=" * 60)
    print("PHASE 3 INDIVIDUAL CALLS")
    print("=" * 60)

    for item in results:

        print(
            f"{item['patent_id']}: "
            f"{item['synthesis_time']:.2f} sec"
        )

    # =====================================================
    # Verification
    # =====================================================

    print()
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    assert retrieval["stage1_indices"].size <= 4

    assert len(
        retrieval["stage2_indices"]
    ) <= len(
        retrieval["stage1_indices"]
    )

    assert len(results) <= 5

    for item in results:

        synthesis = item["synthesis"]

        assert synthesis["verdict"]

        assert synthesis["overlap_summary"]

        assert synthesis["key_difference"]

        assert synthesis["supporting_evidence"]

    print()
    print("✓ Phase 2 completed")
    print("✓ Phase 3 completed")
    print("✓ Individual synthesis timings recorded")
    print("✓ Output validation passed")
    print("✓ Timing breakdown completed")


if __name__ == "__main__":
    main()