from data.toy_patents import TOY_PATENTS
from src.phase2_embedding_retrieval.embedding_pipeline import (
    get_client,
    build_indexes,
)
from src.phase3_llm_synthesis.phase3 import (
    run_patent_similarity,
)


QUERY = (
    "A deep learning system analyzes camera images "
    "to identify and classify objects in real time."
)


def main():
    print("=" * 60)
    print("PHASE 3 FAILURE-MODE TEST")
    print("=" * 60)

    client = get_client()

    # --------------------------------------------------
    # Build normal index
    # --------------------------------------------------

    print("\nBUILDING INDEX")

    index_data = build_indexes(
        client=client,
        patents=TOY_PATENTS,
        batch_size=100,
        delay=0,
    )

    print("\n✓ Normal index built")

    # --------------------------------------------------
    # TEST 1 — Normal case
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST 1 — NORMAL CASE")
    print("=" * 60)

    result = run_patent_similarity(
        client=client,
        index_data=index_data,
        query=QUERY,
        top_k=4,
        max_results=5,
        claims_batch_size=100,
        claims_delay=0,
    )

    assert len(result["stage1_candidates"]) > 0
    assert len(result["stage2_candidates"]) > 0

    print(
        f"✓ Stage 1 candidates: "
        f"{len(result['stage1_candidates'])}"
    )

    print(
        f"✓ Stage 2 candidates: "
        f"{len(result['stage2_candidates'])}"
    )

    print(
        f"✓ Phase 3 results: "
        f"{len(result['results'])}"
    )

    # --------------------------------------------------
    # TEST 2 — Fewer than 5 results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST 2 — FEWER THAN 5 RESULTS")
    print("=" * 60)

    result = run_patent_similarity(
        client=client,
        index_data=index_data,
        query=QUERY,
        top_k=2,
        max_results=5,
        claims_batch_size=100,
        claims_delay=0,
    )

    assert len(result["results"]) <= 2

    print(
        f"✓ Returned only available candidates: "
        f"{len(result['results'])}"
    )

    # --------------------------------------------------
    # TEST 3 — No claims available
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST 3 — NO CLAIMS")
    print("=" * 60)

    no_claim_patents = []

    for patent in TOY_PATENTS:
        copied = patent.copy()
        copied["claims"] = None
        no_claim_patents.append(copied)

    no_claim_index = build_indexes(
        client=client,
        patents=no_claim_patents,
        batch_size=100,
        delay=0,
    )

    result = run_patent_similarity(
        client=client,
        index_data=no_claim_index,
        query=QUERY,
        top_k=4,
        max_results=5,
        claims_batch_size=100,
        claims_delay=0,
    )

    assert len(result["stage1_candidates"]) > 0
    assert len(result["stage2_candidates"]) == 0
    assert len(result["results"]) == 0

    print(
        "✓ Stage 1 candidates exist"
    )

    print(
        "✓ Stage 2 candidates correctly reduced to 0"
    )

    print(
        "✓ Phase 3 correctly returned 0 results"
    )

    # --------------------------------------------------
    # TEST 4 — Invalid query
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST 4 — INVALID QUERY")
    print("=" * 60)

    try:

        run_patent_similarity(
            client=client,
            index_data=index_data,
            query="",
            top_k=4,
            max_results=5,
        )

        raise AssertionError(
            "Expected ValueError for empty query."
        )

    except ValueError:

        print(
            "✓ Empty query correctly rejected"
        )

    # --------------------------------------------------
    # TEST 5 — Invalid max_results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST 5 — INVALID max_results")
    print("=" * 60)

    try:

        run_patent_similarity(
            client=client,
            index_data=index_data,
            query=QUERY,
            top_k=4,
            max_results=0,
        )

        raise AssertionError(
            "Expected ValueError for max_results=0."
        )

    except ValueError:

        print(
            "✓ Invalid max_results correctly rejected"
        )

    # --------------------------------------------------
    # Final
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FAILURE-MODE TEST COMPLETE")
    print("=" * 60)

    print("\n✓ Normal case passed")
    print("✓ Fewer-than-5-results case passed")
    print("✓ No-claims case passed")
    print("✓ Invalid-query case passed")
    print("✓ Invalid-max-results case passed")

    print("\nALL FAILURE-MODE TESTS PASSED")


if __name__ == "__main__":
    main()