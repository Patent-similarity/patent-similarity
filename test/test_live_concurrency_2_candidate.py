import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import time
import json

from data.toy_patents import TOY_PATENTS
from src.phase2_embedding_retrieval.embedding_pipeline import (
    get_client,
    build_indexes,
)
import src.phase3_llm_synthesis.phase3 as phase3


# =========================================================
# Test configuration
# =========================================================

FAIL_PATENT_ID = "P003"

QUERY = (
    "A deep learning system analyzes camera images "
    "to identify and classify objects in real time."
)


# =========================================================
# Test-only worker wrapper
# =========================================================

REAL_SYNTHESIZE_CANDIDATE = phase3._synthesize_candidate


def test_synthesize_candidate(
    client,
    query,
    patent,
    rank,
    retrieval,
    idx,
):
    """
    Test-only wrapper.

    P003 fails immediately, before touching the Gemini client.

    Every other candidate goes through the real production
    _synthesize_candidate() implementation unchanged.
    """

    if patent["id"] == FAIL_PATENT_ID:
        raise RuntimeError(
            f"DELIBERATE TEST FAILURE for patent {patent['id']}"
        )

    return REAL_SYNTHESIZE_CANDIDATE(
        client=client,
        query=query,
        patent=patent,
        rank=rank,
        retrieval=retrieval,
        idx=idx,
    )


# =========================================================
# Main test
# =========================================================

def main():

    print("=" * 60)
    print("PHASE 3 DETERMINISTIC CONCURRENCY FAILURE TEST")
    print("=" * 60)

    client = get_client()

    # -----------------------------------------------------
    # Build the real 5-patent index
    # -----------------------------------------------------

    five_real = [
        p
        for p in TOY_PATENTS
        if p["id"] in {
            "P001",
            "P002",
            "P003",
            "P004",
            "P005",
        }
    ]

    print("\nBUILDING INDEX")

    index_data = build_indexes(
        client=client,
        patents=five_real,
        batch_size=100,
        delay=0,
    )

    print("\n✓ Real 5-patent index built")

    # -----------------------------------------------------
    # Temporarily replace the concurrent worker
    # -----------------------------------------------------

    phase3._synthesize_candidate = test_synthesize_candidate

    try:

        print(
            f"\n✓ Test wrapper installed "
            f"(forced failure: {FAIL_PATENT_ID})"
        )

        start = time.time()

        result = phase3.run_patent_similarity(
            client=client,
            index_data=index_data,
            query=QUERY,
            top_k=5,
            max_results=5,
            claims_batch_size=100,
            claims_delay=0,
            max_workers=2,
        )

        elapsed = time.time() - start

    finally:

        # -------------------------------------------------
        # ALWAYS restore the real production worker.
        # -------------------------------------------------

        phase3._synthesize_candidate = REAL_SYNTHESIZE_CANDIDATE

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print(f"\nElapsed: {elapsed:.2f}s")

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    # -----------------------------------------------------
    # Assertions
    # -----------------------------------------------------

    assert len(result["stage1_candidates"]) == 5

    assert len(result["stage2_candidates"]) == 5

    assert len(result["results"]) == 5

    # Results must remain in rank order.
    ranks = [
        result["results"][i]["rank"]
        for i in range(len(result["results"]))
    ]

    assert ranks == [1, 2, 3, 4, 5], (
        f"Results are not in rank order: {ranks}"
    )

    # Find the deliberately failed candidate.
    failed_results = [
        r
        for r in result["results"]
        if r["patent_id"] == FAIL_PATENT_ID
    ]

    assert len(failed_results) == 1

    failed = failed_results[0]

    assert "error" in failed

    assert "synthesis" not in failed

    assert "DELIBERATE TEST FAILURE" in failed["error"]

    # Every other candidate must have a real synthesis.
    successful_results = [
        r
        for r in result["results"]
        if r["patent_id"] != FAIL_PATENT_ID
    ]

    assert len(successful_results) == 4

    for r in successful_results:
        assert "synthesis" in r
        assert "error" not in r

    # -----------------------------------------------------
    # Final confirmation
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST PASSED")
    print("=" * 60)

    print(
        f"\n✓ {FAIL_PATENT_ID} failed before any Gemini synthesis call"
    )

    print(
        "✓ Remaining 4 candidates used real synthesize_patent()"
    )

    print(
        "✓ All 5 candidates are present in results"
    )

    print(
        "✓ Results remain in rank order"
    )

    print(
        "✓ Failed candidate contains an error marker"
    )

    print(
        "✓ Failed candidate has no synthesis"
    )

    print(
        f"\n✓ Real Gemini synthesis calls: 4"
    )

    print(
        "\nALL DETERMINISTIC CONCURRENCY FAILURE TESTS PASSED"
    )


if __name__ == "__main__":
    main()