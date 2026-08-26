import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def fake_synthesize_patent(
    client,
    query,
    patent,
    fail_ids=None,
):
    """
    Simulates synthesize_patent() without calling Gemini.

    fail_ids:
        Optional set of patent IDs that should deliberately fail.
    """

    time.sleep(
        random.uniform(0.5, 2.0)
    )

    if fail_ids and patent["id"] in fail_ids:
        raise RuntimeError(
            f"Simulated failure for {patent['id']}"
        )

    return {
        "patent_id": patent["id"],
        "verdict": "HIGH_RELEVANCE",
        "overlap_summary": (
            f"Mock overlap for {patent['id']}"
        ),
        "key_difference": (
            f"Mock difference for {patent['id']}"
        ),
        "supporting_evidence": {
            "quote": "mock quote",
            "source": "claim",
        },
    }
def run_concurrent_mock(
    patents,
    query,
    fail_ids=None,
    max_workers=3,
):
    """
    Run mock synthesis concurrently.

    Results are returned in the original patent/rank order,
    regardless of which worker finishes first.

    Failed candidates are represented as:
        {
            "patent_id": "...",
            "error": "..."
        }
    """

    results = [None] * len(patents)

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        future_to_index = {
            executor.submit(
                fake_synthesize_patent,
                None,
                query,
                patent,
                fail_ids,
            ): index
            for index, patent in enumerate(patents)
        }

        for future in as_completed(
            future_to_index
        ):
            index = future_to_index[future]
            patent = patents[index]

            try:
                results[index] = future.result()

            except Exception as error:
                results[index] = {
                    "patent_id": patent["id"],
                    "error": str(error),
                }

    return results

if __name__ == "__main__":

    query = (
        "A deep learning system analyzes camera images "
        "to identify and classify objects in real time."
    )

    # =========================================================
    # TEST 1 — NORMAL CASE: 3 candidates
    # =========================================================

    print("=" * 60)
    print("TEST 1 — NORMAL CASE: 3 CANDIDATES")
    print("=" * 60)

    patents_3 = [
        {"id": "P003", "title": "Patent 3"},
        {"id": "P001", "title": "Patent 1"},
        {"id": "P005", "title": "Patent 5"},
    ]

    results = run_concurrent_mock(
        patents=patents_3,
        query=query,
        fail_ids={"P001"},
        max_workers=3,
    )

    assert len(results) == 3

    assert results[0]["patent_id"] == "P003"
    assert results[1]["patent_id"] == "P001"
    assert results[2]["patent_id"] == "P005"

    assert "error" in results[1]

    assert results[0]["verdict"] == "HIGH_RELEVANCE"
    assert results[2]["verdict"] == "HIGH_RELEVANCE"

    print("✓ 3 candidates completed")
    print("✓ Rank order preserved")
    print("✓ Deliberate failure captured")


    # =========================================================
    # TEST 2 — FEWER THAN 5: 2 candidates
    # =========================================================

    print("\n" + "=" * 60)
    print("TEST 2 — FEWER THAN 5: 2 CANDIDATES")
    print("=" * 60)

    patents_2 = [
        {"id": "P003", "title": "Patent 3"},
        {"id": "P001", "title": "Patent 1"},
    ]

    results = run_concurrent_mock(
        patents=patents_2,
        query=query,
        max_workers=2,
    )

    assert len(results) == 2

    assert results[0]["patent_id"] == "P003"
    assert results[1]["patent_id"] == "P001"

    assert results[0]["verdict"] == "HIGH_RELEVANCE"
    assert results[1]["verdict"] == "HIGH_RELEVANCE"

    print("✓ 2 candidates returned")
    print("✓ No assumption of 5 candidates")
    print("✓ Rank order preserved")


    # =========================================================
    # TEST 3 — FEWER THAN 5: 4 candidates
    # =========================================================

    print("\n" + "=" * 60)
    print("TEST 3 — FEWER THAN 5: 4 CANDIDATES")
    print("=" * 60)

    patents_4 = [
        {"id": "P003", "title": "Patent 3"},
        {"id": "P001", "title": "Patent 1"},
        {"id": "P005", "title": "Patent 5"},
        {"id": "P002", "title": "Patent 2"},
    ]

    results = run_concurrent_mock(
        patents=patents_4,
        query=query,
        max_workers=3,
    )

    assert len(results) == 4

    expected_ids = [
        "P003",
        "P001",
        "P005",
        "P002",
    ]

    actual_ids = [
        result["patent_id"]
        for result in results
    ]

    assert actual_ids == expected_ids

    print("✓ 4 candidates returned")
    print("✓ No assumption of 5 candidates")
    print("✓ Rank order preserved")


    # =========================================================
    # TEST 4 — ZERO CANDIDATES
    # =========================================================

    print("\n" + "=" * 60)
    print("TEST 4 — ZERO CANDIDATES")
    print("=" * 60)

    patents_0 = []

    results = run_concurrent_mock(
        patents=patents_0,
        query=query,
        max_workers=3,
    )

    assert results == []

    print("✓ Zero candidates returned []")
    print("✓ No exception raised")
    print("✓ No worker was submitted")


    # =========================================================
    # TEST 5 — FAILURE WITH FEWER THAN 5
    # =========================================================

    print("\n" + "=" * 60)
    print("TEST 5 — FAILURE WITH 2 CANDIDATES")
    print("=" * 60)

    results = run_concurrent_mock(
        patents=patents_2,
        query=query,
        fail_ids={"P003"},
        max_workers=2,
    )

    assert len(results) == 2

    # Original order must still be preserved.
    assert results[0]["patent_id"] == "P003"
    assert results[1]["patent_id"] == "P001"

    # P003 deliberately failed.
    assert "error" in results[0]

    # P001 succeeded.
    assert results[1]["verdict"] == "HIGH_RELEVANCE"

    print("✓ Failure captured")
    print("✓ Successful candidate preserved")
    print("✓ Rank order preserved")


    # =========================================================
    # FINAL VERIFICATION
    # =========================================================

    print("\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)

    print("✓ Normal 3-candidate case passed")
    print("✓ 2-candidate case passed")
    print("✓ 4-candidate case passed")
    print("✓ Zero-candidate case passed")
    print("✓ Failure propagation case passed")
    print("✓ Rank preservation passed")
    print("✓ No Gemini API calls were made")

    print("\nALL CONCURRENCY EDGE-CASE TESTS PASSED")