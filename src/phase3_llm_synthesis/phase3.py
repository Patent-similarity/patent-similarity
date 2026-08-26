import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.toy_patents import TOY_PATENTS
from src.phase2_embedding_retrieval.embedding_pipeline import (
    get_client,
    evaluate_query,
    select_full_treatment,
)


# =========================================================
# Configuration
# =========================================================

SYNTHESIS_MODEL = "gemini-3.6-flash"

VERDICTS = {
    "HIGH_RELEVANCE",
    "POSSIBLE_RELEVANCE",
    "LOW_RELEVANCE",
    "INSUFFICIENT_EVIDENCE",
}


# =========================================================
# Phase 3 — synthesis prompt
# =========================================================

def build_prompt(query, patent):
    return f"""
You are evaluating the relevance of a patent candidate to a user's invention
description.

Your task is to determine how technically relevant the candidate patent is to
the user's invention description. Base the judgment only on the text provided
below.

USER QUERY:
{query}

CANDIDATE PATENT ABSTRACT:
{patent["abstract"]}

CANDIDATE PATENT CLAIMS:
{patent["claims"]}

Return exactly one JSON object with exactly these four fields:

{{
  "verdict": "...",
  "overlap_summary": "...",
  "key_difference": "...",
  "supporting_evidence": {{
    "quote": "...",
    "source": "..."
  }}
}}

VERDICT:
The verdict must be exactly one of these four values:

- HIGH_RELEVANCE
- POSSIBLE_RELEVANCE
- LOW_RELEVANCE
- INSUFFICIENT_EVIDENCE

Use HIGH_RELEVANCE when the candidate and query describe substantially
the same invention, technical approach, or core functionality, with strong
technical overlap.

Use POSSIBLE_RELEVANCE when there is meaningful technical overlap, but the
candidate differs in an important technical problem, application, mechanism,
or implementation detail. The connection should be more than merely sharing
general terminology or a broad technology such as "machine learning."

Use LOW_RELEVANCE when the candidate contains enough specific technical
information to establish that it addresses a substantially different
technical problem, application, or mechanism, even if some general
technology or terminology overlaps with the query.

Use INSUFFICIENT_EVIDENCE when the candidate's abstract and claims are too
generic, vague, or boilerplate to establish its actual technical subject
matter well enough to make a reliable relevance judgment.

Do not use LOW_RELEVANCE merely because the candidate fails to mention the
query's specific features. If the candidate itself lacks enough technical
specificity to determine what the invention actually does, use
INSUFFICIENT_EVIDENCE instead.

The distinction is:

- HIGH_RELEVANCE = strong technical match
- POSSIBLE_RELEVANCE = meaningful but incomplete or uncertain technical match
- LOW_RELEVANCE = sufficiently specific candidate, but substantially
  different technical subject matter
- INSUFFICIENT_EVIDENCE = candidate text is too vague or generic to make
  a reliable technical judgment

The candidate always has abstract and claims text in this evaluation.
INSUFFICIENT_EVIDENCE therefore means that the available content is not
technically specific enough to support a reliable judgment. It does not mean
that text is missing.

OVERLAP SUMMARY:
Briefly explain the main meaningful technical overlap between the query and
candidate. Do not treat generic words or broad technologies alone as strong
technical overlap.

KEY DIFFERENCE:
Briefly explain the most important technical difference between the query and
candidate. If the verdict is INSUFFICIENT_EVIDENCE, explain that the candidate
does not provide enough technical specificity to establish a reliable
comparison.

SUPPORTING EVIDENCE:
The supporting_evidence object must contain:

- quote: exactly one contiguous, verbatim quotation copied from either the
  candidate claims or candidate abstract.
- source: exactly either "claim" or "abstract".

Evidence rules:

1. The quote must be copied exactly from the source text provided above.
2. The quote must be a contiguous substring of that source text.
3. Do not paraphrase the quote.
4. Do not summarize inside the quote.
5. Do not combine or stitch together separate passages.
6. Do not invent words that are not present in the source.
7. The quote should directly support the reasoning behind the verdict.
8. When claims text directly supports the verdict, prefer claims evidence
   because claims provide the stronger technical signal.
9. If the claims do not provide suitable evidence, use the abstract instead.
10. For INSUFFICIENT_EVIDENCE, choose a real passage demonstrating the
    candidate's generic or nonspecific technical disclosure when possible.

IMPORTANT:
Return only the JSON object. Do not include markdown fences, commentary,
explanations, or any text before or after the JSON object.
"""


# =========================================================
# Phase 3 — output validation
# =========================================================

def validate_output(result, patent):
    """Validate one Phase 3 synthesis result."""

    assert isinstance(
        result,
        dict,
    ), "Output is not a JSON object."

    required = {
        "verdict",
        "overlap_summary",
        "key_difference",
        "supporting_evidence",
    }

    assert required == set(result.keys()), (
        f"Wrong fields. Got: {set(result.keys())}"
    )

    assert result["verdict"] in VERDICTS, (
        f"Invalid verdict: {result['verdict']}"
    )

    evidence = result["supporting_evidence"]

    assert isinstance(
        evidence,
        dict,
    )

    assert set(evidence.keys()) == {
        "quote",
        "source",
    }

    assert evidence["source"] in {
        "claim",
        "abstract",
    }

    quote = evidence["quote"]

    assert isinstance(
        quote,
        str,
    )

    assert quote.strip(), (
        "Evidence quote is empty."
    )

    if evidence["source"] == "claim":
        source_text = patent["claims"]
    else:
        source_text = patent["abstract"]

    assert source_text is not None

    assert quote in source_text, (
        f"Evidence quote is NOT a contiguous substring "
        f"of the {evidence['source']} text."
    )

    return True


# =========================================================
# Phase 3 — synthesize one patent
# =========================================================

def synthesize_patent(
    client,
    query,
    patent,
):
    """
    Generate and validate a Phase 3 relevance verdict
    for one patent candidate.
    """

    prompt = build_prompt(
        query,
        patent,
    )

    try:

        response = client.models.generate_content(
            model=SYNTHESIS_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
            },
        )

    except Exception as error:

        raise RuntimeError(
            f"Gemini synthesis request failed for "
            f"patent {patent['id']}."
        ) from error

    if not response.text:

        raise ValueError(
            f"Empty synthesis response for "
            f"patent {patent['id']}."
        )

    try:

        result = json.loads(
            response.text
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            f"Invalid JSON returned by synthesis model "
            f"for patent {patent['id']}."
        ) from error

    validate_output(
        result,
        patent,
    )

    return result


# =========================================================
# Internal worker for concurrent Phase 3
# =========================================================

def _synthesize_candidate(
    client,
    query,
    patent,
    rank,
    retrieval,
    idx,
):
    """
    Worker executed by a ThreadPoolExecutor.

    Each candidate is synthesized independently.
    """

    verdict = synthesize_patent(
        client=client,
        query=query,
        patent=patent,
    )

    return {
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
        "synthesis": verdict,
    }


# =========================================================
# Phase 3 — complete end-to-end wrapper
# =========================================================

def run_patent_similarity(
    client,
    index_data,
    query,
    top_k=50,
    max_results=5,
    claims_batch_size=100,
    claims_delay=60.0,
    max_retries=5,
    max_workers=2,
):
    """
    Run the complete patent similarity pipeline.

    Pipeline:

        Raw query
            ↓
        Phase 2 Stage 1
            ↓
        FAISS abstract retrieval
            ↓
        top_k candidates
            ↓
        Phase 2 Stage 2
            ↓
        claims reranking
            ↓
        top max_results candidates
            ↓
        Phase 3 LLM synthesis
            ↓
        validated four-field verdicts

    Args:
        client:
            Gemini client.

        index_data:
            Output from build_indexes().

        query:
            Raw invention description.

        top_k:
            Number of Stage 1 candidates considered for Stage 2.

        max_results:
            Maximum number of Stage 2 candidates sent to Phase 3.

        claims_batch_size:
            Maximum claims embeddings per Gemini request.

        claims_delay:
            Seconds between successful Stage 2 embedding batches.

        max_retries:
            Maximum embedding retries.

    Returns:
        Dictionary containing:

            query
            stage1_candidates
            stage2_candidates
            results
    """

    if not isinstance(
        query,
        str,
    ) or not query.strip():

        raise ValueError(
            "query must be a non-empty string."
        )

    if max_results <= 0:
        raise ValueError(
            "max_results must be greater than 0."
        )

    # -----------------------------------------------------
    # Phase 2 — retrieval + reranking
    # -----------------------------------------------------

    retrieval = evaluate_query(
        client=client,
        index_data=index_data,
        query_text=query,
        top_k=top_k,
        claims_batch_size=claims_batch_size,
        claims_delay=claims_delay,
        max_retries=max_retries,
    )

    # -----------------------------------------------------
    # Select top Stage 2 candidates
    # -----------------------------------------------------

    selected_indices = select_full_treatment(
        retrieval["stage2_indices"],
        max_results=max_results,
    )

    # -----------------------------------------------------
    # Nothing to synthesize
    # -----------------------------------------------------

    if len(selected_indices) == 0:

        return {
            "query": query,
        "stage1_candidates": [
            {
                "patent_id": index_data["patents"][idx]["id"],
                "title": index_data["patents"][idx]["title"],
                "score": float(retrieval["abstract_scores"][idx]),
            }
            for idx in retrieval["stage1_indices"]
        ],

        "stage2_candidates": [
            {
                "patent_id": index_data["patents"][idx]["id"],
                "title": index_data["patents"][idx]["title"],
                "score": float(retrieval["final_scores"][idx]),
            }
            for idx in retrieval["stage2_indices"]
        ],
            "results": [],
        }

    # -----------------------------------------------------
    # Phase 3 — concurrent LLM synthesis
    # -----------------------------------------------------

    # Never create more workers than actual candidates.
    worker_count = min(
        max_workers,
        len(selected_indices),
    )

    results_by_rank = {}

    with ThreadPoolExecutor(
        max_workers=worker_count,
    ) as executor:

        future_to_rank = {}

        for rank, idx in enumerate(
            selected_indices,
            start=1,
        ):

            patent = index_data["patents"][idx]

            future = executor.submit(
                _synthesize_candidate,
                client,
                query,
                patent,
                rank,
                retrieval,
                idx,
            )

            future_to_rank[future] = rank

        # -------------------------------------------------
        # Collect results as workers finish.
        #
        # We store them by rank so the final output keeps
        # the original Phase 2 ranking order.
        # -------------------------------------------------

        for future in as_completed(
            future_to_rank
        ):

            rank = future_to_rank[future]

            try:
                result = future.result()
            except Exception as error:
                patent = index_data["patents"][selected_indices[rank - 1]]
                result = {
                    "rank": rank,
                    "patent_id": patent["id"],
                    "title": patent["title"],
                    "error": str(error),
                }

            results_by_rank[rank] = result

    # -----------------------------------------------------
    # Restore Phase 2 ranking order
    # -----------------------------------------------------

    results = [
        results_by_rank[rank]
        for rank in sorted(
            results_by_rank
        )
    ]

    # -----------------------------------------------------
    # Return combined result
    # -----------------------------------------------------

    return {
        "query": query,

        "stage1_candidates": [
            {
                "patent_id": index_data["patents"][idx]["id"],
                "title": index_data["patents"][idx]["title"],
                "score": float(retrieval["abstract_scores"][idx]),
            }
            for idx in retrieval["stage1_indices"]
        ],

        "stage2_candidates": [
            {
                "patent_id": index_data["patents"][idx]["id"],
                "title": index_data["patents"][idx]["title"],
                "score": float(retrieval["final_scores"][idx]),
            }
            for idx in retrieval["stage2_indices"]
        ],

        "results": results,
    }


# =========================================================
# Standalone Phase 3 test
# =========================================================

def run_test(
    patent,
    query,
):
    """
    Standalone test for the Phase 3 synthesis prompt.
    """

    client = get_client()

    result = synthesize_patent(
        client=client,
        query=query,
        patent=patent,
    )

    print(
        "\n===== VALIDATED OUTPUT ====="
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        "\n✓ JSON is valid"
    )

    print(
        "✓ Verdict is one of the four allowed values"
    )

    print(
        "✓ Evidence source is valid"
    )

    print(
        "✓ Evidence quote is an exact contiguous substring"
    )


# =========================================================
# Standalone prompt tests
# =========================================================

if __name__ == "__main__":

    # P001 — clearly relevant
    p001 = next(
        p
        for p in TOY_PATENTS
        if p["id"] == "P001"
    )

    run_test(
        p001,
        "A deep learning system receives digital images "
        "and uses multiple convolutional layers to "
        "classify each image into a predefined category.",
    )

    # P004 — deliberately different domain
    p004 = next(
        p
        for p in TOY_PATENTS
        if p["id"] == "P004"
    )

    run_test(
        p004,
        "A camera-based deep learning system detects "
        "pedestrians and vehicles for autonomous "
        "vehicle operation.",
    )

    # P002 — ambiguous / partial technical overlap
    p002 = next(
        p
        for p in TOY_PATENTS
        if p["id"] == "P002"
    )

    run_test(
        p002,
        "A deep learning system analyzes camera images "
        "to identify and classify objects in real time.",
    )

    # P006 — insufficient evidence
    p006 = {
        "id": "TOY-INSUFFICIENT-EVIDENCE",
        "title": "Generic Data Processing System",
        "abstract": (
            "Systems and methods for processing data using "
            "computational techniques are provided. The "
            "system may receive data, process the data, "
            "and provide an output."
        ),
        "claims": (
            "A system comprising a processor configured "
            "to receive data, process the data using "
            "computational techniques, and provide an "
            "output."
        ),
    }

    run_test(
        p006,
        "A computer vision system uses a convolutional "
        "neural network to detect pedestrians and "
        "vehicles in camera images for autonomous "
        "vehicle operation.",
    )
