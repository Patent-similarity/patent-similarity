import json

from data.toy_patents import TOY_PATENTS
from src.phase2_embedding_retrieval.embedding_pipeline import get_client


VERDICTS = {
    "HIGH_RELEVANCE",
    "POSSIBLE_RELEVANCE",
    "LOW_RELEVANCE",
    "INSUFFICIENT_EVIDENCE",
}


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


def validate_output(result, patent):
    assert isinstance(result, dict), "Output is not a JSON object."

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

    assert isinstance(evidence, dict)
    assert set(evidence.keys()) == {"quote", "source"}

    assert evidence["source"] in {"claim", "abstract"}

    quote = evidence["quote"]

    assert isinstance(quote, str)
    assert quote.strip(), "Evidence quote is empty."

    if evidence["source"] == "claim":
        source_text = patent["claims"]
    else:
        source_text = patent["abstract"]

    assert source_text is not None

    assert quote in source_text, (
        f"Evidence quote is NOT a contiguous substring of the "
        f"{evidence['source']} text."
    )

    return True


def run_test(patent, query):
    client = get_client()

    prompt = build_prompt(query, patent)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
        },
    )

    raw = response.text

    print("\n===== RAW MODEL OUTPUT =====")
    print(raw)

    result = json.loads(raw)

    validate_output(result, patent)

    print("\n===== VALIDATED OUTPUT =====")
    print(json.dumps(result, indent=2))

    print("\n✓ JSON is valid")
    print("✓ Verdict is one of the four allowed values")
    print("✓ Evidence source is valid")
    print("✓ Evidence quote is an exact contiguous substring")


# P001 — clearly relevant
p001 = next(p for p in TOY_PATENTS if p["id"] == "P001")

run_test(
    p001,
    "A deep learning system receives digital images and uses multiple "
    "convolutional layers to classify each image into a predefined category.",
)


# P004 — deliberately different domain
p004 = next(p for p in TOY_PATENTS if p["id"] == "P004")

run_test(
    p004,
    "A camera-based deep learning system detects pedestrians and vehicles "
    "for autonomous vehicle operation.",
)

# P002 — ambiguous / partial technical overlap
p002 = next(p for p in TOY_PATENTS if p["id"] == "P002")

run_test(
    p002,
    "A deep learning system analyzes camera images to identify and "
    "classify objects in real time.",
)

p006 = {
    "id": "TOY-INSUFFICIENT-EVIDENCE",
    "title": "Generic Data Processing System",
    "abstract": (
        "Systems and methods for processing data using computational "
        "techniques are provided. The system may receive data, process "
        "the data, and provide an output."
    ),
    "claims": (
        "A system comprising a processor configured to receive data, "
        "process the data using computational techniques, and provide "
        "an output."
    ),
}

run_test(
    p006,
    "A computer vision system uses a convolutional neural network "
    "to detect pedestrians and vehicles in camera images for autonomous "
    "vehicle operation.",
)