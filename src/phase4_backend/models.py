from pydantic import BaseModel, Field
from typing import Literal


class SearchRequest(BaseModel):
    """What the caller sends us: a free-text invention description."""
    query_text: str = Field(
        ...,
        min_length=10,
        description="Free-text description of the invention to search for similar patents."
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top matching patents to return (1-20)."
    )


class SupportingEvidence(BaseModel):
    """A verbatim excerpt backing a synthesis verdict."""
    quote: str = Field(..., description="Verbatim substring from the source patent.")
    source: Literal["claim", "abstract"]


class FullTreatmentResult(BaseModel):
    """
    A top-5, Stage-2-eligible result: has claims-match data, so gets
    the full synthesis treatment (verdict + reasoning + evidence).
    """
    patent_id: str
    title: str
    score: float = Field(..., ge=0.0, le=1.0)
    verdict: Literal[
        "HIGH_RELEVANCE",
        "POSSIBLE_RELEVANCE",
        "LOW_RELEVANCE",
        "INSUFFICIENT_EVIDENCE"
    ]
    overlap_summary: str
    key_difference: str
    supporting_evidence: SupportingEvidence


class BasicMatch(BaseModel):
    """
    A rank 6-50, Stage-1-only result: no claims match, so no synthesis --
    just the raw retrieval hit.
    """
    patent_id: str
    title: str
    score: float = Field(..., ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    """
    What we send back: up to 5 fully-synthesized results, plus up to 45
    additional lower-confidence matches with no synthesis.
    """
    query_text: str
    full_treatment_results: list[FullTreatmentResult]
    additional_matches: list[BasicMatch]
