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


class PatentMatch(BaseModel):
    """A single retrieved patent result."""
    patent_id: str
    title: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    abstract_snippet: str


class SearchResponse(BaseModel):
    """What we send back: ranked matches + a synthesized verdict."""
    query_text: str
    results: list[PatentMatch]
    verdict: str
    verdict_category: Literal["novel", "similar_prior_art", "likely_infringing", "inconclusive"]
    explanation: str
