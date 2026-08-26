"""
Phase 4 models - matches the Phase 3 -> Phase 4 contract from partner's handoff.
"""

from pydantic import BaseModel
from typing import Optional, List, Union
from enum import Enum


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    complete = "complete"
    failed = "failed"


class SearchRequest(BaseModel):
    query: str


class SearchJobCreated(BaseModel):
    job_id: str
    status: JobStatus  # always "pending" at creation


class Candidate(BaseModel):
    """Stage 1 / Stage 2 candidate objects - real objects, not raw FAISS indices."""
    patent_id: str
    title: str
    score: float


class SupportingEvidence(BaseModel):
    quote: str
    source: str


class Synthesis(BaseModel):
    verdict: str  # HIGH_RELEVANCE | POSSIBLE_RELEVANCE | LOW_RELEVANCE | INSUFFICIENT_EVIDENCE
    overlap_summary: str
    key_difference: str
    supporting_evidence: SupportingEvidence


class ResultCandidate(BaseModel):
    """
    A single ranked result. Exactly one of `synthesis` or `error` is present,
    never both -- this is the individual-candidate success/failure contract.
    """
    rank: int
    patent_id: str
    title: str
    abstract_score: float
    claim_score: float
    final_score: float
    synthesis: Optional[Synthesis] = None
    error: Optional[str] = None


class PipelineResult(BaseModel):
    query: str
    stage1_candidates: List[Candidate]
    stage2_candidates: List[Candidate]
    results: List[ResultCandidate]


class JobStatusResponse(BaseModel):
    """
    Response shape for GET /search/{job_id}. `result` is only present when
    status == "complete". `error` is only present when status == "failed"
    (this is a PIPELINE-level failure, not an individual candidate failure --
    individual candidate failures live inside PipelineResult.results as
    per-candidate `error` fields and do NOT set this top-level status to failed).
    """
    job_id: str
    status: JobStatus
    result: Optional[PipelineResult] = None
    error: Optional[str] = None