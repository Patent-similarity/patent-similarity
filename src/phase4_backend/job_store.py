"""
In-memory job store. Adequate for this project's scope -- not persistent,
not multi-process-safe. If this needs to survive server restarts or scale
across workers later, swap for Redis/a database, but the interface
(create/get/update) can stay the same.
"""

import uuid
from typing import Dict, Optional
from models import JobStatus, JobStatusResponse, PipelineResult

_jobs: Dict[str, JobStatusResponse] = {}


def create_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobStatusResponse(job_id=job_id, status=JobStatus.pending)
    return job_id


def get_job(job_id: str) -> Optional[JobStatusResponse]:
    return _jobs.get(job_id)


def mark_running(job_id: str) -> None:
    _jobs[job_id].status = JobStatus.running


def mark_complete(job_id: str, result: PipelineResult) -> None:
    _jobs[job_id].status = JobStatus.complete
    _jobs[job_id].result = result


def mark_failed(job_id: str, error: str) -> None:
    """
    Only call this for a PIPELINE-level failure (e.g. invalid query,
    invalid config -- the ValueError cases from partner's handoff).
    Individual candidate failures are NOT pipeline failures -- those are
    encoded inside PipelineResult.results as per-candidate `error` fields,
    and the job should still be marked complete, not failed.
    """
    _jobs[job_id].status = JobStatus.failed
    _jobs[job_id].error = error