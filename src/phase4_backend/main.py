"""
Phase 4 - FastAPI application layer.

Architecture, per partner's Phase 3 handoff:
- run_patent_similarity() is SYNCHRONOUS and takes 75-130s real-world.
- POST /search must NOT block the HTTP request for that duration.
- Gemini client and Phase 2 FAISS index are initialized ONCE at startup,
  not per-request.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from models import SearchRequest, SearchJobCreated, JobStatusResponse, JobStatus
import job_store

# TODO(integration): import the real Phase 3 entry point once wired up:
# from phase3_llm_synthesis.phase3 import run_patent_similarity
# from phase2_embedding_retrieval.pipeline import get_client, load_index

app = FastAPI(title="Patent Similarity Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Startup: initialize expensive, reusable resources ONCE.
# Per partner's handoff -- do NOT rebuild the FAISS index per request.
# ---------------------------------------------------------------------
gemini_client = None
phase2_index = None


@app.on_event("startup")
def startup_event():
    global gemini_client, phase2_index
    # TODO(integration): replace with real initialization once wired up:
    # gemini_client = get_client()  # reuses GEMINI_API_KEY per partner's existing get_client()
    # phase2_index = load_index()
    print("Startup: Gemini client and Phase 2 FAISS index would load here.")


# ---------------------------------------------------------------------
# Background worker -- this is where the actual pipeline call happens.
# ---------------------------------------------------------------------
def run_search_job(job_id: str, query: str):
    job_store.mark_running(job_id)
    try:
        # TODO(integration): replace with the real call once wired up:
        # raw_result = run_patent_similarity(
        #     client=gemini_client,
        #     index_data=phase2_index,
        #     query=query,
        # )
        # result = PipelineResult(**raw_result)
        #
        # Individual candidate failures are NOT pipeline failures -- they
        # arrive already encoded as per-candidate `error` fields inside
        # raw_result["results"], and should NOT be caught here or turned
        # into a job-level failure. Only a raised exception from
        # run_patent_similarity() itself (e.g. ValueError for empty query
        # or invalid config) represents a pipeline-level failure.

        raise NotImplementedError("Real Phase 3 call not wired up yet.")

    except ValueError as e:
        # Pipeline-level failure (e.g. empty query, invalid config) --
        # this is the ONLY case that should set the job to failed.
        job_store.mark_failed(job_id, str(e))
    except Exception as e:
        job_store.mark_failed(job_id, str(e))


# ---------------------------------------------------------------------
# POST /search -- create job, start background work, return immediately.
# ---------------------------------------------------------------------
@app.post("/search", response_model=SearchJobCreated)
def create_search(request: SearchRequest, background_tasks: BackgroundTasks):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    job_id = job_store.create_job()
    background_tasks.add_task(run_search_job, job_id, request.query)
    return SearchJobCreated(job_id=job_id, status=JobStatus.pending)


# ---------------------------------------------------------------------
# GET /search/{job_id} -- status/result lookup ONLY. Never re-runs the
# pipeline. Reads the stored job state and returns it as-is.
# ---------------------------------------------------------------------
@app.get("/search/{job_id}", response_model=JobStatusResponse)
def get_search_status(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/health")
def health():
    return {"status": "ok"}