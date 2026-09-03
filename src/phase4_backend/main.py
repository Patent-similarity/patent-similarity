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

from models import SearchRequest, SearchJobCreated, JobStatusResponse, JobStatus, PipelineResult
import job_store

import json
import sys
from pathlib import Path

# Repo root needs to be importable for phase2/phase3 modules and data.real_corpus
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import faiss
import numpy as np

from data.real_corpus import load_real_corpus
from src.phase2_embedding_retrieval.embedding_pipeline import get_client
from src.phase3_llm_synthesis.phase3 import run_patent_similarity

# Confirmed with partner -- exact paths + formats, see Phase 4 handoff discussion.
FAISS_INDEX_PATH = PROJECT_ROOT / "embeddings" / "faiss_index" / "patent_similarity.faiss"
METADATA_PATH = PROJECT_ROOT / "embeddings" / "faiss_index" / "patent_similarity_metadata.json"

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


# ---------------------------------------------------------------------
# Startup: initialize expensive, reusable resources ONCE.
# Per partner's handoff -- do NOT rebuild the FAISS index per request.
# ---------------------------------------------------------------------
gemini_client = None
phase2_index = None


@app.on_event("startup")
def startup_event():
    global gemini_client, phase2_index

    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        print(
            f"Startup: real FAISS index not found yet at {FAISS_INDEX_PATH} -- "
            f"partner's assembly step (see handoff) not complete. "
            f"Search will fail until these files exist."
        )
        return

    print(f"Loading FAISS index from {FAISS_INDEX_PATH} ...")
    abstract_index = faiss.read_index(str(FAISS_INDEX_PATH))
    print(f"FAISS index loaded: {abstract_index.ntotal} vectors, dim {abstract_index.d}")

    print(f"Loading metadata from {METADATA_PATH} ...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)  # [{"patent_id": "...", "title": "..."}, ...], FAISS-row-ordered
    print(f"Loaded metadata for {len(metadata)} patents")

    # The metadata JSON gives us patent_id/title in FAISS row order, but
    # run_patent_similarity() needs full records (abstract + claims) for
    # the Phase 3 synthesis prompt. Reconstruct the full patent list in
    # the SAME ORDER as the metadata -- this order must match the FAISS
    # index exactly, since stage1/stage2 indices are positional integers.
    print("Loading full corpus for abstract/claims text ...")
    full_corpus = load_real_corpus()
    corpus_by_id = {p["id"]: p for p in full_corpus}

    patents = []
    missing = []
    for entry in metadata:
        pid = str(entry["patent_id"])
        if pid in corpus_by_id:
            patents.append(corpus_by_id[pid])
        else:
            # Should not happen if metadata was built from the same
            # patents.db -- flagging loudly rather than silently
            # misaligning FAISS row positions.
            missing.append(pid)
            patents.append({"id": pid, "title": entry.get("title", ""), "abstract": "", "claims": None})

    if missing:
        print(f"WARNING: {len(missing)} patent_ids in metadata not found in current corpus: {missing[:5]}...")

    phase2_index = {
        "patents": patents,
        "abstract_index": abstract_index,
    }

    gemini_client = get_client()

    print(f"Startup complete: {len(patents)} patents indexed and ready.")


# ---------------------------------------------------------------------
# Background worker -- this is where the actual pipeline call happens.
# ---------------------------------------------------------------------
def run_search_job(job_id: str, query: str):
    job_store.mark_running(job_id)

    if phase2_index is None or gemini_client is None:
        job_store.mark_failed(
            job_id,
            "Search index not loaded -- real FAISS index/metadata not found "
            "at startup. Check partner's assembly step has completed."
        )
        return

    try:
        raw_result = run_patent_similarity(
            client=gemini_client,
            index_data=phase2_index,
            query=query,
        )
        result = PipelineResult(**raw_result)
        job_store.mark_complete(job_id, result)

        # Individual candidate failures are NOT pipeline failures -- they
        # arrive already encoded as per-candidate `error` fields inside
        # raw_result["results"], and are NOT caught here or turned into
        # a job-level failure. Only a raised exception from
        # run_patent_similarity() itself represents a pipeline-level
        # failure (caught below).

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