"""
Phase 4 - FastAPI application layer.

Architecture:
- run_patent_similarity() is SYNCHRONOUS and may take 75-130s.
- POST /search does NOT block for the full pipeline duration.
- Gemini client and FAISS index are initialized ONCE at startup.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    SearchRequest,
    SearchJobCreated,
    JobStatusResponse,
    JobStatus,
    PipelineResult,
)

from . import job_store
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make project root importable
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# Required libraries
# ---------------------------------------------------------
import faiss
import numpy as np

from data.real_corpus import load_real_corpus
from src.phase2_embedding_retrieval.embedding_pipeline import get_client
from src.phase3_llm_synthesis.phase3 import run_patent_similarity


# ---------------------------------------------------------
# REAL FAISS FILES
# These files are in the project root:
#
# patent-similarity/
# ├── patent_similarity.faiss
# └── patent_similarity_metadata.json
# ---------------------------------------------------------
FAISS_INDEX_PATH = PROJECT_ROOT / "embeddings" / "faiss_index" / "patent_similarity.faiss"
METADATA_PATH = PROJECT_ROOT / "embeddings" / "faiss_index" / "patent_similarity_metadata.json"


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------
app = FastAPI(title="Patent Similarity Search API")


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),  # set ALLOWED_ORIGINS env var (comma-separated) before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Global resources
# Initialized ONCE during application startup
# ---------------------------------------------------------
gemini_client = None
phase2_index = None


# ---------------------------------------------------------
# Startup
# Load:
# 1. Gemini client
# 2. Real FAISS index
# 3. Metadata
# 4. Full patent corpus
# ---------------------------------------------------------
@app.on_event("startup")
def startup_event():
    global gemini_client, phase2_index

    # -----------------------------------------------------
    # Check whether real FAISS files exist
    # -----------------------------------------------------
    if not FAISS_INDEX_PATH.exists():
        print(
            f"ERROR: FAISS index not found at:\n"
            f"{FAISS_INDEX_PATH}"
        )
        return

    if not METADATA_PATH.exists():
        print(
            f"ERROR: Metadata file not found at:\n"
            f"{METADATA_PATH}"
        )
        return

    # -----------------------------------------------------
    # Load FAISS index
    # -----------------------------------------------------
    print("=" * 60)
    print("Loading REAL FAISS index...")
    print(f"Path: {FAISS_INDEX_PATH}")

    abstract_index = faiss.read_index(str(FAISS_INDEX_PATH))

    print(
        f"FAISS index loaded successfully: "
        f"{abstract_index.ntotal} vectors, "
        f"dimension {abstract_index.d}"
    )

    # -----------------------------------------------------
    # Verify expected index dimensions
    # -----------------------------------------------------
    if abstract_index.ntotal != 6779:
        print(
            f"WARNING: Expected 6779 vectors, "
            f"but found {abstract_index.ntotal}"
        )

    if abstract_index.d != 3072:
        print(
            f"WARNING: Expected dimension 3072, "
            f"but found {abstract_index.d}"
        )

    # -----------------------------------------------------
    # Load metadata
    # -----------------------------------------------------
    print("=" * 60)
    print("Loading FAISS metadata...")
    print(f"Path: {METADATA_PATH}")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(
        f"Metadata loaded successfully: "
        f"{len(metadata)} patents"
    )

    # -----------------------------------------------------
    # Verify metadata/index alignment
    # -----------------------------------------------------
    if len(metadata) != abstract_index.ntotal:
        raise RuntimeError(
            f"FAISS/metadata mismatch: "
            f"FAISS has {abstract_index.ntotal} vectors, "
            f"but metadata has {len(metadata)} entries."
        )

    print(
        "FAISS and metadata alignment verified: "
        f"{len(metadata)} entries"
    )

    # -----------------------------------------------------
    # Load full corpus
    #
    # Metadata only contains:
    # patent_id + title
    #
    # Phase 3 requires:
    # abstract + claims
    # -----------------------------------------------------
    print("=" * 60)
    print("Loading full patent corpus...")

    full_corpus = load_real_corpus()

    print(
        f"Full corpus loaded: "
        f"{len(full_corpus)} patents"
    )

    # -----------------------------------------------------
    # Create lookup dictionary
    # -----------------------------------------------------
    corpus_by_id = {
        p["id"]: p
        for p in full_corpus
    }

    # -----------------------------------------------------
    # Reconstruct patents in EXACT FAISS row order
    # -----------------------------------------------------
    patents = []
    missing = []

    for entry in metadata:

        pid = str(entry["patent_id"])

        if pid in corpus_by_id:

            patents.append(
                corpus_by_id[pid]
            )

        else:

            missing.append(pid)

            patents.append(
                {
                    "id": pid,
                    "title": entry.get("title", ""),
                    "abstract": "",
                    "claims": None,
                }
            )

    # -----------------------------------------------------
    # Warn if some patents are missing
    # -----------------------------------------------------
    if missing:
        print(
            f"WARNING: {len(missing)} patent IDs "
            f"from metadata were not found in the "
            f"current corpus."
        )

        print(
            f"First missing IDs: {missing[:5]}"
        )

    # -----------------------------------------------------
    # Build Phase 2 index data structure
    # -----------------------------------------------------
    phase2_index = {
        "patents": patents,
        "abstract_index": abstract_index,
    }

    # -----------------------------------------------------
    # Initialize Gemini client
    # -----------------------------------------------------
    print("=" * 60)
    print("Initializing Gemini client...")

    gemini_client = get_client()

    # -----------------------------------------------------
    # Final startup verification
    # -----------------------------------------------------
    print("=" * 60)
    print("STARTUP COMPLETE")
    print(
        f"FAISS vectors : {abstract_index.ntotal}"
    )
    print(
        f"Vector dimension : {abstract_index.d}"
    )
    print(
        f"Metadata entries : {len(metadata)}"
    )
    print(
        f"Patent records : {len(patents)}"
    )
    print("=" * 60)


# ---------------------------------------------------------
# Background worker
# ---------------------------------------------------------
def run_search_job(job_id: str, query: str):

    job_store.mark_running(job_id)

    # -----------------------------------------------------
    # Make sure startup completed successfully
    # -----------------------------------------------------
    if phase2_index is None or gemini_client is None:

        job_store.mark_failed(
            job_id,
            "Search index not loaded. "
            "Check that the real FAISS index and metadata "
            "exist and that startup completed successfully."
        )

        return

    try:

        # -------------------------------------------------
        # Run the actual patent similarity pipeline
        # -------------------------------------------------
        raw_result = run_patent_similarity(
            client=gemini_client,
            index_data=phase2_index,
            query=query,
        )

        # -------------------------------------------------
        # Validate pipeline result
        # -------------------------------------------------
        result = PipelineResult(
            **raw_result
        )

        # -------------------------------------------------
        # Store successful result
        # -------------------------------------------------
        job_store.mark_complete(
            job_id,
            result
        )

    except ValueError as e:

        # Pipeline-level validation/configuration error
        job_store.mark_failed(
            job_id,
            str(e)
        )

    except Exception as e:

        # Unexpected pipeline error
        job_store.mark_failed(
            job_id,
            str(e)
        )


# ---------------------------------------------------------
# POST /search
#
# Creates a job and immediately returns job ID.
# The expensive pipeline runs in the background.
# ---------------------------------------------------------
@app.post(
    "/search",
    response_model=SearchJobCreated
)
def create_search(
    request: SearchRequest,
    background_tasks: BackgroundTasks
):

    # -----------------------------------------------------
    # Validate query
    # -----------------------------------------------------
    if not request.query or not request.query.strip():

        raise HTTPException(
            status_code=422,
            detail="Query must not be empty."
        )

    # -----------------------------------------------------
    # Create job
    # -----------------------------------------------------
    job_id = job_store.create_job()

    # -----------------------------------------------------
    # Run pipeline in background
    # -----------------------------------------------------
    background_tasks.add_task(
        run_search_job,
        job_id,
        request.query
    )

    # -----------------------------------------------------
    # Return immediately
    # -----------------------------------------------------
    return SearchJobCreated(
        job_id=job_id,
        status=JobStatus.pending
    )


# ---------------------------------------------------------
# GET /search/{job_id}
#
# Returns current job status/result.
# Does NOT run the pipeline again.
# ---------------------------------------------------------
@app.get(
    "/search/{job_id}",
    response_model=JobStatusResponse
)
def get_search_status(job_id: str):

    job = job_store.get_job(job_id)

    if job is None:

        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    return job


# ---------------------------------------------------------
# GET /health
# ---------------------------------------------------------
@app.get("/health")
def health():

    return {
        "status": "ok"
    }
