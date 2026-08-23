from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import SearchRequest, SearchResponse
from mock_data import get_mock_search_response

app = FastAPI(
    title="Patent Similarity Search API",
    description="Free-text patent similarity search backend (Phase 4).",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to actual frontend origin before production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Simple liveness check -- confirms the API is running."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search_patents(request: SearchRequest):
    """
    Accepts a free-text invention description and returns ranked similar
    patents with a synthesized verdict.

    Currently backed by mock data (see mock_data.py) since Phase 2/3
    (retrieval + synthesis) are not yet finalized by the partner. Swap
    get_mock_search_response() for the real pipeline call once ready.
    """
    return get_mock_search_response(
        query_text=request.query_text,
        top_k=request.top_k
    )
