from models import SearchResponse, PatentMatch


def get_mock_search_response(query_text: str, top_k: int) -> SearchResponse:
    """
    Placeholder standing in for real Phase 2 (retrieval) + Phase 3 (synthesis)
    output. Returns fake but realistically-shaped data so Phase 4 can be built
    and tested independently of Phase 2/3's completion.

    TODO: replace this function call in main.py with the real Phase 2/3
    pipeline once the partner's interface is finalized.
    """
    fake_results = [
        PatentMatch(
            patent_id="8928658",
            title="Photon mapping using kd-trees for real-time rendering",
            similarity_score=0.91,
            abstract_snippet="A method for accelerating photon mapping using spatial kd-tree structures..."
        ),
        PatentMatch(
            patent_id="8933933",
            title="Early Z-mode rendering pipeline for GPU optimization",
            similarity_score=0.87,
            abstract_snippet="A GPU rendering pipeline that performs early depth testing to reduce..."
        ),
        PatentMatch(
            patent_id="8934666",
            title="Object and scene segmentation via class boundary detection",
            similarity_score=0.79,
            abstract_snippet="A computer vision method for segmenting objects within a scene using..."
        ),
    ]

    return SearchResponse(
        query_text=query_text,
        results=fake_results[:top_k],
        verdict="Potentially similar prior art found",
        verdict_category="similar_prior_art",
        explanation="Mock explanation: this is placeholder data, not a real Phase 3 synthesis result."
    )
