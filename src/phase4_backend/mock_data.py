from models import SearchResponse, FullTreatmentResult, BasicMatch, SupportingEvidence


def get_mock_search_response(query_text: str, top_k: int) -> SearchResponse:
    """
    Placeholder standing in for real Phase 2 (retrieval) + Phase 3 (synthesis)
    output. Shape matches the partner's confirmed real schema: top 5 results
    get full synthesis treatment, ranks 6-50 get score-only basic matches.

    TODO: replace this function call in main.py with the real Phase 2/3
    pipeline once the partner's endpoint is exposed for integration.
    """
    full_treatment = [
        FullTreatmentResult(
            patent_id="8928658",
            title="Photon mapping using kd-trees for real-time rendering",
            score=0.91,
            verdict="POSSIBLE_RELEVANCE",
            overlap_summary="Both systems utilize neural networks to analyze visual image data and identify specific targets or features.",
            key_difference="The user query focuses on real-time object identification in camera images, whereas the candidate patent is limited to medical imaging for tumor detection.",
            supporting_evidence=SupportingEvidence(
                quote="extracting visual features using a neural network",
                source="claim"
            )
        ),
        FullTreatmentResult(
            patent_id="8933933",
            title="Early Z-mode rendering pipeline for GPU optimization",
            score=0.87,
            verdict="HIGH_RELEVANCE",
            overlap_summary="Both describe GPU-based rendering pipelines optimized for real-time depth handling.",
            key_difference="The candidate patent is specific to Z-buffer ordering, while the query describes a more general rendering acceleration technique.",
            supporting_evidence=SupportingEvidence(
                quote="performing early depth testing prior to fragment shading",
                source="claim"
            )
        ),
    ]

    additional = [
        BasicMatch(
            patent_id="8934666",
            title="Object and scene segmentation via class boundary detection",
            score=0.62
        ),
        BasicMatch(
            patent_id="8929621",
            title="Segmentation and surface matching",
            score=0.55
        ),
    ]

    return SearchResponse(
        query_text=query_text,
        full_treatment_results=full_treatment[:min(top_k, 5)],
        additional_matches=additional
    )
