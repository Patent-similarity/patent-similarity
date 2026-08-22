"""
Phase 2 — two-stage patent similarity retrieval pipeline.

Stage 1: abstract-only embedding search (Gemini gemini-embedding-001 +
         FAISS IndexFlatIP, cosine via normalized inner product).
Stage 2: claims-embedding rerank on the Stage 1 candidates.
Final score = ABSTRACT_WEIGHT * abstract_score + CLAIM_WEIGHT * claim_score.

This module is corpus-agnostic: it operates on whatever list of patent
dicts (with "abstract" and "claims" fields) is passed in. Swapping the
toy corpus for the real Phase 1 G06T corpus does not require changes
here — only the data source passed into build_indexes() changes.
"""

import os

import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai

ABSTRACT_WEIGHT = 0.3
CLAIM_WEIGHT = 0.7


def get_client():
    """Load the Gemini API key from .env and return a configured client."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. "
            "Check that your .env file contains GEMINI_API_KEY=your_key."
        )

    return genai.Client(api_key=api_key)


def patent_to_text(patent):
    """Return the Stage 1 representation of a patent: abstract only.

    Deliberately abstract-only, per the locked two-stage design. An
    earlier version of this pipeline concatenated title+abstract+claims
    for Stage 1 by mistake; this function exists to make that decision
    explicit and prevent regressing it.
    """
    return patent["abstract"]


def get_embedding(client, text):
    """Return a Gemini embedding as a float32 NumPy vector."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return np.array(response.embeddings[0].values, dtype="float32")


def build_indexes(client, patents):
    """Build Stage 1 (abstract) FAISS index and Stage 2 claim embeddings.

    Args:
        client: Gemini client from get_client().
        patents: list of dicts, each with "abstract" and "claims" keys.

    Returns:
        dict with:
            "patents": the input list (order matches embedding rows)
            "abstract_embeddings": (N, D) normalized float32 array
            "claim_embeddings": (N, D) normalized float32 array
            "abstract_index": FAISS IndexFlatIP over abstract_embeddings
    """
    abstract_texts = [patent_to_text(p) for p in patents]
    abstract_embeddings = np.array(
        [get_embedding(client, text) for text in abstract_texts],
        dtype="float32",
    )
    faiss.normalize_L2(abstract_embeddings)

    claim_embeddings = np.array(
        [get_embedding(client, p["claims"]) for p in patents],
        dtype="float32",
    )
    faiss.normalize_L2(claim_embeddings)

    dimension = abstract_embeddings.shape[1]
    abstract_index = faiss.IndexFlatIP(dimension)
    abstract_index.add(abstract_embeddings)

    return {
        "patents": patents,
        "abstract_embeddings": abstract_embeddings,
        "claim_embeddings": claim_embeddings,
        "abstract_index": abstract_index,
    }


def evaluate_query(client, index_data, query_text):
    """Run one query through Stage 1 (abstract) and Stage 2 (claims rerank).

    Returns a dict of scores and rankings (indices into index_data["patents"]).
    """
    query_embedding = get_embedding(client, query_text).reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    abstract_scores = np.dot(
        index_data["abstract_embeddings"], query_embedding[0]
    )
    stage1_indices = np.argsort(abstract_scores)[::-1]

    claim_scores = np.dot(
        index_data["claim_embeddings"], query_embedding[0]
    )
    final_scores = (
        ABSTRACT_WEIGHT * abstract_scores + CLAIM_WEIGHT * claim_scores
    )
    stage2_indices = np.argsort(final_scores)[::-1]

    return {
        "abstract_scores": abstract_scores,
        "claim_scores": claim_scores,
        "final_scores": final_scores,
        "stage1_indices": stage1_indices,
        "stage2_indices": stage2_indices,
    }


def print_ranking(patents, indices, scores):
    """Print rankings in a consistent format for manual eyeball review."""
    for rank, idx in enumerate(indices, start=1):
        patent = patents[idx]
        print(f"Rank {rank}")
        print(f"Patent ID: {patent['id']}")
        print(f"Title: {patent['title']}")
        print(f"Score: {scores[idx]:.4f}")
        print("-" * 50)
