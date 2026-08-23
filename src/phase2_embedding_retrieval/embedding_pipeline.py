"""
Phase 2 — two-stage patent similarity retrieval pipeline.

Stage 1: abstract-only embedding search (Gemini gemini-embedding-001 +
         FAISS IndexFlatIP, cosine via normalized inner product).
Stage 2: claims-embedding rerank on the Stage 1 candidates.

Patents with missing claims are excluded from Stage 2 entirely.
They remain available in the Stage 1 abstract-only ranking.

This module is corpus-agnostic: it operates on whatever list of patent
dicts (with "abstract" and "claims" fields) is passed in.
"""

import os
import time
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
    """Return the Stage 1 representation of a patent: abstract only."""
    return patent["abstract"]


def get_embedding(client, text):
    """Return a Gemini embedding as a float32 NumPy vector."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )

    return np.array(
        response.embeddings[0].values,
        dtype="float32",
    )


def build_indexes(client, patents, delay=0.7):
    """Build Stage 1 abstract embeddings and FAISS index.

    Claims are no longer embedded here. Claims embedding now happens
    inside evaluate_query, scoped to that query's top-K Stage 1
    candidates only — not the whole corpus upfront. This avoids
    wasting embedding calls on patents that never surface as a
    Stage 1 candidate for any given query.

    `delay` paces requests to stay under the free-tier RPM limit
    (100 requests/minute for gemini-embedding-001).

    Args:
        client: Gemini client from get_client().
        patents: list of dicts, each with "abstract" and "claims" keys.

    Returns:
        dict with:
            "patents": the input list
            "abstract_embeddings": normalized float32 array
            "abstract_index": FAISS IndexFlatIP over abstract embeddings
    """

    abstract_texts = [patent_to_text(p) for p in patents]

    abstract_embeddings = []
    for text in abstract_texts:
        abstract_embeddings.append(get_embedding(client, text))
        time.sleep(delay)

    abstract_embeddings = np.array(abstract_embeddings, dtype="float32")
    faiss.normalize_L2(abstract_embeddings)

    dimension = abstract_embeddings.shape[1]

    abstract_index = faiss.IndexFlatIP(dimension)
    abstract_index.add(abstract_embeddings)

    return {
        "patents": patents,
        "abstract_embeddings": abstract_embeddings,
        "abstract_index": abstract_index,
    }


def evaluate_query(client, index_data, query_text, top_k=50, delay=0.7):
    """Run one query through Stage 1 and Stage 2.

    Stage 1:
        Every patent is ranked by abstract similarity. Only the
        top_k best are kept — this is the capped list the caller
        uses for the Stage-1-only "lighter treatment" tier.

    Stage 2:
        From that same top_k, patents with claims=None are dropped.
        Claims are embedded only for the survivors, on the fly, then
        reranked by the weighted abstract+claims score. This is the
        only place claims ever get embedded — never for the whole
        corpus, never for candidates outside the top_k.

    Claims-missing candidates never enter Stage 2, and Stage 2 is
    never backfilled from Stage 1 to hit any particular count.
    """

    patents = index_data["patents"]

    # ---------------------------------------------------------
    # Embed and normalize the query
    # ---------------------------------------------------------

    query_embedding = get_embedding(client, query_text).reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    query_vector = query_embedding[0]

    # ---------------------------------------------------------
    # Stage 1: abstract-only ranking, capped at top_k
    # ---------------------------------------------------------

    abstract_scores = np.dot(
        index_data["abstract_embeddings"],
        query_vector,
    )

    stage1_indices = np.argsort(abstract_scores)[::-1][:top_k]

    # ---------------------------------------------------------
    # Stage 2: claims reranking, scoped to top_k survivors only
    # ---------------------------------------------------------

    # Only patents in the top_k AND with claims available are eligible.
    stage2_candidates = [
        idx for idx in stage1_indices
        if patents[idx]["claims"] is not None
    ]

    claim_scores = np.full(len(patents), np.nan, dtype="float32")
    final_scores = np.full(len(patents), np.nan, dtype="float32")

    for idx in stage2_candidates:
        claim_embedding = get_embedding(client, patents[idx]["claims"])
        time.sleep(delay)

        claim_embedding = claim_embedding.reshape(1, -1)
        faiss.normalize_L2(claim_embedding)

        claim_scores[idx] = np.dot(claim_embedding[0], query_vector)

        final_scores[idx] = (
        ABSTRACT_WEIGHT * abstract_scores[idx]
        + CLAIM_WEIGHT * claim_scores[idx]
        )

    stage2_indices = np.array(
        sorted(
            stage2_candidates,
            key=lambda idx: final_scores[idx],
            reverse=True,
        )
    )

    return {
        "abstract_scores": abstract_scores,
        "claim_scores": claim_scores,
        "final_scores": final_scores,
        "stage1_indices": stage1_indices,
        "stage2_indices": stage2_indices,
    }


def select_full_treatment(stage2_indices, max_results=5):
    """Select up to 5 Stage 2-eligible patents.

    Stage 1 candidates are never used to fill missing slots.
    """

    return stage2_indices[:max_results]


def print_ranking(patents, indices, scores):
    """Print rankings in a consistent format for manual review."""

    for rank, idx in enumerate(
        indices,
        start=1,
    ):
        patent = patents[idx]

        print(f"Rank {rank}")
        print(f"Patent ID: {patent['id']}")
        print(f"Title: {patent['title']}")
        print(f"Score: {scores[idx]:.4f}")
        print("-" * 50)