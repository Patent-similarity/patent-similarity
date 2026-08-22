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


def build_indexes(client, patents):
    """Build Stage 1 abstract embeddings and Stage 2 claim embeddings.

    Patents with missing claims receive no claim embedding.
    They remain eligible for Stage 1 but are excluded from Stage 2.
    """

    # ---------------------------------------------------------
    # Stage 1: embed every patent's abstract
    # ---------------------------------------------------------

    abstract_texts = [
        patent_to_text(p)
        for p in patents
    ]

    abstract_embeddings = np.array(
        [
            get_embedding(client, text)
            for text in abstract_texts
        ],
        dtype="float32",
    )

    # Normalize so inner product behaves like cosine similarity.
    faiss.normalize_L2(abstract_embeddings)

    # ---------------------------------------------------------
    # Stage 2: embed claims only when claims are available
    # ---------------------------------------------------------

    claim_embeddings = []

    for patent in patents:

        if patent["claims"] is None:
            # No claims = no Stage 2 embedding.
            # We deliberately do NOT call get_embedding(None).
            claim_embeddings.append(None)

        else:
            embedding = get_embedding(
                client,
                patent["claims"],
            )

            embedding = embedding.reshape(1, -1)

            faiss.normalize_L2(embedding)

            claim_embeddings.append(
                embedding[0]
            )

    # ---------------------------------------------------------
    # Build Stage 1 FAISS index
    # ---------------------------------------------------------

    dimension = abstract_embeddings.shape[1]

    abstract_index = faiss.IndexFlatIP(dimension)

    abstract_index.add(
        abstract_embeddings
    )

    return {
        "patents": patents,
        "abstract_embeddings": abstract_embeddings,
        "claim_embeddings": claim_embeddings,
        "abstract_index": abstract_index,
    }


def evaluate_query(client, index_data, query_text):
    """Run one query through Stage 1 and Stage 2.

    Stage 1:
        All patents are ranked using abstract similarity.

    Stage 2:
        Only patents with claims are reranked using the
        weighted abstract + claims score.

    Claims-missing patents remain in Stage 1 but are excluded
    from Stage 2.
    """

    # ---------------------------------------------------------
    # Embed and normalize the query
    # ---------------------------------------------------------

    query_embedding = get_embedding(
        client,
        query_text,
    ).reshape(1, -1)

    faiss.normalize_L2(
        query_embedding
    )

    query_vector = query_embedding[0]

    # ---------------------------------------------------------
    # Stage 1: abstract-only ranking
    # ---------------------------------------------------------

    abstract_scores = np.dot(
        index_data["abstract_embeddings"],
        query_vector,
    )

    stage1_indices = np.argsort(
        abstract_scores
    )[::-1]

    # ---------------------------------------------------------
    # Stage 2: claims reranking
    # ---------------------------------------------------------

    # Only patents with claims embeddings are eligible.
    stage2_indices = [
        idx
        for idx, embedding in enumerate(
            index_data["claim_embeddings"]
        )
        if embedding is not None
    ]

    # Missing-claims patents get NaN because they have
    # no Stage 2 score.
    claim_scores = np.full(
        len(index_data["patents"]),
        np.nan,
        dtype="float32",
    )

    final_scores = np.full(
        len(index_data["patents"]),
        np.nan,
        dtype="float32",
    )

    # Calculate Stage 2 scores only for eligible patents.
    for idx in stage2_indices:

        claim_scores[idx] = np.dot(
            index_data["claim_embeddings"][idx],
            query_vector,
        )

        final_scores[idx] = (
            ABSTRACT_WEIGHT * abstract_scores[idx]
            + CLAIM_WEIGHT * claim_scores[idx]
        )

    # Sort only the Stage 2-eligible patents.
    stage2_indices = np.array(
        sorted(
            stage2_indices,
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