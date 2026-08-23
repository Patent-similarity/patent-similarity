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


def build_indexes(
    client,
    patents,
    batch_size=50,
    delay=31.0,
    max_retries=5,
):
    """Build Stage 1 abstract embeddings and FAISS index using batches.

    Important:
    Although multiple texts can be sent in one embed_content call,
    the embedding quota may count each text toward the RPM limit.

    With a 100 RPM quota, a conservative strategy is:
        - 50 embeddings per batch
        - approximately 31 seconds between batches

    This keeps the embedding workload below the per-minute limit while
    still reducing Python/API overhead through batched requests.

    Args:
        client: Gemini client from get_client().
        patents: List of patent dictionaries.
        batch_size: Number of abstracts per embedding batch.
        delay: Seconds to wait between successful batches.
        max_retries: Maximum retries for a failed batch.

    Returns:
        dict containing:
            patents
            abstract_embeddings
            abstract_index
    """

    abstract_embeddings = []

    total_patents = len(patents)

    for start in range(0, total_patents, batch_size):

        end = min(start + batch_size, total_patents)

        batch = patents[start:end]

        batch_texts = [
            patent_to_text(patent)
            for patent in batch
        ]

        batch_number = start // batch_size + 1

        print()
        print(
            f"Embedding batch {batch_number}: "
            f"patents {start + 1}-{end} of {total_patents}"
        )

        # ---------------------------------------------
        # Embed batch with retry / backoff
        # ---------------------------------------------

        success = False

        for attempt in range(max_retries + 1):

            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch_texts,
                )

                batch_embeddings = [
                    np.array(
                        embedding.values,
                        dtype="float32",
                    )
                    for embedding in response.embeddings
                ]

                # -------------------------------------
                # Safety check: one embedding per input
                # -------------------------------------

                if len(batch_embeddings) != len(batch):

                    raise ValueError(
                        f"Embedding count mismatch: "
                        f"expected {len(batch)}, "
                        f"got {len(batch_embeddings)}."
                    )

                # Preserve verified input order.
                abstract_embeddings.extend(
                    batch_embeddings
                )

                print(
                    f"Completed batch {batch_number} "
                    f"({len(batch_embeddings)} embeddings)"
                )

                success = True

                break

            except Exception as error:

                if attempt == max_retries:

                    raise RuntimeError(
                        f"Failed to embed patents "
                        f"{start + 1}-{end} "
                        f"after {max_retries} retries."
                    ) from error

                # Exponential backoff.
                backoff = min(
                    2 ** attempt,
                    60,
                )

                print(
                    f"Batch {batch_number} failed:"
                )

                print(error)

                print(
                    f"Retrying in {backoff} seconds..."
                )

                time.sleep(backoff)

        if not success:

            raise RuntimeError(
                f"Batch {batch_number} did not complete."
            )

        # ---------------------------------------------
        # Inter-batch pacing
        # ---------------------------------------------

        if end < total_patents:

            print(
                f"Waiting {delay} seconds "
                f"before the next batch..."
            )

            time.sleep(delay)

    # ---------------------------------------------
    # Convert list to NumPy array
    # ---------------------------------------------

    abstract_embeddings = np.array(
        abstract_embeddings,
        dtype="float32",
    )

    # ---------------------------------------------
    # Final embedding count check
    # ---------------------------------------------

    if len(abstract_embeddings) != len(patents):

        raise ValueError(
            f"Final embedding count mismatch: "
            f"{len(abstract_embeddings)} embeddings "
            f"for {len(patents)} patents."
        )

    # ---------------------------------------------
    # Normalize for cosine similarity
    # ---------------------------------------------

    faiss.normalize_L2(
        abstract_embeddings
    )

    # ---------------------------------------------
    # Build FAISS IndexFlatIP
    # ---------------------------------------------

    dimension = abstract_embeddings.shape[1]

    abstract_index = faiss.IndexFlatIP(
        dimension
    )

    abstract_index.add(
        abstract_embeddings
    )

    # ---------------------------------------------
    # Final FAISS consistency check
    # ---------------------------------------------

    if abstract_index.ntotal != len(patents):

        raise ValueError(
            f"FAISS index mismatch: "
            f"{abstract_index.ntotal} vectors "
            f"for {len(patents)} patents."
        )

    print()
    print("=" * 50)
    print("INDEX BUILD COMPLETE")
    print("=" * 50)

    print(f"Patents:       {len(patents)}")

    print(
        f"Embeddings:    "
        f"{len(abstract_embeddings)}"
    )

    print(
        f"FAISS vectors: "
        f"{abstract_index.ntotal}"
    )

    print(
        f"Vector dimension: "
        f"{dimension}"
    )

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