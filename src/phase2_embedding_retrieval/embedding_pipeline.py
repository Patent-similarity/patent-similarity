"""
Phase 2 — two-stage patent similarity retrieval pipeline.

Stage 1:
    Abstract-only embedding search using Gemini gemini-embedding-001
    and FAISS IndexFlatIP.

Stage 2:
    Claims embeddings rerank the Stage 1 candidates using:

        final_score =
            ABSTRACT_WEIGHT * abstract_score
            +
            CLAIM_WEIGHT * claim_score

Patents with missing claims are excluded from Stage 2.

This module also contains conservative RPM/TPM rate limiting for
Gemini embedding requests.
"""

import os
import re
import time

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai


# =========================================================
# Configuration
# =========================================================

ABSTRACT_WEIGHT = 0.3
CLAIM_WEIGHT = 0.7

EMBEDDING_MODEL = "gemini-embedding-001"

# Gemini maximum number of inputs in one embedding request.
MAX_BATCH_SIZE = 100


# =========================================================
# Rate-limit configuration
# =========================================================
#
# API limits:
#
#     RPM = 100
#     TPM = 30,000
#
# We intentionally stay below those limits.
#
#     Safe RPM = 90
#     Safe TPM = 27,000
#
# We also keep individual requests much smaller than the
# minute budget. This is important because it lets multiple
# requests fit inside the same minute instead of making one
# huge request consume almost the entire TPM window.
#

RPM_LIMIT = 100
TPM_LIMIT = 30_000

SAFE_RPM_LIMIT = 90
SAFE_TPM_LIMIT = 27_000


# ---------------------------------------------------------
# Conservative token estimate
# ---------------------------------------------------------
#
# The previous 0.30 estimate was too optimistic.
#
# Your actual Gemini errors showed, for example:
#
#     34 claims -> approximately 34,452 tokens
#
# So we deliberately use 0.50 tokens/character.
#
# This is an estimate, not Google's tokenizer.
# The purpose is to stay safely below the real TPM limit.
#

TOKENS_PER_CHARACTER = 0.50


# ---------------------------------------------------------
# Maximum estimated tokens per individual request
# ---------------------------------------------------------
#
# 10,000 is deliberately conservative.
#
# It means roughly:
#
#     request 1 = ~10k
#     request 2 = ~10k
#     request 3 = ~7k
#
# can fit within one 27k safety window.
#
# The query embedding also consumes a small amount of TPM.
#

MAX_REQUEST_TOKENS = 10_000


# =========================================================
# Embedding rate limiter
# =========================================================

class EmbeddingRateLimiter:
    """
    Conservative rolling one-minute Gemini embedding limiter.

    Tracks:
        - requests
        - estimated input tokens

    Important:
        A request is only reserved after there is enough
        capacity for it.

    This prevents the old problem where the limiter allowed
    a request that itself was larger than the remaining
    TPM budget.
    """

    def __init__(
        self,
        rpm_limit=SAFE_RPM_LIMIT,
        tpm_limit=SAFE_TPM_LIMIT,
    ):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit

        self.window_start = time.monotonic()

        self.request_count = 0
        self.token_count = 0

    # -----------------------------------------------------
    # Token estimation
    # -----------------------------------------------------

    @staticmethod
    def estimate_tokens(texts):
        """
        Conservatively estimate input tokens.

        This deliberately overestimates rather than attempting
        to reproduce Google's tokenizer exactly.
        """

        if not texts:
            return 0

        total_characters = sum(
            len(text or "")
            for text in texts
        )

        estimated = int(
            total_characters * TOKENS_PER_CHARACTER
        )

        # At least one token per input.
        estimated = max(
            estimated,
            len(texts),
        )

        return estimated

    # -----------------------------------------------------
    # Reset minute window
    # -----------------------------------------------------

    def _reset_if_needed(self):
        elapsed = (
            time.monotonic()
            - self.window_start
        )

        if elapsed >= 60:
            self.window_start = time.monotonic()
            self.request_count = 0
            self.token_count = 0

            print()
            print("Rate-limit window reset.")

    # -----------------------------------------------------
    # Wait for capacity
    # -----------------------------------------------------

    def wait_for_capacity(
        self,
        estimated_tokens,
    ):
        """
        Wait until this request can safely be sent.

        Returns only after the request has been reserved.
        """

        if estimated_tokens <= 0:
            estimated_tokens = 1

        if estimated_tokens > self.tpm_limit:
            raise ValueError(
                f"Single request requires approximately "
                f"{estimated_tokens:,} tokens, which exceeds "
                f"the configured TPM limit of "
                f"{self.tpm_limit:,}."
            )

        while True:

            self._reset_if_needed()

            request_ok = (
                self.request_count + 1
                <= self.rpm_limit
            )

            token_ok = (
                self.token_count
                + estimated_tokens
                <= self.tpm_limit
            )

            if request_ok and token_ok:
                break

            elapsed = (
                time.monotonic()
                - self.window_start
            )

            remaining = max(
                0.5,
                60.0 - elapsed + 0.5,
            )

            reasons = []

            if not request_ok:
                reasons.append(
                    f"RPM "
                    f"{self.request_count + 1}/"
                    f"{self.rpm_limit}"
                )

            if not token_ok:
                reasons.append(
                    f"TPM "
                    f"{self.token_count + estimated_tokens:,}/"
                    f"{self.tpm_limit:,}"
                )

            print()
            print(
                f"Rate limiter waiting "
                f"{remaining:.1f}s "
                f"({', '.join(reasons)})"
            )

            time.sleep(remaining)

        # -------------------------------------------------
        # Reserve only after capacity is confirmed.
        # -------------------------------------------------

        self.request_count += 1
        self.token_count += estimated_tokens

        print(
            f"Rate usage: "
            f"requests="
            f"{self.request_count}/{self.rpm_limit}, "
            f"estimated_tokens="
            f"{self.token_count:,}/"
            f"{self.tpm_limit:,}"
        )


# One limiter for the entire Python process.
RATE_LIMITER = EmbeddingRateLimiter()


# =========================================================
# Gemini client
# =========================================================

def get_client():
    """Load Gemini API key and create a client."""

    load_dotenv()

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. "
            "Check that your .env file contains "
            "GEMINI_API_KEY=your_key."
        )

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# Patent text helpers
# =========================================================

def patent_to_text(patent):
    """Return abstract text used for Stage 1."""

    return patent["abstract"]


# =========================================================
# Retry helpers
# =========================================================

def extract_retry_delay(error):
    """
    Extract Google's suggested retry delay when available.

    Returns:
        float seconds, or None.
    """

    error_text = str(error)

    patterns = [
        r"retryDelay[^0-9]*([0-9]+(?:\.[0-9]+)?)s",
        r"retry in[^0-9]*([0-9]+(?:\.[0-9]+)?)s",
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            error_text,
            flags=re.IGNORECASE,
        )

        if matches:
            try:
                return float(
                    matches[-1]
                )
            except ValueError:
                pass

    return None


def is_rate_limit_error(error):
    """Return True for Gemini quota/rate-limit errors."""

    text = str(error).upper()

    return (
        "429" in text
        or "RESOURCE_EXHAUSTED" in text
        or "QUOTA" in text
        or "RATE LIMIT" in text
    )


# =========================================================
# Shared embedding helper
# =========================================================

def embed_texts_with_retry(
    client,
    texts,
    context_label,
    max_retries=5,
):
    """
    Embed a list of texts with conservative rate limiting.

    The request is checked against both:
        - RPM
        - TPM

    before it is sent.
    """

    if not texts:
        return []

    estimated_tokens = (
        RATE_LIMITER.estimate_tokens(texts)
    )

    if estimated_tokens > MAX_REQUEST_TOKENS:

        raise ValueError(
            f"{context_label} contains approximately "
            f"{estimated_tokens:,} estimated input tokens, "
            f"which exceeds the configured per-request "
            f"target of {MAX_REQUEST_TOKENS:,}. "
            f"The caller must split the batch first."
        )

    for attempt in range(
        max_retries + 1
    ):

        # -------------------------------------------------
        # Wait BEFORE sending.
        # -------------------------------------------------

        RATE_LIMITER.wait_for_capacity(
            estimated_tokens
        )

        try:

            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
            )

            embeddings = [
                np.array(
                    embedding.values,
                    dtype="float32",
                )
                for embedding in response.embeddings
            ]

            if len(embeddings) != len(texts):

                raise ValueError(
                    f"Embedding count mismatch for "
                    f"{context_label}: expected "
                    f"{len(texts)}, got "
                    f"{len(embeddings)}."
                )

            return embeddings

        except Exception as error:

            if attempt == max_retries:

                raise RuntimeError(
                    f"Failed to embed "
                    f"{context_label} after "
                    f"{max_retries} retries."
                ) from error

            server_delay = (
                extract_retry_delay(error)
            )

            if server_delay is not None:

                backoff = min(
                    max(server_delay, 1.0),
                    120.0,
                )

            else:

                backoff = min(
                    2 ** attempt,
                    60.0,
                )

            print()
            print(
                f"{context_label} embedding failed:"
            )
            print(error)

            print(
                f"Retrying in "
                f"{backoff:.1f}s..."
            )

            time.sleep(backoff)


# =========================================================
# Single embedding
# =========================================================

def get_embedding(
    client,
    text,
    max_retries=5,
):
    """Return one Gemini embedding."""

    embeddings = embed_texts_with_retry(
        client=client,
        texts=[text],
        context_label="single embedding",
        max_retries=max_retries,
    )

    return embeddings[0]


# =========================================================
# Token-safe batch splitting
# =========================================================

def split_texts_by_token_budget(
    texts,
    max_tokens=9800,
):
    """
    Split texts into batches whose whole-batch estimated
    token count stays below max_tokens.

    Texts are never split individually.

    The same whole-batch token estimation method is used here
    and later by embed_texts_with_retry(), preventing a batch
    from being accepted by the splitter but rejected by the
    embedding helper because of per-item rounding differences.

    Returns:
        List[List[str]]
    """

    batches = []
    current_batch = []

    for text in texts:

        text = text or ""

        # ---------------------------------------------
        # First check whether this individual text can
        # fit inside one request.
        # ---------------------------------------------

        single_text_tokens = (
            RATE_LIMITER.estimate_tokens(
                [text]
            )
        )

        if single_text_tokens > max_tokens:

            if current_batch:

                batches.append(
                    current_batch
                )

                current_batch = []

            raise ValueError(
                f"One text requires approximately "
                f"{single_text_tokens:,} estimated tokens, "
                f"which exceeds the per-request target "
                f"of {max_tokens:,}."
            )

        # ---------------------------------------------
        # Try adding this text to the current batch.
        #
        # IMPORTANT:
        # Estimate the COMPLETE candidate batch instead
        # of summing individually rounded estimates.
        # ---------------------------------------------

        candidate_batch = (
            current_batch + [text]
        )

        candidate_tokens = (
            RATE_LIMITER.estimate_tokens(
                candidate_batch
            )
        )

        if (
            current_batch
            and candidate_tokens > max_tokens
        ):

            # Current batch is complete.
            batches.append(
                current_batch
            )

            # Start a new batch with this text.
            current_batch = [
                text
            ]

        else:

            current_batch.append(
                text
            )

    # ---------------------------------------------
    # Add the final batch.
    # ---------------------------------------------

    if current_batch:

        batches.append(
            current_batch
        )

    return batches


# =========================================================
# Stage 1 — build abstract embeddings
# =========================================================

def build_indexes(
    client,
    patents,
    batch_size=MAX_BATCH_SIZE,
    delay=0.0,
    max_retries=5,
):
    """
    Build Stage 1 abstract embeddings and FAISS index.

    Requests are automatically constrained by the token budget.
    """

    if not patents:
        raise ValueError(
            "Cannot build an index from an empty patent list."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0."
        )

    batch_size = min(
        batch_size,
        MAX_BATCH_SIZE,
    )

    abstract_embeddings = []

    total_patents = len(patents)

    # -----------------------------------------------------
    # Process patents in normal batches first.
    # Then token-split each batch if necessary.
    # -----------------------------------------------------

    for start in range(
        0,
        total_patents,
        batch_size,
    ):

        end = min(
            start + batch_size,
            total_patents,
        )

        batch = patents[start:end]

        batch_texts = [
            patent_to_text(patent)
            for patent in batch
        ]

        safe_batches = (
            split_texts_by_token_budget(
                batch_texts
            )
        )

        print()
        print(
            f"Embedding patents "
            f"{start + 1}-{end} "
            f"of {total_patents}"
        )

        print(
            f"Split into "
            f"{len(safe_batches)} "
            f"token-safe request(s)"
        )

        for batch_number, safe_batch in enumerate(
            safe_batches,
            start=1,
        ):

            estimated = (
                RATE_LIMITER.estimate_tokens(
                    safe_batch
                )
            )

            print(
                f"Embedding batch "
                f"{batch_number}/"
                f"{len(safe_batches)}"
            )

            print(
                f"Estimated input tokens: "
                f"{estimated:,}"
            )

            embeddings = embed_texts_with_retry(
                client=client,
                texts=safe_batch,
                context_label=(
                    f"Stage 1 batch "
                    f"{batch_number}"
                ),
                max_retries=max_retries,
            )

            abstract_embeddings.extend(
                embeddings
            )

            print(
                f"Completed batch "
                f"{batch_number}/"
                f"{len(safe_batches)}"
            )

            if (
                delay > 0
                and batch_number < len(safe_batches)
            ):
                time.sleep(delay)

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    abstract_embeddings = np.array(
        abstract_embeddings,
        dtype="float32",
    )

    if len(abstract_embeddings) != len(patents):

        raise ValueError(
            "Final embedding count does not match "
            "number of patents."
        )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    faiss.normalize_L2(
        abstract_embeddings
    )

    # -----------------------------------------------------
    # FAISS
    # -----------------------------------------------------

    dimension = (
        abstract_embeddings.shape[1]
    )

    abstract_index = faiss.IndexFlatIP(
        dimension
    )

    abstract_index.add(
        abstract_embeddings
    )

    if abstract_index.ntotal != len(patents):

        raise ValueError(
            "FAISS index size does not match "
            "number of patents."
        )

    print()
    print(
        "Index build complete."
    )
    print(
        f"Patents: {len(patents)}"
    )
    print(
        f"Embeddings: "
        f"{len(abstract_embeddings)}"
    )
    print(
        f"FAISS vectors: "
        f"{abstract_index.ntotal}"
    )

    return {
        "patents": patents,
        "abstract_embeddings": abstract_embeddings,
        "abstract_index": abstract_index,
    }


# =========================================================
# Stage 1 + Stage 2 evaluation
# =========================================================

def evaluate_query(
    client,
    index_data,
    query_text,
    top_k=50,
    claims_batch_size=MAX_BATCH_SIZE,
    claims_delay=0.0,
    max_retries=5,
):
    """
    Run Stage 1 and Stage 2.

    Stage 1:
        Abstract similarity over the full FAISS index.

    Stage 2:
        Claim embeddings for Stage 1 candidates that have
        claims, followed by weighted reranking.
    """

    patents = index_data["patents"]

    if not patents:
        raise ValueError(
            "Cannot evaluate a query against "
            "an empty index."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    claims_batch_size = min(
        claims_batch_size,
        MAX_BATCH_SIZE,
    )

    # -----------------------------------------------------
    # Query embedding
    # -----------------------------------------------------

    query_embedding = get_embedding(
        client=client,
        text=query_text,
        max_retries=max_retries,
    ).reshape(1, -1)

    faiss.normalize_L2(
        query_embedding
    )

    # -----------------------------------------------------
    # Stage 1
    # -----------------------------------------------------

    top_k = min(
        top_k,
        len(patents),
    )

    stage1_scores, stage1_indices = (
        index_data["abstract_index"].search(
            query_embedding,
            top_k,
        )
    )

    stage1_scores = (
        stage1_scores[0]
    )

    stage1_indices = (
        stage1_indices[0]
    )

    # -----------------------------------------------------
    # Abstract scores
    # -----------------------------------------------------

    abstract_scores = np.full(
        len(patents),
        np.nan,
        dtype="float32",
    )

    abstract_scores[
        stage1_indices
    ] = stage1_scores

    # -----------------------------------------------------
    # Stage 2 candidates
    # -----------------------------------------------------

    stage2_candidates = [
        idx
        for idx in stage1_indices
        if patents[idx]["claims"] is not None
    ]

    claim_scores = np.full(
        len(patents),
        np.nan,
        dtype="float32",
    )

    final_scores = np.full(
        len(patents),
        np.nan,
        dtype="float32",
    )

    if stage2_candidates:

        claim_texts = [
            patents[idx]["claims"]
            for idx in stage2_candidates
        ]

        # -------------------------------------------------
        # Token-safe claim batches
        # -------------------------------------------------

        safe_batches = (
            split_texts_by_token_budget(
                claim_texts
            )
        )

        print(
            f"Stage 2: "
            f"{len(claim_texts)} claims split into "
            f"{len(safe_batches)} "
            f"token-safe request(s)"
        )

        all_claim_embeddings = []

        processed_claims = 0

        for batch_number, claim_batch in enumerate(
            safe_batches,
            start=1,
        ):

            batch_size_actual = (
                len(claim_batch)
            )

            batch_start = (
                processed_claims + 1
            )

            batch_end = (
                processed_claims
                + batch_size_actual
            )

            estimated = (
                RATE_LIMITER.estimate_tokens(
                    claim_batch
                )
            )

            print()
            print(
                f"Embedding Stage 2 claims "
                f"{batch_start}-{batch_end} "
                f"of {len(claim_texts)}"
            )

            print(
                f"Stage 2 batch "
                f"{batch_number}/"
                f"{len(safe_batches)}"
            )

            print(
                f"Estimated input tokens: "
                f"{estimated:,}"
            )

            chunk_embeddings = (
                embed_texts_with_retry(
                    client=client,
                    texts=claim_batch,
                    context_label=(
                        f"Stage 2 claims batch "
                        f"{batch_number}"
                    ),
                    max_retries=max_retries,
                )
            )

            all_claim_embeddings.extend(
                chunk_embeddings
            )

            processed_claims = (
                batch_end
            )

            print(
                f"Completed Stage 2 batch "
                f"{batch_number}/"
                f"{len(safe_batches)}"
            )

            if (
                claims_delay > 0
                and batch_number < len(safe_batches)
            ):
                time.sleep(
                    claims_delay
                )

        # -------------------------------------------------
        # Safety check
        # -------------------------------------------------

        if (
            len(all_claim_embeddings)
            != len(stage2_candidates)
        ):

            raise ValueError(
                "Claim embedding count does not match "
                "the number of Stage 2 candidates."
            )

        # -------------------------------------------------
        # Normalize claim embeddings
        # -------------------------------------------------

        claim_matrix = np.array(
            all_claim_embeddings,
            dtype="float32",
        )

        faiss.normalize_L2(
            claim_matrix
        )

        # -------------------------------------------------
        # Scores
        # -------------------------------------------------

        for position, idx in enumerate(
            stage2_candidates
        ):

            claim_scores[idx] = np.dot(
                claim_matrix[position],
                query_embedding[0],
            )

            final_scores[idx] = (
                ABSTRACT_WEIGHT
                * abstract_scores[idx]
                +
                CLAIM_WEIGHT
                * claim_scores[idx]
            )

    # -----------------------------------------------------
    # Stage 2 ranking
    # -----------------------------------------------------

    stage2_indices = np.array(
        sorted(
            stage2_candidates,
            key=lambda idx: final_scores[idx],
            reverse=True,
        ),
        dtype=int,
    )

    return {
        "abstract_scores": abstract_scores,
        "claim_scores": claim_scores,
        "final_scores": final_scores,
        "stage1_indices": stage1_indices,
        "stage2_indices": stage2_indices,
    }


# =========================================================
# Select Stage 2 results
# =========================================================

def select_full_treatment(
    stage2_indices,
    max_results=5,
):
    """Select up to max_results Stage 2 results."""

    return stage2_indices[
        :max_results
    ]


# =========================================================
# Print rankings
# =========================================================

def print_ranking(
    patents,
    indices,
    scores,
):
    """Print rankings in a consistent format."""

    for rank, idx in enumerate(
        indices,
        start=1,
    ):

        patent = patents[idx]

        print(
            f"Rank {rank}"
        )

        print(
            f"Patent ID: "
            f"{patent['id']}"
        )

        print(
            f"Title: "
            f"{patent['title']}"
        )

        print(
            f"Score: "
            f"{scores[idx]:.4f}"
        )

        print(
            "-" * 50
        )