import hashlib
import json
import os
import sqlite3
import time

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from dotenv import load_dotenv

from src.phase2_embedding_retrieval.embedding_pipeline import (
    embed_texts_with_retry,
    split_texts_by_token_budget,
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()


# ------------------------------------------------------------
# Corpus
# ------------------------------------------------------------

EXPECTED_PATENT_COUNT = 6779


# ------------------------------------------------------------
# Gemini batching
# ------------------------------------------------------------

BATCH_SIZE = 100

# Conservative free-tier daily safety target.
MAX_ITEMS_PER_DAY = 1000

# Delay between logical batches.
BATCH_DELAY_SECONDS = 60


# ------------------------------------------------------------
# Embedding validation
# ------------------------------------------------------------

EXPECTED_EMBEDDING_DIM = 3072


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

INPUT_DATA_PATH = Path(
    "data/patents.db"
)

OUTPUT_DIR = Path(
    "embeddings/full_corpus"
)

CHECKPOINT_PATH = (
    OUTPUT_DIR / "checkpoint.json"
)


# ============================================================
# DATE / QUOTA TIMEZONE
# ============================================================

GEMINI_RESET_TIMEZONE = ZoneInfo(
    "America/Los_Angeles"
)


def get_today_string():
    """
    Return the current date in the quota-reset timezone.
    """

    return datetime.now(
        GEMINI_RESET_TIMEZONE
    ).date().isoformat()


# ============================================================
# CORPUS HASH
# ============================================================

def compute_corpus_hash(path):
    """
    Compute a SHA-256 hash of the raw corpus database file.

    The hash is stored in the checkpoint and checked on every
    resume. If the database changes during the multi-day build,
    the build refuses to continue.
    """

    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_embedding_client():
    """
    Create the Google GenAI client.
    """
    from google import genai
    import os

    return genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        vertexai=False,
    )

# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_configuration():
    """
    Validate important configuration values before making
    embedding API calls.
    """

    if EXPECTED_EMBEDDING_DIM is None:

        raise RuntimeError(
            "EXPECTED_EMBEDDING_DIM is not set."
        )

    if (
        not isinstance(
            EXPECTED_EMBEDDING_DIM,
            int,
        )
        or EXPECTED_EMBEDDING_DIM <= 0
    ):

        raise RuntimeError(
            "EXPECTED_EMBEDDING_DIM must be "
            "a positive integer."
        )

    if BATCH_SIZE <= 0:

        raise RuntimeError(
            "BATCH_SIZE must be greater than zero."
        )

    if BATCH_SIZE > 100:

        raise RuntimeError(
            "BATCH_SIZE cannot exceed 100."
        )

    if MAX_ITEMS_PER_DAY <= 0:

        raise RuntimeError(
            "MAX_ITEMS_PER_DAY must be greater than zero."
        )

    if not INPUT_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Phase 1 database does not exist:\n"
            f"{INPUT_DATA_PATH}"
        )


# ============================================================
# PATENT ID HELPER
# ============================================================

def get_patent_id(patent):
    """
    Return the patent identifier.
    """

    patent_id = patent.get(
        "id"
    )

    if patent_id is None:

        patent_id = patent.get(
            "patent_id"
        )

    if patent_id is None:

        raise RuntimeError(
            "Patent record has neither 'id' "
            "nor 'patent_id'."
        )

    return patent_id


# ============================================================
# DATA LOADING
# ============================================================

def load_patents():
    """
    Load the finalized patent corpus from SQLite.

    Only patents with usable abstracts are included.

    Order is deterministic:
        ORDER BY patent_id
    """

    patents = []

    connection = sqlite3.connect(
        INPUT_DATA_PATH
    )

    try:

        cursor = connection.execute(
            """
            SELECT
                patent_id,
                title,
                abstract
            FROM patents
            WHERE abstract IS NOT NULL
              AND TRIM(abstract) <> ''
            ORDER BY patent_id
            """
        )

        for patent_id, title, abstract in cursor:

            patent = {
                "id": patent_id,
                "title": title,
                "abstract": abstract,
            }

            get_patent_id(
                patent
            )

            patents.append(
                patent
            )

    finally:

        connection.close()

    print(
        f"Loaded {len(patents)} patents "
        f"with usable abstracts."
    )

    # Hard corpus invariant.

    if len(patents) != EXPECTED_PATENT_COUNT:

        raise RuntimeError(
            f"\nCORPUS COUNT MISMATCH.\n"
            f"Expected: {EXPECTED_PATENT_COUNT}\n"
            f"Found: {len(patents)}\n\n"
            f"Refusing to make embedding calls."
        )

    print(
        f"Corpus count verified: "
        f"{EXPECTED_PATENT_COUNT}"
    )

    return patents


# ============================================================
# BATCHING
# ============================================================

def get_batches(patents):
    """
    Generate deterministic logical batches.

    Each batch contains at most BATCH_SIZE patents.
    """

    for start in range(
        0,
        len(patents),
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            len(patents),
        )

        batch_index = (
            start // BATCH_SIZE
        )

        yield (
            batch_index,
            patents[start:end],
        )


# ============================================================
# BATCH PATHS
# ============================================================

def get_batch_paths(batch_index):
    """
    Return final and temporary paths for one batch.
    """

    batch_name = (
        f"batch_{batch_index:03d}"
    )

    embeddings_path = (
        OUTPUT_DIR
        / f"{batch_name}_embeddings.npy"
    )

    metadata_path = (
        OUTPUT_DIR
        / f"{batch_name}_metadata.jsonl"
    )

    embeddings_tmp = (
        OUTPUT_DIR
        / f"{batch_name}_embeddings.npy.tmp"
    )

    metadata_tmp = (
        OUTPUT_DIR
        / f"{batch_name}_metadata.jsonl.tmp"
    )

    return (
        embeddings_path,
        metadata_path,
        embeddings_tmp,
        metadata_tmp,
    )


# ============================================================
# ATOMIC EMBEDDING WRITE
# ============================================================

def write_embeddings_atomic(
    embeddings,
    tmp_path,
    final_path,
):
    """
    Write embeddings to a temporary file and atomically
    replace the final file.
    """

    with tmp_path.open(
        "wb"
    ) as file:

        np.save(
            file,
            embeddings,
        )

        file.flush()

        os.fsync(
            file.fileno()
        )

    os.replace(
        tmp_path,
        final_path,
    )


# ============================================================
# ATOMIC METADATA WRITE
# ============================================================

def write_metadata_atomic(
    batch,
    tmp_path,
    final_path,
):
    """
    Write metadata atomically.
    """

    with tmp_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for patent in batch:

            metadata = {
                "patent_id": get_patent_id(
                    patent
                ),
                "title": patent.get(
                    "title"
                ),
            }

            file.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                )
                + "\n"
            )

        file.flush()

        os.fsync(
            file.fileno()
        )

    os.replace(
        tmp_path,
        final_path,
    )


# ============================================================
# ATOMIC CHECKPOINT WRITE
# ============================================================

def write_checkpoint_atomic(
    last_completed_batch,
    processed_count,
    daily_count,
    daily_date,
    corpus_hash,
):
    """
    Atomically write the checkpoint.

    The checkpoint is written LAST after embeddings and metadata
    have already been committed and verified.
    """

    checkpoint_tmp = (
        OUTPUT_DIR
        / "checkpoint.json.tmp"
    )

    checkpoint = {
        "last_completed_batch":
            last_completed_batch,

        "processed_count":
            processed_count,

        "daily_count":
            daily_count,

        "daily_date":
            daily_date,

        "corpus_hash":
            corpus_hash,
    }

    with checkpoint_tmp.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            checkpoint,
            file,
            indent=2,
        )

        file.write(
            "\n"
        )

        file.flush()

        os.fsync(
            file.fileno()
        )

    os.replace(
        checkpoint_tmp,
        CHECKPOINT_PATH,
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint():
    """
    Load an existing checkpoint.

    Returns None for a fresh build.

    Older checkpoints without daily quota tracking or corpus
    hashing are migrated conservatively.
    """

    if not CHECKPOINT_PATH.exists():

        return None

    try:

        with CHECKPOINT_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:

            checkpoint = json.load(
                file
            )

    except Exception as error:

        raise RuntimeError(
            f"Could not read checkpoint:\n"
            f"{CHECKPOINT_PATH}"
        ) from error

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Checkpoint is not a JSON object."
        )

    # --------------------------------------------------------
    # Required original fields
    # --------------------------------------------------------

    required_fields = {
        "last_completed_batch",
        "processed_count",
    }

    missing = (
        required_fields
        - checkpoint.keys()
    )

    if missing:

        raise RuntimeError(
            "Checkpoint is missing required fields: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Validate original fields
    # --------------------------------------------------------

    if not isinstance(
        checkpoint[
            "last_completed_batch"
        ],
        int,
    ):

        raise RuntimeError(
            "checkpoint.last_completed_batch "
            "must be an integer."
        )

    if not isinstance(
        checkpoint[
            "processed_count"
        ],
        int,
    ):

        raise RuntimeError(
            "checkpoint.processed_count "
            "must be an integer."
        )

    if (
        checkpoint[
            "last_completed_batch"
        ]
        < 0
    ):

        raise RuntimeError(
            "checkpoint.last_completed_batch "
            "cannot be negative."
        )

    if (
        checkpoint[
            "processed_count"
        ]
        < 0
    ):

        raise RuntimeError(
            "checkpoint.processed_count "
            "cannot be negative."
        )

    # --------------------------------------------------------
    # Daily quota migration
    # --------------------------------------------------------

    today = get_today_string()

    if (
        "daily_count"
        not in checkpoint
    ):

        checkpoint[
            "daily_count"
        ] = min(
            checkpoint[
                "processed_count"
            ],
            MAX_ITEMS_PER_DAY,
        )

        checkpoint[
            "daily_date"
        ] = today

        print(
            "\nOld checkpoint detected."
        )

        print(
            "Initializing persistent daily quota tracking."
        )

        print(
            f"Conservative daily count: "
            f"{checkpoint['daily_count']}"
        )

    else:

        daily_count = checkpoint[
            "daily_count"
        ]

        if (
            not isinstance(
                daily_count,
                int,
            )
            or daily_count < 0
        ):

            raise RuntimeError(
                "checkpoint.daily_count must be "
                "a non-negative integer."
            )

        stored_daily_date = (
            checkpoint.get(
                "daily_date"
            )
        )

        if not isinstance(
            stored_daily_date,
            str,
        ):

            raise RuntimeError(
                "checkpoint.daily_date "
                "must be a string."
            )

        if stored_daily_date != today:

            print(
                "\nNew calendar day detected."
            )

            print(
                f"Previous daily date: "
                f"{stored_daily_date}"
            )

            print(
                f"Current date: "
                f"{today}"
            )

            checkpoint[
                "daily_count"
            ] = 0

            checkpoint[
                "daily_date"
            ] = today

            print(
                "Daily embedding count reset to 0."
            )

    # --------------------------------------------------------
    # Corpus hash migration
    # --------------------------------------------------------

    if (
        "corpus_hash"
        not in checkpoint
    ):

        checkpoint[
            "corpus_hash"
        ] = None

        print(
            "\nOld checkpoint detected without corpus hash."
        )

        print(
            "The current corpus hash will be adopted "
            "after existing completed batches are verified."
        )

    else:

        corpus_hash = checkpoint[
            "corpus_hash"
        ]

        if (
            corpus_hash is not None
            and not isinstance(
                corpus_hash,
                str,
            )
        ):

            raise RuntimeError(
                "checkpoint.corpus_hash "
                "must be a string."
            )

    return checkpoint


# ============================================================
# BATCH FILE VERIFICATION
# ============================================================

def verify_batch_files(
    batch_index,
    expected_row_count,
):
    """
    Verify embeddings and metadata for a completed batch.
    """

    (
        embeddings_path,
        metadata_path,
        _,
        _,
    ) = get_batch_paths(
        batch_index
    )

    # --------------------------------------------------------
    # Existence
    # --------------------------------------------------------

    if not embeddings_path.exists():

        raise RuntimeError(
            f"Checkpoint claims batch "
            f"{batch_index} is complete, "
            f"but embeddings file is missing:\n"
            f"{embeddings_path}"
        )

    if not metadata_path.exists():

        raise RuntimeError(
            f"Checkpoint claims batch "
            f"{batch_index} is complete, "
            f"but metadata file is missing:\n"
            f"{metadata_path}"
        )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    try:

        embeddings = np.load(
            embeddings_path
        )

    except Exception as error:

        raise RuntimeError(
            f"Could not load embeddings "
            f"for batch {batch_index}."
        ) from error

    if embeddings.ndim != 2:

        raise RuntimeError(
            f"Batch {batch_index} has invalid "
            f"embedding shape: "
            f"{embeddings.shape}"
        )

    if (
        embeddings.shape[0]
        != expected_row_count
    ):

        raise RuntimeError(
            f"Batch {batch_index} embedding "
            f"row count mismatch.\n"
            f"Expected: {expected_row_count}\n"
            f"Found: {embeddings.shape[0]}"
        )

    if (
        embeddings.shape[1]
        != EXPECTED_EMBEDDING_DIM
    ):

        raise RuntimeError(
            f"Batch {batch_index} embedding "
            f"dimension mismatch.\n"
            f"Expected: "
            f"{EXPECTED_EMBEDDING_DIM}\n"
            f"Found: {embeddings.shape[1]}"
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_count = 0

    try:

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:

                    metadata = json.loads(
                        line
                    )

                except json.JSONDecodeError as error:

                    raise RuntimeError(
                        f"Invalid metadata JSON "
                        f"in batch {batch_index}, "
                        f"line {line_number}."
                    ) from error

                if (
                    metadata.get(
                        "patent_id"
                    )
                    is None
                ):

                    raise RuntimeError(
                        f"Metadata line "
                        f"{line_number} in batch "
                        f"{batch_index} has no patent_id."
                    )

                metadata_count += 1

    except RuntimeError:

        raise

    except Exception as error:

        raise RuntimeError(
            f"Could not read metadata "
            f"for batch {batch_index}."
        ) from error

    if (
        metadata_count
        != expected_row_count
    ):

        raise RuntimeError(
            f"Batch {batch_index} metadata "
            f"row count mismatch.\n"
            f"Expected: {expected_row_count}\n"
            f"Found: {metadata_count}"
        )

    if (
        embeddings.shape[0]
        != metadata_count
    ):

        raise RuntimeError(
            f"Batch {batch_index} contains "
            f"{embeddings.shape[0]} embeddings "
            f"but {metadata_count} metadata records."
        )

    print(
        f"Verified batch {batch_index}: "
        f"{metadata_count} rows, "
        f"dimension {embeddings.shape[1]}"
    )


# ============================================================
# VERIFY CHECKPOINT
# ============================================================

def verify_checkpoint(
    patents,
    checkpoint,
):
    """
    Verify every batch claimed as complete.
    """

    last_completed_batch = (
        checkpoint[
            "last_completed_batch"
        ]
    )

    processed_count = (
        checkpoint[
            "processed_count"
        ]
    )

    batches = list(
        get_batches(
            patents
        )
    )

    if (
        last_completed_batch
        >= len(batches)
    ):

        raise RuntimeError(
            f"Checkpoint references batch "
            f"{last_completed_batch}, but only "
            f"{len(batches)} batches exist."
        )

    expected_processed_count = sum(
        len(batch)
        for _, batch in batches[
            :last_completed_batch + 1
        ]
    )

    if (
        processed_count
        != expected_processed_count
    ):

        raise RuntimeError(
            f"Checkpoint processed_count mismatch.\n"
            f"Checkpoint says: "
            f"{processed_count}\n"
            f"Expected from completed batches: "
            f"{expected_processed_count}"
        )

    print(
        f"\nVerifying completed batches "
        f"0-{last_completed_batch}..."
    )

    for batch_index in range(
        last_completed_batch + 1
    ):

        _, batch = batches[
            batch_index
        ]

        verify_batch_files(
            batch_index=batch_index,
            expected_row_count=len(batch),
        )

    print(
        "Checkpoint verification complete."
    )


# ============================================================
# TEMP FILE CLEANUP
# ============================================================

def remove_stale_temp_files(
    batch_index,
):
    """
    Remove temporary files from interrupted writes.

    Temporary files are never treated as completed batches.
    """

    (
        _,
        _,
        embeddings_tmp,
        metadata_tmp,
    ) = get_batch_paths(
        batch_index
    )

    for path in (
        embeddings_tmp,
        metadata_tmp,
    ):

        if path.exists():

            print(
                f"Removing stale temporary file:\n"
                f"{path}"
            )

            path.unlink()


# ============================================================
# EMBED ONE LOGICAL BATCH
# ============================================================

def embed_logical_batch(
    client,
    batch_index,
    batch,
):
    """
    Embed one logical batch.

    The logical batch can be split into multiple token-safe
    API requests.
    """

    texts = []

    for patent in batch:

        abstract = patent.get(
            "abstract"
        )

        if (
            abstract is None
            or not str(abstract).strip()
        ):

            raise RuntimeError(
                f"Patent "
                f"{get_patent_id(patent)} "
                f"in batch {batch_index} "
                f"has no usable abstract."
            )

        texts.append(
            str(abstract)
        )

    # --------------------------------------------------------
    # Token-safe splitting
    # --------------------------------------------------------

    safe_batches = (
        split_texts_by_token_budget(
            texts
        )
    )

    print(
        f"Logical batch {batch_index}: "
        f"{len(texts)} patents"
    )

    print(
        f"Token-safe API requests: "
        f"{len(safe_batches)}"
    )

    # --------------------------------------------------------
    # Sequential embedding
    # --------------------------------------------------------

    all_embeddings = []

    for request_number, safe_batch in enumerate(
        safe_batches,
        start=1,
    ):

        print(
            f"\nBatch {batch_index} — "
            f"API request "
            f"{request_number}/"
            f"{len(safe_batches)}"
        )

        print(
            f"Texts in request: "
            f"{len(safe_batch)}"
        )

        embeddings = (
            embed_texts_with_retry(
                client=client,
                texts=safe_batch,
                context_label=(
                    f"Full corpus batch "
                    f"{batch_index}, "
                    f"request "
                    f"{request_number}"
                ),
                max_retries=5,
            )
        )

        all_embeddings.extend(
            embeddings
        )

    # --------------------------------------------------------
    # Count validation
    # --------------------------------------------------------

    if (
        len(all_embeddings)
        != len(batch)
    ):

        raise RuntimeError(
            f"Batch {batch_index} embedding "
            f"count mismatch.\n"
            f"Expected: {len(batch)}\n"
            f"Found: {len(all_embeddings)}"
        )

    embeddings = np.asarray(
        all_embeddings,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    if embeddings.ndim != 2:

        raise RuntimeError(
            f"Batch {batch_index} produced "
            f"invalid embedding shape: "
            f"{embeddings.shape}"
        )

    if (
        embeddings.shape[0]
        != len(batch)
    ):

        raise RuntimeError(
            f"Batch {batch_index} embedding "
            f"row count mismatch."
        )

    if (
        embeddings.shape[1]
        != EXPECTED_EMBEDDING_DIM
    ):

        raise RuntimeError(
            f"Batch {batch_index} embedding "
            f"dimension mismatch.\n"
            f"Expected: "
            f"{EXPECTED_EMBEDDING_DIM}\n"
            f"Found: {embeddings.shape[1]}"
        )

    print(
        f"\nBatch {batch_index} embedding matrix: "
        f"{embeddings.shape}"
    )

    return embeddings


# ============================================================
# MAIN BUILD
# ============================================================

def build_full_embeddings():
    """
    Build durable per-batch abstract embeddings.

    Features:

    - deterministic SQLite corpus loading
    - corpus count verification
    - corpus SHA-256 verification
    - persistent daily quota tracking
    - token-safe embedding requests
    - atomic file writes
    - checkpoint resume
    - completed-batch verification

    FAISS assembly is intentionally separate.
    """

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    validate_configuration()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    patents = load_patents()

    # --------------------------------------------------------
    # Compute corpus hash
    # --------------------------------------------------------

    current_corpus_hash = (
        compute_corpus_hash(
            INPUT_DATA_PATH
        )
    )

    print(
        f"\nCorpus SHA-256:\n"
        f"{current_corpus_hash}"
    )

    # --------------------------------------------------------
    # Build logical batches
    # --------------------------------------------------------

    batches = list(
        get_batches(
            patents
        )
    )

    print(
        f"\nTotal logical batches: "
        f"{len(batches)}"
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = load_checkpoint()

    start_batch = 0
    processed_count = 0

    daily_count = 0
    daily_date = get_today_string()

    if checkpoint is not None:

        print(
            "\nExisting checkpoint found:"
        )

        print(
            json.dumps(
                checkpoint,
                indent=2,
            )
        )

        # ----------------------------------------------------
        # Corpus hash verification
        # ----------------------------------------------------

        stored_hash = checkpoint.get(
            "corpus_hash"
        )

        if stored_hash is not None:

            if (
                stored_hash
                != current_corpus_hash
            ):

                raise RuntimeError(
                    f"\nCORPUS HASH MISMATCH.\n\n"
                    f"Checkpoint hash:\n"
                    f"{stored_hash}\n\n"
                    f"Current hash:\n"
                    f"{current_corpus_hash}\n\n"
                    f"The corpus database has changed "
                    f"since this embedding build started.\n\n"
                    f"Refusing to resume."
                )

            print(
                "\nCorpus hash verified: "
                "matches checkpoint."
            )

        else:

            print(
                "\nNo historical corpus hash exists "
                "in this old checkpoint."
            )

            print(
                "Existing completed batches will be "
                "verified before adopting the current hash."
            )

        # ----------------------------------------------------
        # Verify completed files
        # ----------------------------------------------------

        verify_checkpoint(
            patents=patents,
            checkpoint=checkpoint,
        )

        last_completed_batch = (
            checkpoint[
                "last_completed_batch"
            ]
        )

        start_batch = (
            last_completed_batch + 1
        )

        processed_count = (
            checkpoint[
                "processed_count"
            ]
        )

        daily_count = (
            checkpoint[
                "daily_count"
            ]
        )

        daily_date = (
            checkpoint[
                "daily_date"
            ]
        )

        print(
            "\nResume approved."
        )

        print(
            f"Starting from logical batch "
            f"{start_batch}."
        )

    else:

        print(
            "\nNo checkpoint found."
        )

        print(
            "Starting fresh from batch 0."
        )

    # --------------------------------------------------------
    # Daily quota status
    # --------------------------------------------------------

    today = get_today_string()

    if daily_date != today:

        daily_date = today
        daily_count = 0

    print(
        "\nDaily quota status:"
    )

    print(
        f"Date: {daily_date}"
    )

    print(
        f"Completed today: "
        f"{daily_count}/"
        f"{MAX_ITEMS_PER_DAY}"
    )

    # --------------------------------------------------------
    # Already complete?
    # --------------------------------------------------------

    if (
        start_batch
        >= len(batches)
    ):

        print(
            "\nAll logical batches are already complete."
        )

        print(
            "No embedding API calls required."
        )

        return

    # --------------------------------------------------------
    # Daily guard
    # --------------------------------------------------------

    if (
        daily_count
        >= MAX_ITEMS_PER_DAY
    ):

        print(
            "\nDAILY SAFETY LIMIT REACHED."
        )

        print(
            f"Completed today: "
            f"{daily_count}/"
            f"{MAX_ITEMS_PER_DAY}"
        )

        print(
            "Stopping cleanly."
        )

        print(
            "Run the script again after the next "
            "Gemini quota reset."
        )

        return

    # --------------------------------------------------------
    # Create Gemini client
    # --------------------------------------------------------

    client = get_embedding_client()

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    for batch_index in range(
        start_batch,
        len(batches),
    ):

        _, batch = batches[
            batch_index
        ]

        # ----------------------------------------------------
        # Check whether a new quota day has started
        # ----------------------------------------------------

        today = get_today_string()

        if daily_date != today:

            print(
                "\nNew calendar day detected."
            )

            daily_date = today
            daily_count = 0

            print(
                "Daily embedding count reset to 0."
            )

        # ----------------------------------------------------
        # Persistent daily guard
        # ----------------------------------------------------

        if (
            daily_count
            + len(batch)
            > MAX_ITEMS_PER_DAY
        ):

            print(
                "\nNEXT BATCH WOULD EXCEED "
                "DAILY SAFETY LIMIT."
            )

            print(
                f"Completed today: "
                f"{daily_count}"
            )

            print(
                f"Next batch size: "
                f"{len(batch)}"
            )

            print(
                f"Daily limit: "
                f"{MAX_ITEMS_PER_DAY}"
            )

            print(
                "\nStopping cleanly."
            )

            return

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"PROCESSING LOGICAL BATCH "
            f"{batch_index + 1}/"
            f"{len(batches)}"
        )

        print(
            f"Batch index: "
            f"{batch_index}"
        )

        print(
            f"Patents in batch: "
            f"{len(batch)}"
        )

        print(
            f"Previously processed total: "
            f"{processed_count}"
        )

        print(
            f"Today's completed count: "
            f"{daily_count}/"
            f"{MAX_ITEMS_PER_DAY}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Remove stale temporary files
        # ----------------------------------------------------

        remove_stale_temp_files(
            batch_index
        )

        # ----------------------------------------------------
        # Embed
        # ----------------------------------------------------

        embeddings = embed_logical_batch(
            client=client,
            batch_index=batch_index,
            batch=batch,
        )

        # ----------------------------------------------------
        # Get output paths
        # ----------------------------------------------------

        (
            embeddings_path,
            metadata_path,
            embeddings_tmp,
            metadata_tmp,
        ) = get_batch_paths(
            batch_index
        )

        # ----------------------------------------------------
        # Commit embeddings
        # ----------------------------------------------------

        print(
            "\nWriting embeddings..."
        )

        write_embeddings_atomic(
            embeddings=embeddings,
            tmp_path=embeddings_tmp,
            final_path=embeddings_path,
        )

        print(
            f"Committed embeddings:\n"
            f"{embeddings_path}"
        )

        # ----------------------------------------------------
        # Commit metadata
        # ----------------------------------------------------

        print(
            "Writing metadata..."
        )

        write_metadata_atomic(
            batch=batch,
            tmp_path=metadata_tmp,
            final_path=metadata_path,
        )

        print(
            f"Committed metadata:\n"
            f"{metadata_path}"
        )

        # ----------------------------------------------------
        # Verify committed batch
        # ----------------------------------------------------

        print(
            "\nVerifying committed batch..."
        )

        verify_batch_files(
            batch_index=batch_index,
            expected_row_count=len(batch),
        )

        # ----------------------------------------------------
        # Update counts only after verification
        # ----------------------------------------------------

        processed_count += len(
            batch
        )

        daily_count += len(
            batch
        )

        # ----------------------------------------------------
        # Checkpoint last
        # ----------------------------------------------------

        write_checkpoint_atomic(
            last_completed_batch=batch_index,
            processed_count=processed_count,
            daily_count=daily_count,
            daily_date=daily_date,
            corpus_hash=current_corpus_hash,
        )

        print(
            "Checkpoint committed."
        )

        print(
            f"Last completed batch: "
            f"{batch_index}"
        )

        print(
            f"Total processed: "
            f"{processed_count}/"
            f"{len(patents)}"
        )

        print(
            f"Today's processed: "
            f"{daily_count}/"
            f"{MAX_ITEMS_PER_DAY}"
        )

        # ----------------------------------------------------
        # Stop if daily limit reached
        # ----------------------------------------------------

        if (
            daily_count
            >= MAX_ITEMS_PER_DAY
        ):

            print(
                "\n"
                + "=" * 70
            )

            print(
                "DAILY EMBEDDING SAFETY LIMIT REACHED"
            )

            print(
                "=" * 70
            )

            print(
                f"Today's completed embeddings: "
                f"{daily_count}"
            )

            print(
                f"Daily safety limit: "
                f"{MAX_ITEMS_PER_DAY}"
            )

            print(
                f"Total corpus progress: "
                f"{processed_count}/"
                f"{len(patents)}"
            )

            print(
                "\nStopping cleanly."
            )

            print(
                "Run the same command after the next "
                "quota reset to resume automatically."
            )

            return

        # ----------------------------------------------------
        # Delay before next logical batch
        # ----------------------------------------------------

        if (
            batch_index
            < len(batches) - 1
        ):

            print(
                f"\nWaiting "
                f"{BATCH_DELAY_SECONDS}s "
                f"before next logical batch..."
            )

            time.sleep(
                BATCH_DELAY_SECONDS
            )

    # --------------------------------------------------------
    # Final completion
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FULL CORPUS EMBEDDING BUILD COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Patents embedded: "
        f"{len(patents)}"
    )

    print(
        f"Logical batches: "
        f"{len(batches)}"
    )

    print(
        f"Embedding dimension: "
        f"{EXPECTED_EMBEDDING_DIM}"
    )

    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        f"Checkpoint:\n"
        f"{CHECKPOINT_PATH}"
    )

    print(
        "\nFAISS assembly was intentionally "
        "NOT performed here."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    build_full_embeddings()