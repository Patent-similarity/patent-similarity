import json
import os
from pathlib import Path

import faiss
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_DIR = Path(
    "embeddings/full_corpus"
)

OUTPUT_DIR = Path(
    "embeddings/faiss_index"
)

EXPECTED_COUNT = 6779
EXPECTED_DIM = 3072
EXPECTED_BATCH_COUNT = 68

INDEX_PATH = (
    OUTPUT_DIR
    / "patent_similarity.faiss"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "patent_similarity_metadata.json"
)


# ============================================================
# KNOWN EVALUATION PATENTS
# ============================================================
#
# Add the remaining S1-S3 / D1-D3 patent IDs here once you
# have the confirmed list.
#
# Do not invent IDs for this check.
# ============================================================

SPOT_CHECK_PATENT_IDS = [
    # S1
    "11615557",
    "11620767",

    # S2
    "11676310",
    "11790567",

    # S3
    "10674162",
    "11818368",

    # D1
    "11676310",
    "11869140",

    # D2
    "11615557",
    "11776314",

    # D3
    "12307167",
    "10229528",
]


# ============================================================
# LOAD ALL BATCHES
# ============================================================

def load_all_batches():
    """
    Load every completed batch in logical batch order.

    The function requires exactly EXPECTED_BATCH_COUNT batches
    and requires their indices to be contiguous from 0 onward.
    """

    batch_files = sorted(
        BATCH_DIR.glob(
            "batch_*_embeddings.npy"
        )
    )

    if len(batch_files) != EXPECTED_BATCH_COUNT:

        raise RuntimeError(
            f"Batch file count mismatch.\n"
            f"Expected: {EXPECTED_BATCH_COUNT}\n"
            f"Found: {len(batch_files)}\n"
            f"Directory: {BATCH_DIR}"
        )

    all_embeddings = []
    all_metadata = []

    expected_batch_index = 0

    for emb_path in batch_files:

        batch_name = (
            emb_path.stem
            .replace(
                "_embeddings",
                "",
            )
        )

        if not batch_name.startswith(
            "batch_"
        ):

            raise RuntimeError(
                f"Unexpected batch filename: "
                f"{emb_path.name}"
            )

        try:

            batch_index = int(
                batch_name.split(
                    "_"
                )[1]
            )

        except (
            IndexError,
            ValueError,
        ) as error:

            raise RuntimeError(
                f"Could not determine batch index "
                f"from filename: {emb_path.name}"
            ) from error

        if (
            batch_index
            != expected_batch_index
        ):

            raise RuntimeError(
                f"Batch sequence mismatch.\n"
                f"Expected batch: "
                f"{expected_batch_index}\n"
                f"Found batch: "
                f"{batch_index}\n"
                f"File: {emb_path.name}"
            )

        meta_path = (
            BATCH_DIR
            / f"{batch_name}_metadata.jsonl"
        )

        if not meta_path.exists():

            raise RuntimeError(
                f"Missing metadata for "
                f"{batch_name}:\n"
                f"{meta_path}"
            )

        # ----------------------------------------------------
        # Load embeddings
        # ----------------------------------------------------

        try:

            embeddings = np.load(
                emb_path
            )

        except Exception as error:

            raise RuntimeError(
                f"Could not load embeddings:\n"
                f"{emb_path}"
            ) from error

        if embeddings.ndim != 2:

            raise RuntimeError(
                f"{batch_name}: expected a 2-D "
                f"embedding matrix, got "
                f"shape {embeddings.shape}"
            )

        if (
            embeddings.shape[1]
            != EXPECTED_DIM
        ):

            raise RuntimeError(
                f"{batch_name}: embedding dimension "
                f"mismatch.\n"
                f"Expected: {EXPECTED_DIM}\n"
                f"Found: {embeddings.shape[1]}"
            )

        if not np.isfinite(
            embeddings
        ).all():

            raise RuntimeError(
                f"{batch_name}: embeddings contain "
                f"NaN or infinite values."
            )

        # ----------------------------------------------------
        # Load metadata
        # ----------------------------------------------------

        metadata = []

        with meta_path.open(
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

                    record = json.loads(
                        line
                    )

                except json.JSONDecodeError as error:

                    raise RuntimeError(
                        f"{batch_name}: invalid JSON "
                        f"on metadata line "
                        f"{line_number}."
                    ) from error

                if not isinstance(
                    record,
                    dict,
                ):

                    raise RuntimeError(
                        f"{batch_name}: metadata line "
                        f"{line_number} is not a JSON object."
                    )

                if (
                    "patent_id"
                    not in record
                ):

                    raise RuntimeError(
                        f"{batch_name}: metadata line "
                        f"{line_number} has no patent_id."
                    )

                if (
                    record["patent_id"]
                    is None
                ):

                    raise RuntimeError(
                        f"{batch_name}: metadata line "
                        f"{line_number} has null patent_id."
                    )

                metadata.append(
                    record
                )

        # ----------------------------------------------------
        # Per-batch alignment check
        # ----------------------------------------------------

        if (
            embeddings.shape[0]
            != len(metadata)
        ):

            raise RuntimeError(
                f"{batch_name}: "
                f"{embeddings.shape[0]} embeddings "
                f"but {len(metadata)} metadata records."
            )

        print(
            f"Loaded {batch_name}: "
            f"{len(metadata)} records, "
            f"dimension {embeddings.shape[1]}"
        )

        all_embeddings.append(
            embeddings
        )

        all_metadata.extend(
            metadata
        )

        expected_batch_index += 1

    # --------------------------------------------------------
    # Concatenate in batch order
    # --------------------------------------------------------

    if not all_embeddings:

        raise RuntimeError(
            "No embedding batches were loaded."
        )

    embeddings = np.vstack(
        all_embeddings
    )

    return (
        embeddings,
        all_metadata,
    )


# ============================================================
# GLOBAL CORPUS CONSISTENCY
# ============================================================

def verify_global_consistency(
    embeddings,
    metadata,
):
    """
    Verify the complete corpus invariant before FAISS
    construction:

        6,779 metadata records
        6,779 embedding rows
        3,072 dimensions
        6,779 unique patent IDs
        no duplicate patent IDs
        no NaN/Inf embeddings
    """

    # --------------------------------------------------------
    # Metadata count
    # --------------------------------------------------------

    if (
        len(metadata)
        != EXPECTED_COUNT
    ):

        raise RuntimeError(
            f"Total record count mismatch.\n"
            f"Expected: {EXPECTED_COUNT}\n"
            f"Found: {len(metadata)}"
        )

    # --------------------------------------------------------
    # Embedding count
    # --------------------------------------------------------

    if (
        embeddings.shape[0]
        != EXPECTED_COUNT
    ):

        raise RuntimeError(
            f"Total embedding count mismatch.\n"
            f"Expected: {EXPECTED_COUNT}\n"
            f"Found: {embeddings.shape[0]}"
        )

    # --------------------------------------------------------
    # Embedding dimension
    # --------------------------------------------------------

    if (
        embeddings.shape[1]
        != EXPECTED_DIM
    ):

        raise RuntimeError(
            f"Embedding dimension mismatch.\n"
            f"Expected: {EXPECTED_DIM}\n"
            f"Found: {embeddings.shape[1]}"
        )

    # --------------------------------------------------------
    # Finite values
    # --------------------------------------------------------

    if not np.isfinite(
        embeddings
    ).all():

        raise RuntimeError(
            "Global embedding matrix contains "
            "NaN or infinite values."
        )

    # --------------------------------------------------------
    # Patent ID uniqueness
    # --------------------------------------------------------

    patent_ids = []

    for position, record in enumerate(
        metadata
    ):

        patent_id = record.get(
            "patent_id"
        )

        if patent_id is None:

            raise RuntimeError(
                f"Metadata position "
                f"{position} has no patent_id."
            )

        patent_ids.append(
            str(patent_id)
        )

    unique_patent_ids = set(
        patent_ids
    )

    if (
        len(unique_patent_ids)
        != EXPECTED_COUNT
    ):

        seen = set()
        duplicates = set()

        for patent_id in patent_ids:

            if patent_id in seen:

                duplicates.add(
                    patent_id
                )

            else:

                seen.add(
                    patent_id
                )

        raise RuntimeError(
            f"Duplicate patent_ids found.\n"
            f"Expected unique IDs: {EXPECTED_COUNT}\n"
            f"Found unique IDs: "
            f"{len(unique_patent_ids)}\n"
            f"Duplicates: "
            f"{sorted(duplicates)}"
        )

    # --------------------------------------------------------
    # Final invariant
    # --------------------------------------------------------

    print()
    print(
        "Global corpus consistency verified:"
    )

    print(
        f"  Records:       {len(metadata)}"
    )

    print(
        f"  Unique IDs:    {len(unique_patent_ids)}"
    )

    print(
        f"  Embeddings:    {embeddings.shape[0]}"
    )

    print(
        f"  Dimensions:    {embeddings.shape[1]}"
    )

    print(
        "  Duplicates:    0"
    )


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_index(
    embeddings,
):
    """
    Build IndexFlatIP from the already verified,
    batch-ordered embedding matrix.
    """

    index = faiss.IndexFlatIP(
        EXPECTED_DIM
    )

    index.add(
        embeddings
    )

    if (
        index.ntotal
        != EXPECTED_COUNT
    ):

        raise RuntimeError(
            f"FAISS index count mismatch "
            f"after add().\n"
            f"Expected: {EXPECTED_COUNT}\n"
            f"Found: {index.ntotal}"
        )

    print(
        f"FAISS index built: "
        f"{index.ntotal} vectors"
    )

    return index


# ============================================================
# SPOT CHECK
# ============================================================

def spot_check(
    index,
    metadata,
    patent_ids_to_check,
):
    """
    Verify known patent IDs against the actual FAISS rows.

    Because metadata and embeddings were concatenated in the
    same batch/row order, this directly tests that alignment
    survived assembly.
    """

    id_to_position = {}

    for position, record in enumerate(
        metadata
    ):

        patent_id = str(
            record["patent_id"]
        )

        id_to_position[
            patent_id
        ] = position

    print()
    print(
        "Spot-checking known patents..."
    )

    checked = 0

    for patent_id in patent_ids_to_check:

        patent_id = str(
            patent_id
        )

        if (
            patent_id
            not in id_to_position
        ):

            raise RuntimeError(
                f"Spot-check patent_id "
                f"{patent_id} was not found "
                f"in assembled metadata."
            )

        position = id_to_position[
            patent_id
        ]

        # IndexFlatIP supports reconstruct().
        vector = index.reconstruct(
            position
        )

        if vector.shape != (
            EXPECTED_DIM,
        ):

            raise RuntimeError(
                f"{patent_id}: reconstructed "
                f"vector has unexpected shape "
                f"{vector.shape}."
            )

        if not np.isfinite(
            vector
        ).all():

            raise RuntimeError(
                f"{patent_id}: reconstructed "
                f"vector contains NaN or Inf."
            )

        norm = float(
            np.linalg.norm(
                vector
            )
        )

        if (
            norm <= 0
        ):

            raise RuntimeError(
                f"{patent_id}: reconstructed "
                f"vector is zero."
            )

        if not np.isclose(
            norm,
            1.0,
            atol=1e-3,
        ):

            raise RuntimeError(
                f"{patent_id}: unexpected "
                f"vector norm {norm:.6f}; "
                f"expected approximately 1.0."
            )

        print(
            f"  OK: {patent_id} "
            f"(position {position}, "
            f"norm={norm:.6f}, "
            f"dim={vector.shape[0]})"
        )

        checked += 1

    if checked == 0:

        raise RuntimeError(
            "No spot-check patents were checked."
        )

    print(
        f"Spot checks passed: "
        f"{checked}"
    )


# ============================================================
# ATOMIC FAISS WRITE
# ============================================================

def write_index_atomic(
    index,
):
    """
    Write FAISS index to a temporary file and atomically
    replace the final path.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_tmp = Path(
        str(INDEX_PATH)
        + ".tmp"
    )

    # Remove an abandoned temporary file if present.
    if index_tmp.exists():

        index_tmp.unlink()

    try:

        faiss.write_index(
            index,
            str(index_tmp),
        )

        os.replace(
            index_tmp,
            INDEX_PATH,
        )

    except Exception:

        if index_tmp.exists():

            index_tmp.unlink()

        raise

    print(
        f"Wrote FAISS index:\n"
        f"{INDEX_PATH}"
    )


# ============================================================
# ATOMIC METADATA WRITE
# ============================================================

def write_metadata_atomic(
    metadata,
):
    """
    Write metadata JSON to a temporary file and atomically
    replace the final path.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_tmp = Path(
        str(METADATA_PATH)
        + ".tmp"
    )

    if metadata_tmp.exists():

        metadata_tmp.unlink()

    try:

        with metadata_tmp.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

            file.write("\n")

            file.flush()

            os.fsync(
                file.fileno()
            )

        os.replace(
            metadata_tmp,
            METADATA_PATH,
        )

    except Exception:

        if metadata_tmp.exists():

            metadata_tmp.unlink()

        raise

    print(
        f"Wrote metadata:\n"
        f"{METADATA_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "PATENT SIMILARITY — FAISS ASSEMBLY"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # 1. Load all batches
    # --------------------------------------------------------

    print()
    print(
        "Loading all embedding batches..."
    )

    embeddings, metadata = (
        load_all_batches()
    )

    # --------------------------------------------------------
    # 2. Global consistency verification
    # --------------------------------------------------------

    print()
    print(
        "Verifying global corpus consistency..."
    )

    verify_global_consistency(
        embeddings,
        metadata,
    )

    # --------------------------------------------------------
    # 3. Build FAISS
    # --------------------------------------------------------

    print()
    print(
        "Building IndexFlatIP..."
    )

    index = build_index(
        embeddings
    )

    # --------------------------------------------------------
    # 4. Sanity-check index
    # --------------------------------------------------------

    print()
    print(
        "Sanity-checking assembled FAISS index..."
    )

    if (
        index.ntotal
        != EXPECTED_COUNT
    ):

        raise RuntimeError(
            f"FAISS sanity check failed.\n"
            f"Expected ntotal: {EXPECTED_COUNT}\n"
            f"Actual ntotal: {index.ntotal}"
        )

    print(
        f"Index ntotal verified: "
        f"{index.ntotal}"
    )

    # --------------------------------------------------------
    # 5. Spot checks
    # --------------------------------------------------------

    spot_check(
        index=index,
        metadata=metadata,
        patent_ids_to_check=(
            SPOT_CHECK_PATENT_IDS
        ),
    )

    # --------------------------------------------------------
    # 6. Atomic writes
    # --------------------------------------------------------
    #
    # Only write after ALL validation above succeeds.
    # --------------------------------------------------------

    print()
    print(
        "Writing final artifacts atomically..."
    )

    write_index_atomic(
        index
    )

    write_metadata_atomic(
        metadata
    )

    # --------------------------------------------------------
    # 7. Final declaration
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "FAISS ASSEMBLY COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Vectors indexed: "
        f"{index.ntotal}"
    )

    print(
        f"Dimension: "
        f"{EXPECTED_DIM}"
    )

    print(
        f"Unique patents: "
        f"{len(metadata)}"
    )

    print(
        f"FAISS index:\n"
        f"{INDEX_PATH}"
    )

    print(
        f"Metadata:\n"
        f"{METADATA_PATH}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()