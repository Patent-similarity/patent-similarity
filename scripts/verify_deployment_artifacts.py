from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import faiss


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAISS_PATH = (
    PROJECT_ROOT
    / "embeddings"
    / "faiss_index"
    / "patent_similarity.faiss"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "embeddings"
    / "faiss_index"
    / "patent_similarity_metadata.json"
)

DATABASE_PATH = PROJECT_ROOT / "data" / "patents.db"

EXPECTED_DIMENSION = 3072
EXPECTED_VECTOR_COUNT = 6779

EXPECTED_FAISS_SHA256 = (
    "D5B4B5CED58E0FD8D8EEF97311058F7C31C24CC8B14BCC8AB531E8677A9C2911"
)

EXPECTED_METADATA_SHA256 = (
    "5D3F3513EF72CF4418A16494C54E7D1132B7A56F0FA97A2CC5DE6EFD099ED0E7"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest().upper()


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def check_file(path: Path, label: str) -> None:
    if not path.is_file():
        fail(f"{label} not found: {path}")

    print(f"[OK] {label}: {path}")


def main() -> None:
    print("=== Deployment Artifact Verification ===")
    print()

    # ---------------------------------------------------------
    # 1. Required files
    # ---------------------------------------------------------

    check_file(FAISS_PATH, "FAISS index")
    check_file(METADATA_PATH, "FAISS metadata")
    check_file(DATABASE_PATH, "Patent database")

    print()

    # ---------------------------------------------------------
    # 2. FAISS checksum
    # ---------------------------------------------------------

    actual_faiss_sha256 = sha256_file(FAISS_PATH)

    if actual_faiss_sha256 != EXPECTED_FAISS_SHA256:
        fail(
            "FAISS SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FAISS_SHA256}\n"
            f"Actual:   {actual_faiss_sha256}"
        )

    print(f"[OK] FAISS SHA256: {actual_faiss_sha256}")

    # ---------------------------------------------------------
    # 3. Metadata checksum
    # ---------------------------------------------------------

    actual_metadata_sha256 = sha256_file(METADATA_PATH)

    if actual_metadata_sha256 != EXPECTED_METADATA_SHA256:
        fail(
            "Metadata SHA256 mismatch.\n"
            f"Expected: {EXPECTED_METADATA_SHA256}\n"
            f"Actual:   {actual_metadata_sha256}"
        )

    print(f"[OK] Metadata SHA256: {actual_metadata_sha256}")

    print()

    # ---------------------------------------------------------
    # 4. Load FAISS
    # ---------------------------------------------------------

    try:
        index = faiss.read_index(str(FAISS_PATH))
    except Exception as exc:
        fail(f"Could not load FAISS index: {exc}")

    print("[OK] FAISS index loaded")

    # ---------------------------------------------------------
    # 5. Validate FAISS dimensions
    # ---------------------------------------------------------

    if index.d != EXPECTED_DIMENSION:
        fail(
            "FAISS dimension mismatch.\n"
            f"Expected: {EXPECTED_DIMENSION}\n"
            f"Actual:   {index.d}"
        )

    print(f"[OK] FAISS dimension: {index.d}")

    # ---------------------------------------------------------
    # 6. Validate vector count
    # ---------------------------------------------------------

    if index.ntotal != EXPECTED_VECTOR_COUNT:
        fail(
            "FAISS vector count mismatch.\n"
            f"Expected: {EXPECTED_VECTOR_COUNT}\n"
            f"Actual:   {index.ntotal}"
        )

    print(f"[OK] FAISS vector count: {index.ntotal}")

    print()

    # ---------------------------------------------------------
    # 7. Load metadata
    # ---------------------------------------------------------

    try:
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:
        fail(f"Could not load metadata JSON: {exc}")

    if not isinstance(metadata, list):
        fail(
            "Metadata JSON must contain a list. "
            f"Found: {type(metadata).__name__}"
        )

    if len(metadata) != EXPECTED_VECTOR_COUNT:
        fail(
            "Metadata count mismatch.\n"
            f"Expected: {EXPECTED_VECTOR_COUNT}\n"
            f"Actual:   {len(metadata)}"
        )

    print(f"[OK] Metadata count: {len(metadata)}")

    # ---------------------------------------------------------
    # 8. Validate metadata structure
    # ---------------------------------------------------------

    for position, item in enumerate(metadata):
        if not isinstance(item, dict):
            fail(
                f"Metadata entry {position} is not an object."
            )

        if "patent_id" not in item:
            fail(
                f"Metadata entry {position} is missing "
                "'patent_id'."
            )

        if "title" not in item:
            fail(
                f"Metadata entry {position} is missing "
                "'title'."
            )

    print("[OK] Metadata structure")

    # ---------------------------------------------------------
    # 9. Validate metadata / FAISS positional alignment
    # ---------------------------------------------------------

    # The current production index was built using deterministic
    # patent ordering. This check verifies that every metadata
    # position corresponds to an actual patent_id in the database.

    import sqlite3

    try:
        connection = sqlite3.connect(DATABASE_PATH)

        rows = connection.execute(
            """
            SELECT patent_id
            FROM patents
            WHERE abstract IS NOT NULL
            ORDER BY patent_id
            """
        ).fetchall()

        connection.close()

    except Exception as exc:
        fail(f"Could not read patent database: {exc}")

    database_patent_ids = [row[0] for row in rows]
    metadata_patent_ids = [item["patent_id"] for item in metadata]

    if len(database_patent_ids) != EXPECTED_VECTOR_COUNT:
        fail(
            "Database abstract-bearing patent count mismatch.\n"
            f"Expected: {EXPECTED_VECTOR_COUNT}\n"
            f"Actual:   {len(database_patent_ids)}"
        )

    if metadata_patent_ids != database_patent_ids:
        fail(
            "Metadata/FAISS/database positional alignment failed."
        )

    print(
        "[OK] Metadata aligns with the "
        "6,779 abstract-bearing database patents"
    )

    print()

    # ---------------------------------------------------------
    # 10. Success
    # ---------------------------------------------------------

    print("=== Deployment Artifact Verification PASSED ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nVerification interrupted.")
        sys.exit(130)