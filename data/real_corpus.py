import sqlite3


def load_real_corpus(db_path="data/patents.db"):
    """Load patents from the real SQLite corpus.

    Patents with missing or empty abstracts are excluded. The rebuilt
    schema (year+subgroup stratified) dropped the abstract_status
    column present in the original schema; missing abstracts are now
    identified directly via NULL/empty checks on the abstract field.

    Patents with missing claims keep claims=None so the
    retrieval pipeline can exclude them from Stage 2.
    """

    conn = sqlite3.connect(db_path)

    rows = conn.execute(
        """
        SELECT
            patent_id,
            title,
            abstract,
            claims_text,
            claims_status
        FROM patents
        WHERE abstract IS NOT NULL AND TRIM(abstract) != ''
        """
    ).fetchall()

    conn.close()

    patents = []

    for patent_id, title, abstract, claims_text, claims_status in rows:

        if claims_status == "missing":
            claims = None
        else:
            claims = claims_text

        patents.append(
            {
                "id": str(patent_id),
                "title": title,
                "abstract": abstract,
                "claims": claims,
            }
        )

    return patents

   