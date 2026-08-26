import sqlite3

conn = sqlite3.connect("../../data/patents.db")
cursor = conn.cursor()

concepts = [
    ("object detection", "%detect%"),
    ("object tracking", "%track%"),
    ("moving object", "%moving object%"),
]

for label, pattern in concepts:
    print(f"\n=== {label} (claims_status = found) ===")
    cursor.execute(
        "SELECT patent_id, title FROM patents "
        "WHERE title LIKE ? AND claims_status = 'found' "
        "LIMIT 8",
        (pattern,)
    )
    rows = cursor.fetchall()
    if not rows:
        print("  (no matches)")
    for pid, title in rows:
        print(f"  {pid} -- {title}")

conn.close()