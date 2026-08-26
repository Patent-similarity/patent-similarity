import sqlite3
conn = sqlite3.connect("../../data/patents.db")
cursor = conn.cursor()

concepts = [
    ("object detection", "%detect%"),
    ("object tracking", "%track%"),
    ("image segmentation", "%segment%"),
    ("surface reconstruction", "%reconstruct%"),
    ("digital watermarking", "%watermark%"),
    ("gaze tracking", "%gaze%"),
]

for label, pattern in concepts:
    print(f"\n=== {label} ===")
    cursor.execute("SELECT patent_id, title FROM patents WHERE title LIKE ? LIMIT 5", (pattern,))
    rows = cursor.fetchall()
    if not rows:
        print("  (no matches)")
    for pid, title in rows:
        print(f"  {pid} — {title}")

conn.close()