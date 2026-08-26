import sqlite3

conn = sqlite3.connect("../../data/patents.db")
cursor = conn.cursor()

candidates = [9589368, 9092877, 9189691, 9205850]

for pid in candidates:
    cursor.execute(
        "SELECT patent_id, title, claims_text FROM patents WHERE patent_id = ?",
        (pid,)
    )
    row = cursor.fetchone()
    if row is None:
        print(f"\n=== {pid} -- NOT FOUND ===")
        continue
    pid_, title, claims = row
    print(f"\n=== {pid_} -- {title} ===")
    # Print first ~800 chars of claims text -- enough to judge the
    # actual technical approach without dumping the entire claim set.
    print((claims or "")[:800])

conn.close()