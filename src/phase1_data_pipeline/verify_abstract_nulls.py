import sqlite3

conn = sqlite3.connect("patents.db")
cur = conn.cursor()

# Grab rows where abstract looks empty
cur.execute("""
    SELECT patent_id, abstract 
    FROM patents 
    WHERE abstract IS NULL OR TRIM(abstract) = ''
""")
rows = cur.fetchall()
print(f"Total flagged: {len(rows)}")

for r in rows[:20]:
    print(repr(r))

conn.close()