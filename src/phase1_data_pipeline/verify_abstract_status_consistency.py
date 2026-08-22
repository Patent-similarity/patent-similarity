import sqlite3

conn = sqlite3.connect("patents.db")
cur = conn.cursor()

# Sanity check: any mismatch between abstract_status and actual abstract value?
cur.execute("""
    SELECT COUNT(*) FROM patents
    WHERE (abstract_status = 'found' AND abstract IS NULL)
       OR (abstract_status = 'missing' AND abstract IS NOT NULL)
""")
mismatches = cur.fetchone()[0]
print(f"Mismatches: {mismatches}")

conn.close()