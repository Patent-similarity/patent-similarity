import sqlite3

conn = sqlite3.connect("patents.db")
cur = conn.cursor()

# Step 1: Add the new column (only if it doesn't already exist)
try:
    cur.execute("ALTER TABLE patents ADD COLUMN abstract_status TEXT")
    print("Column 'abstract_status' added.")
except sqlite3.OperationalError as e:
    print(f"Skipped adding column (may already exist): {e}")

# Step 2: Populate it based on whether abstract is NULL
cur.execute("""
    UPDATE patents
    SET abstract_status = CASE
        WHEN abstract IS NULL THEN 'missing'
        ELSE 'found'
    END
""")
conn.commit()

# Step 3: Verify
cur.execute("SELECT DISTINCT abstract_status, COUNT(*) FROM patents GROUP BY abstract_status")
for row in cur.fetchall():
    print(row)

conn.close()