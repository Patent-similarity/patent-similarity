import sqlite3

conn = sqlite3.connect("data/patents.db")
cur = conn.cursor()

print("Total patents:", cur.execute("SELECT COUNT(*) FROM patents").fetchone())
print("Sample cpc_group values:", cur.execute("SELECT DISTINCT cpc_group FROM patents LIMIT 20").fetchall())

conn.close()