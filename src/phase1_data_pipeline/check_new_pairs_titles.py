import sqlite3

conn = sqlite3.connect("../../data/patents.db")
cursor = conn.cursor()

eval_ids = [8983133, 9014421, 9070197, 9025861, 9105091, 11714487]

for pid in eval_ids:
    cursor.execute("SELECT patent_id, title FROM patents WHERE patent_id = ?", (pid,))
    row = cursor.fetchone()
    print(row)

conn.close()