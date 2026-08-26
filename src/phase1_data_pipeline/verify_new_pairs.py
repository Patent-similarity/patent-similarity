"""
STATUS: Secondary/exploratory eval-pair set, NOT the primary evaluation
set. The primary evaluation set (S1-S3/D1-D3) is owned by the Phase 2/3
partner and already validated with real results against the corpus --
see his findings/eval documentation for the authoritative set.

This file documents an alternative pairing explored during Phase 1
reconciliation, kept for potential future use as an additional
sanity-check set, per partner's explicit suggestion.
"""

import sqlite3

# Fresh eval pairs, independently selected by concept search (see
# search_concepts.py), not browsed from an already-built corpus sample.
#
# Similar pair: both are person-tracking systems, differing in target
# granularity (eyes vs. whole body) -- a real, defensible conceptual
# overlap without being a trivial duplicate.
#   8929589 -- "Systems and methods for high-resolution gaze tracking"
#   8970487 -- "Human tracking system"
#
# Dissimilar control: digital watermarking is an unrelated CV problem
# domain from tracking -- no meaningful technique or purpose overlap.
#   9105091 -- "Watermark detection using a propagation map"

eval_ids = [8929589, 8970487, 9105091]

conn = sqlite3.connect("../../data/patents.db")
cursor = conn.cursor()

placeholders = ",".join("?" * len(eval_ids))
cursor.execute(f"SELECT patent_id FROM patents WHERE patent_id IN ({placeholders})", eval_ids)
found = [row[0] for row in cursor.fetchall()]
print(f"Found: {len(found)}/{len(eval_ids)}")
print("Missing:", set(eval_ids) - set(found))

conn.close()