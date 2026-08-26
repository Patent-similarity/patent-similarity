import faiss
import numpy as np
import pickle
import os

# A tiny, structurally-correct FAISS index + patent list, purely to
# prove the startup-loading code works. NOT real embeddings -- random
# vectors standing in for the real 768-dim Gemini embeddings the
# partner's Phase 2 will eventually produce.

DIM = 768  # matches Gemini embedding dimensionality
NUM_FAKE_PATENTS = 5

np.random.seed(42)
fake_vectors = np.random.rand(NUM_FAKE_PATENTS, DIM).astype("float32")

index = faiss.IndexFlatIP(DIM)  # inner product, matches typical similarity search setup
index.add(fake_vectors)

os.makedirs("dummy_index", exist_ok=True)
faiss.write_index(index, "dummy_index/index.faiss")

fake_patents = [
    {"patent_id": f"DUMMY{i}", "title": f"Fake patent {i} for testing"}
    for i in range(NUM_FAKE_PATENTS)
]
with open("dummy_index/patents.pkl", "wb") as f:
    pickle.dump(fake_patents, f)

print(f"Dummy index built: {index.ntotal} vectors, dim {index.d}")
print(f"Dummy patents list: {len(fake_patents)} entries")