FROM python:3.14-slim

WORKDIR /app

# Install production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src ./src

# Copy only required runtime data
COPY data/toy_patents.py ./data/toy_patents.py
COPY data/patents.db ./data/patents.db
COPY data/real_corpus.py ./data/real_corpus.py
COPY data/__init__.py ./data/__init__.py

# Copy FAISS index and metadata
COPY embeddings/faiss_index ./embeddings/faiss_index

EXPOSE 8000

CMD ["uvicorn", "src.phase4_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]