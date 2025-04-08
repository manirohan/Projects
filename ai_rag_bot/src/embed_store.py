# src/embed_store.py

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle

CHUNK_DIR = "data/processed_chunks"
INDEX_DIR = "embeddings/faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

def load_chunks():
    chunks = []
    metadata = []
    for path in Path(CHUNK_DIR).glob("*.txt"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                chunks.append(text)
                metadata.append(str(path.name))  # Save filename as metadata
    return chunks, metadata

def build_faiss_index(chunks, metadata, model_name="all-MiniLM-L6-v2"):
    print("🔍 Loading embedding model...")
    model = SentenceTransformer(model_name)

    print("🧠 Encoding chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Save FAISS index
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss_index.bin"))

    # Save metadata
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Stored {len(chunks)} chunks in FAISS index")

if __name__ == "__main__":
    chunks, metadata = load_chunks()
    build_faiss_index(chunks, metadata)
