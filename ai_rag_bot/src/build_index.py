import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "embeddings/faiss_index/faiss_index.bin"
METADATA_PATH = "embeddings/faiss_index/metadata.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

def add_to_index(chunks):
    # Load existing index and metadata
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(384)  # 384 = dimension of MiniLM embeddings
        metadata = []

    # Load embedder
    model = SentenceTransformer(EMBED_MODEL)

    # Embed new chunks
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts)
    vectors = np.array(vectors).astype("float32")

    # Add to index + update metadata
    index.add(vectors)
    metadata.extend([chunk["id"] for chunk in chunks])

    # Save
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
