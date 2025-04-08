import os
import uuid
from typing import List, Dict
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from src.chunk_utils import chunk_text  # if you use a helper

CHUNK_DIR = "data/processed_chunks"

def preprocess_document(file_path: str) -> List[Dict]:
    # Step 1: Read PDF
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    # Step 2: Split into chunks
    chunks = chunk_text(full_text, max_length=300, overlap=50)  # you can tweak

    # Step 3: Save chunks to disk and return list
    os.makedirs(CHUNK_DIR, exist_ok=True)
    chunk_dicts = []
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())[:8] + ".txt"
        with open(f"{CHUNK_DIR}/{chunk_id}", "w", encoding="utf-8") as f:
            f.write(chunk)
        chunk_dicts.append({
            "id": chunk_id,
            "text": chunk
        })

    return chunk_dicts
