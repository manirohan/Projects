# src/rag_bot.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

client = OpenAI()  # Automatically picks up OPENAI_API_KEY from env


# ==== CONFIG ====
INDEX_PATH = "embeddings/faiss_index/faiss_index.bin"
METADATA_PATH = "embeddings/faiss_index/metadata.pkl"
CHUNK_DIR = "data/processed_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4"  # or "gpt-3.5-turbo"

# ==== LOAD EVERYTHING ====
print("📦 Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

model = SentenceTransformer(EMBED_MODEL)

# ==== RETRIEVAL FUNCTION ====
def retrieve_relevant_chunks(query, k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)

    retrieved_chunks = []
    for i in I[0]:
        fname = metadata[i]
        with open(f"{CHUNK_DIR}/{fname}", "r", encoding="utf-8") as f:
            retrieved_chunks.append(f.read())
    return retrieved_chunks

# ==== GENERATE ANSWER ====
def generate_answer(query, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""
You are an expert AI assistant trained on AI research papers.

Answer the following question using ONLY the context provided below.
If the answer is not in the context, say "I don't know based on the given information."

Context:
{context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# ==== MAIN CHAT LOOP ====
def chat():
    print("\n💬 Ask me something about AI research papers! (type 'exit' to quit)\n")
    while True:
        query = input("🧠 You: ")
        if query.lower() in ["exit", "quit"]:
            break
        chunks = retrieve_relevant_chunks(query)
        answer = generate_answer(query, chunks)
        print(f"\n🤖 RAGBot: {answer}\n")

if __name__ == "__main__":
    chat()
