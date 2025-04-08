import streamlit as st
st.set_page_config(page_title="AI Paper RAG Bot", layout="wide")

from pathlib import Path
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ==== Setup ====
client = OpenAI()
EMBED_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4"
CHUNK_DIR = "data/processed_chunks"
INDEX_PATH = "embeddings/faiss_index/faiss_index.bin"
METADATA_PATH = "embeddings/faiss_index/metadata.pkl"

@st.cache_resource
def load_index_and_model():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    model = SentenceTransformer(EMBED_MODEL)
    return index, metadata, model

index, metadata, embedder = load_index_and_model()

# ==== UI ====
st.title("📚 AI Research RAG Bot")
st.markdown("Ask a question based on recent AI papers.")

query = st.text_input("🔍 Your Question:", "")

def retrieve_chunks(query, k=3):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), k)
    chunks = []
    for i in I[0]:
        fname = metadata[i]
        with open(f"{CHUNK_DIR}/{fname}", "r", encoding="utf-8") as f:
            chunks.append((fname, f.read()))
    return chunks

def generate_answer(query, context_chunks):
    context = "\n\n---\n\n".join([chunk[1] for chunk in context_chunks])
    prompt = f"""
You are an expert assistant for AI research.

Use only the context below to answer. If not found, say "I don't know."

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

if query:
    with st.spinner("Retrieving context and generating answer..."):
        context_chunks = retrieve_chunks(query)
        answer = generate_answer(query, context_chunks)

        st.subheader("🤖 Answer")
        st.write(answer)

        with st.expander("📄 Show Retrieved Chunks"):
            for fname, text in context_chunks:
                st.markdown(f"**{fname}**")
                st.markdown(f"> {text}")
