# src/data_pipeline.py

import os
import requests
import xml.etree.ElementTree as ET

ARXIV_API_URL = "http://export.arxiv.org/api/query"
SAVE_DIR = "data/raw_papers"

def fetch_arxiv_papers(query="cs.AI", max_results=50):
    os.makedirs(SAVE_DIR, exist_ok=True)

    params = {
        "search_query": f"cat:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    response = requests.get(ARXIV_API_URL, params=params)
    root = ET.fromstring(response.content)

    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entries = root.findall('atom:entry', ns)

    for i, entry in enumerate(entries):
        title = entry.find('atom:title', ns).text.strip()
        abstract = entry.find('atom:summary', ns).text.strip()
        paper_id = entry.find('atom:id', ns).text.split('/')[-1]
        authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]

        paper = {
            "id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors
        }

        with open(os.path.join(SAVE_DIR, f"{paper_id}.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(paper, f, indent=2)

        print(f"✅ Saved paper {i+1}: {title[:60]}...")

if __name__ == "__main__":
    fetch_arxiv_papers(query="cs.AI", max_results=25)



import json
import re
from pathlib import Path
from textwrap import wrap

PROCESSED_DIR = "data/processed_chunks"
CHUNK_SIZE = 500  # Approximate number of tokens per chunk

def clean_text(text):
    # Basic cleanup: remove newlines, LaTeX, extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\$.*?\$', '', text)  # Remove inline LaTeX
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE):
    # Simple character-based splitting
    sentences = wrap(text, width=chunk_size * 5)  # Approx. chars to match token size
    return sentences

def process_and_chunk_papers():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    raw_files = Path(SAVE_DIR).glob("*.json")

    for file_path in raw_files:
        with open(file_path, "r", encoding="utf-8") as f:
            paper = json.load(f)

        abstract = clean_text(paper.get("abstract", ""))
        chunks = chunk_text(abstract)

        for i, chunk in enumerate(chunks):
            chunk_file = f"{paper['id']}_chunk_{i}.txt"
            with open(os.path.join(PROCESSED_DIR, chunk_file), "w", encoding="utf-8") as f:
                f.write(chunk)

        print(f"✅ Processed & chunked {paper['id']} into {len(chunks)} chunks")

if __name__ == "__main__":
    fetch_arxiv_papers(query="cs.AI", max_results=25)
    process_and_chunk_papers()
