"""
Retrieval-Augmented Generation core.

Reusable building blocks used by the Streamlit app:
  - get_embedder():      cached local embedding model
  - embed(texts):        encode text -> normalized float32 vectors
  - build_index(chunks): build an in-memory FAISS index from chunk records
  - search(...):         retrieve top-k chunks for a question
  - generate(...):       ask the Groq LLM to answer from context

Plus disk-index helpers (retrieve / answer) used by the bundled sample notes
and the CLI. Requires the GROQ_API_KEY environment variable for generation.
"""
import json
import os
from functools import lru_cache

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

import config

load_dotenv()  # read GROQ_API_KEY from a .env file if present

SYSTEM_PROMPT = (
    "You are a helpful study/research assistant. "
    "Answer the user's question using ONLY the context provided below. "
    "If the answer is not in the context, say you don't have that in the "
    "provided documents. Be concise, use bullet points where helpful, and "
    "do not invent facts."
)


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load and cache the local embedding model (shared across sessions)."""
    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    return get_embedder().encode(
        texts, normalize_embeddings=True
    ).astype("float32")


# --------------------------------------------------------------------------
# In-memory index (used for user-uploaded documents)
# --------------------------------------------------------------------------
def build_index(chunks: list[dict]):
    """Build a FAISS index from chunk records (each dict has a 'text' field)."""
    embeddings = embed([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search(index, chunks: list[dict], question: str,
           top_k: int = config.TOP_K) -> list[dict]:
    """Return the top_k most similar chunks with their similarity score."""
    q_emb = embed([question])
    scores, ids = index.search(q_emb, min(top_k, len(chunks)))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        record = dict(chunks[idx])
        record["score"] = float(score)
        results.append(record)
    return results


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def build_prompt(question: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate(question: str, context_chunks: list[dict]) -> str:
    """Ask the Groq LLM to answer using the given context chunks."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and add it to .env (local) or "
            "Streamlit secrets (cloud)."
        )
    client = Groq()  # reads GROQ_API_KEY from the environment
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, context_chunks)},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# --------------------------------------------------------------------------
# Disk-index helpers (bundled sample notes + CLI)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_disk():
    index = faiss.read_index(str(config.INDEX_FILE))
    chunks = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))
    return index, chunks


def retrieve(question: str, top_k: int = config.TOP_K) -> list[dict]:
    index, chunks = _load_disk()
    return search(index, chunks, question, top_k)


def answer(question: str, top_k: int = config.TOP_K) -> dict:
    """Retrieve from the on-disk sample index and generate an answer."""
    chunks = retrieve(question, top_k)
    return {"answer": generate(question, chunks), "sources": chunks}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is operations management?"
    print(f"Q: {q}\n")
    result = answer(q)
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['source']} (score {s['score']:.2f})")
