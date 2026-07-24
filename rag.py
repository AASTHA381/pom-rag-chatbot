"""
Retrieval-Augmented Generation core.

Loads the FAISS index built by ingest.py, retrieves the most relevant note
chunks for a question, and asks a Groq-hosted LLM to answer using only
that context. Requires the GROQ_API_KEY environment variable.
"""
import json
import os
from functools import lru_cache

import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

import config

load_dotenv()  # read GROQ_API_KEY from a .env file if present

SYSTEM_PROMPT = (
    "You are a study assistant for an Operations Management course. "
    "Answer the student's question using ONLY the context provided below. "
    "If the answer is not in the context, say you don't have that in the notes. "
    "Be concise, use bullet points where helpful, and do not invent facts."
)


@lru_cache(maxsize=1)
def _load():
    """Load and cache the embedding model, FAISS index, and chunk metadata."""
    model = SentenceTransformer(config.EMBED_MODEL)
    index = faiss.read_index(str(config.INDEX_FILE))
    chunks = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))
    return model, index, chunks


def retrieve(question: str, top_k: int = config.TOP_K) -> list[dict]:
    """Return the top_k most similar chunks with their similarity score."""
    model, index, chunks = _load()
    q_emb = model.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        record = dict(chunks[idx])
        record["score"] = float(score)
        results.append(record)
    return results


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(question: str, top_k: int = config.TOP_K) -> dict:
    """Retrieve context and generate an answer. Returns answer + sources."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and add it to a .env file (see .env.example) or export it."
        )
    chunks = retrieve(question, top_k)
    prompt = build_prompt(question, chunks)
    client = Groq()  # reads GROQ_API_KEY from the environment
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": chunks,
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is operations management?"
    print(f"Q: {q}\n")
    result = answer(q)
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['source']} (score {s['score']:.2f})")
