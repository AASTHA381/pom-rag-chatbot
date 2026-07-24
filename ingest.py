"""
Ingestion pipeline for the bundled sample notes: read the Operations
Management notes, split them into overlapping chunks, embed each chunk
locally, and store everything in a FAISS index for fast similarity search.

Run once (and again whenever the notes change):

    python ingest.py
"""
import json

import faiss
from sentence_transformers import SentenceTransformer

import config
import textutils


def build_index() -> None:
    files = sorted(config.DATA_DIR.glob("*.md"))
    if not files:
        raise SystemExit(f"No .md files found in {config.DATA_DIR}")

    sources = [(p.name, p.read_text(encoding="utf-8")) for p in files]
    records = textutils.make_chunks(sources)

    print(f"Read {len(files)} files -> {len(records)} chunks")

    print(f"Loading embedding model: {config.EMBED_MODEL}")
    model = SentenceTransformer(config.EMBED_MODEL)
    embeddings = model.encode(
        [r["text"] for r in records],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    config.INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(config.INDEX_FILE))
    config.CHUNKS_FILE.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    print(f"Saved index -> {config.INDEX_FILE}")
    print(f"Saved chunks -> {config.CHUNKS_FILE}")


if __name__ == "__main__":
    build_index()

