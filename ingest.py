"""
Ingestion pipeline: read the Operations Management notes, split them into
overlapping chunks, embed each chunk locally, and store everything in a FAISS
index for fast similarity search.

Run once (and again whenever the notes change):

    python ingest.py
"""
import json
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config


def clean_markdown(text: str) -> str:
    """Strip noisy OCR artefacts so chunks carry mostly useful text."""
    # Drop the "Start/End of picture text" wrappers that surround OCR'd images.
    text = re.sub(r"<!--\s*(Start|End) of picture text\s*-->", "", text)
    # Collapse 3+ blank lines into a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping, paragraph-aware character windows."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            # Carry the tail of the previous chunk for context continuity.
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
            # A single very long paragraph is hard-split.
            while len(current) > size:
                chunks.append(current[:size])
                current = current[size - overlap:]

    if current:
        chunks.append(current)
    return chunks


def build_index() -> None:
    files = sorted(config.DATA_DIR.glob("*.md"))
    if not files:
        raise SystemExit(f"No .md files found in {config.DATA_DIR}")

    records: list[dict] = []
    for path in files:
        raw = clean_markdown(path.read_text(encoding="utf-8"))
        for i, chunk in enumerate(chunk_text(raw, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            records.append(
                {"source": path.name, "chunk_id": i, "text": chunk}
            )

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
