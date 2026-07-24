"""
Text utilities shared by the ingestion pipeline and the upload flow:
cleaning, paragraph-aware chunking, and extracting text from uploaded files
(PDF, TXT, Markdown).
"""
import io
import re

import config


def clean_text(text: str) -> str:
    """Strip noisy OCR/markdown artefacts so chunks carry mostly useful text."""
    # Drop the "Start/End of picture text" wrappers that surround OCR'd images.
    text = re.sub(r"<!--\s*(Start|End) of picture text\s*-->", "", text)
    # Collapse 3+ blank lines into a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:
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


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from an uploaded file's raw bytes."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    # Treat everything else (txt, md, csv, etc.) as UTF-8 text.
    return data.decode("utf-8", errors="ignore")


def make_chunks(sources: list[tuple[str, str]]) -> list[dict]:
    """
    Turn (source_name, raw_text) pairs into chunk records.

    Returns a list of dicts: {"source", "chunk_id", "text"}.
    """
    records: list[dict] = []
    for name, raw in sources:
        for i, chunk in enumerate(chunk_text(clean_text(raw))):
            records.append({"source": name, "chunk_id": i, "text": chunk})
    return records
