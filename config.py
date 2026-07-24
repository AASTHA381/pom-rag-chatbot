"""Shared configuration for the POM RAG chatbot."""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
INDEX_FILE = INDEX_DIR / "faiss.index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

# --- Models ---
# Local sentence-transformers model used to embed both chunks and questions.
# Small and fast — runs fine on an Apple Silicon (M2) CPU, no API needed.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Chat model served by Groq's hosted API (very fast, generous free tier).
# Requires a GROQ_API_KEY. Other options: "llama-3.1-8b-instant", "openai/gpt-oss-20b".
LLM_MODEL = "llama-3.3-70b-versatile"

# --- Chunking ---
CHUNK_SIZE = 900        # target characters per chunk
CHUNK_OVERLAP = 150     # characters shared between neighbouring chunks

# --- Retrieval ---
TOP_K = 4               # number of chunks fed to the LLM as context
