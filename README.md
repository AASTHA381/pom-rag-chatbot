# 📘 Operations Management RAG Chatbot

A local-embeddings + hosted-LLM Retrieval-Augmented Generation (RAG) chatbot
that answers questions about your Operations Management course notes.
Optimised for Apple Silicon (M2) — embeddings run locally, generation runs on
Groq's fast hosted API.

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no key)
- **Vector store:** FAISS (on disk)
- **LLM:** [Groq](https://groq.com) — default `llama-3.3-70b-versatile` (fast, free tier)
- **UI:** Streamlit chat

## How it works

```
notes (data/*.md)
   │  ingest.py
   ▼
clean → chunk → embed → FAISS index (index/)
   │
   ▼  rag.py
question → embed → search top-k chunks → prompt Groq LLM → grounded answer
   │
   ▼  app.py
Streamlit chat UI (with sources)
```

## Setup

1. **Get a free Groq API key** at https://console.groq.com/keys, then:
   ```bash
   cd /tmp/pom-rag-chatbot
   cp .env.example .env      # then paste your key into .env
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Build the index** from the notes in `data/`:
   ```bash
   python ingest.py
   ```

## Run

**Web chat (recommended):**
```bash
streamlit run app.py
```

**Quick terminal test:**
```bash
python rag.py "What are the four sources of process variation?"
```

## Adding / changing notes

Drop more `.md` (or paste text) files into `data/`, then re-run
`python ingest.py` to rebuild the index.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Paths, model names, chunk & retrieval settings |
| `ingest.py` | Clean → chunk → embed → build FAISS index |
| `rag.py` | Retrieve context + generate answer via Groq |
| `app.py` | Streamlit chat interface |
| `data/` | Your source course notes |
| `index/` | Generated FAISS index + chunk metadata (git-ignored) |

## Tuning

Edit `config.py`:
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — how notes are split
- `TOP_K` — how many chunks are fed to the model
- `LLM_MODEL` — swap in any Groq model (e.g. `llama-3.1-8b-instant`)
