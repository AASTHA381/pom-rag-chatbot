# 📄 AskDocs — chat with your documents

Upload your own documents (PDF, TXT, Markdown, CSV) and ask questions about
them. **AskDocs** retrieves the most relevant passages and answers with a hosted
LLM, citing its sources. Embeddings run locally (fast on Apple Silicon);
generation runs on Groq's fast hosted API. A bundled set of Operations
Management notes is available as a one-click sample.

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no key)
- **Vector store:** FAISS (built in-memory, per session)
- **LLM:** [Groq](https://groq.com) — default `llama-3.3-70b-versatile` (fast, free tier)
- **UI:** Streamlit chat

> 📋 Full product spec: [docs/PRD.md](docs/PRD.md)

## Architecture

```mermaid
flowchart TB
    subgraph Client["🌐 Browser (per user session)"]
        UI["Streamlit UI<br/>app.py"]
    end

    subgraph Server["☁️ Streamlit Community Cloud"]
        TU["textutils.py<br/>extract · clean · chunk"]
        RAG["rag.py<br/>embed · index · search · generate"]
        EMB["SentenceTransformer<br/>all-MiniLM-L6-v2 (local)"]
        FAISS[("FAISS index<br/>in-memory, per session")]
    end

    subgraph External["🔌 External API"]
        GROQ["Groq LLM<br/>llama-3.3-70b-versatile"]
    end

    UI -->|upload files| TU
    TU -->|chunks| RAG
    RAG -->|encode| EMB
    EMB -->|vectors| FAISS
    UI -->|question| RAG
    RAG -->|top-k context| GROQ
    GROQ -->|grounded answer| UI
```

## Data / processing flow

```mermaid
flowchart LR
    A["📄 Upload<br/>PDF / TXT / MD / CSV"] --> B["Extract text<br/>(pypdf / decode)"]
    B --> C["Clean<br/>(strip artefacts)"]
    C --> D["Chunk<br/>~900 chars, 150 overlap"]
    D --> E["Embed locally<br/>MiniLM-L6-v2"]
    E --> F[("FAISS index<br/>in-memory")]

    Q["❓ User question"] --> QE["Embed question"]
    QE --> S["Similarity search<br/>top-k"]
    F --> S
    S --> P["Build prompt<br/>context + question"]
    P --> LLM["Groq LLM"]
    LLM --> ANS["✅ Answer + sources"]
```

## Sequence (ask a question)

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI (app.py)
    participant TU as textutils.py
    participant RAG as rag.py
    participant EMB as Embedder (local)
    participant IDX as FAISS (session)
    participant GROQ as Groq LLM

    User->>UI: Upload files + "Build knowledge base"
    UI->>TU: extract_text() + make_chunks()
    TU-->>UI: chunk records
    UI->>RAG: build_index(chunks)
    RAG->>EMB: embed(chunk texts)
    EMB-->>RAG: vectors
    RAG->>IDX: add(vectors)
    IDX-->>UI: index ready (stored in session)

    User->>UI: Ask a question
    UI->>RAG: search(index, chunks, question, top_k)
    RAG->>EMB: embed(question)
    EMB-->>RAG: query vector
    RAG->>IDX: search(query, top_k)
    IDX-->>RAG: top-k chunks + scores
    UI->>RAG: generate(question, context)
    RAG->>GROQ: chat.completions (system + context + question)
    GROQ-->>RAG: answer text
    RAG-->>UI: answer
    UI-->>User: Answer + cited sources
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

## Run

```bash
streamlit run app.py
```

Then in the app: upload your files → **Build knowledge base** → ask questions.
No pre-build step is needed — the index is created from your uploads at runtime.

**Optional — bundled sample:** click *"Try the sample"* in the sidebar to chat
with the included Operations Management notes. To use them from the CLI you can
pre-build a disk index once with `python ingest.py`, then:
```bash
python rag.py "What are the four sources of process variation?"
```

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub (public).
2. On https://share.streamlit.io → **Create app** → pick this repo, branch
   `main`, main file `app.py`.
3. In **Advanced settings → Secrets**, add your key in **TOML** format
   (note the quotes — different from `.env`):
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```
4. **Deploy.** You get a permanent public `*.streamlit.app` link to share.

> The `.env` file is git-ignored and never deployed, so on the cloud the key
> **must** come from Streamlit Secrets. Locally it comes from `.env`.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Paths, model names, chunk & retrieval settings |
| `textutils.py` | Extract text (PDF/TXT/MD/CSV), clean & chunk |
| `rag.py` | Embed, build in-memory index, search, generate via Groq |
| `ingest.py` | Build an on-disk index of the bundled sample notes (optional) |
| `app.py` | Streamlit upload + chat interface |
| `data/` | Bundled sample notes (Operations Management) |
| `index/` | Generated sample index (git-ignored) |

## Tuning

Edit `config.py`:
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — how documents are split
- `TOP_K` — how many chunks are fed to the model
- `LLM_MODEL` — swap in any Groq model (e.g. `llama-3.1-8b-instant`)
