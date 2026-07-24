"""
Streamlit chat UI for a "chat with your documents" RAG app.

Users upload their own PDF / TXT / MD files; the app chunks + embeds them into
a private in-session FAISS index, then answers questions grounded in those
documents (via a Groq-hosted LLM). A bundled set of Operations Management notes
is available as a one-click sample.

Run:  streamlit run app.py
"""
import streamlit as st

import config
import rag
import textutils

st.set_page_config(page_title="Chat with your Documents", page_icon="📄")

MAX_FILES = 20
ACCEPTED = ["pdf", "txt", "md", "markdown", "csv"]

# --- session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None      # FAISS index of the current knowledge base
if "chunks" not in st.session_state:
    st.session_state.chunks = []       # chunk records backing the index
if "kb_name" not in st.session_state:
    st.session_state.kb_name = None    # label describing the loaded knowledge base


def load_knowledge_base(sources: list[tuple[str, str]], label: str) -> None:
    """Chunk + embed the given (name, text) sources into a session index."""
    chunks = textutils.make_chunks(sources)
    if not chunks:
        st.sidebar.error("No readable text found in those files.")
        return
    with st.spinner(f"Indexing {len(sources)} document(s) → {len(chunks)} chunks…"):
        st.session_state.index = rag.build_index(chunks)
        st.session_state.chunks = chunks
        st.session_state.kb_name = label
        st.session_state.messages = []
    st.sidebar.success(f"Ready! Indexed {len(chunks)} chunks from {len(sources)} file(s).")


# --------------------------------------------------------------------------
# Sidebar: build the knowledge base
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Your documents")
    uploaded = st.file_uploader(
        "Upload files to chat with",
        type=ACCEPTED,
        accept_multiple_files=True,
        help="PDF, TXT, Markdown or CSV. Everything stays in your session only.",
    )

    if st.button("Build knowledge base", type="primary", disabled=not uploaded):
        if len(uploaded) > MAX_FILES:
            st.error(f"Please upload at most {MAX_FILES} files.")
        else:
            sources = [(f.name, textutils.extract_text(f.name, f.getvalue())) for f in uploaded]
            load_knowledge_base(sources, f"{len(sources)} uploaded file(s)")

    st.divider()
    if st.button("Try the sample (Operations Management notes)"):
        files = sorted(config.DATA_DIR.glob("*.md"))
        sources = [(p.name, p.read_text(encoding="utf-8")) for p in files]
        load_knowledge_base(sources, "Sample: Operations Management notes")

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, config.TOP_K)
    st.caption(f"Embeddings: `{config.EMBED_MODEL.split('/')[-1]}` · LLM: `{config.LLM_MODEL}`")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("📄 Chat with your Documents")

if st.session_state.index is None:
    st.info(
        "👈 Upload your documents (PDF, TXT, MD, CSV) and click **Build knowledge "
        "base** to start — or try the bundled sample. Then ask questions and get "
        "answers grounded in your files, with sources cited."
    )
    st.stop()

st.caption(f"Knowledge base: **{st.session_state.kb_name}** · "
           f"{len(st.session_state.chunks)} chunks")

# Replay history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}** (score {s['score']:.2f})")

# Handle a new question.
if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                sources = rag.search(
                    st.session_state.index, st.session_state.chunks, prompt, top_k
                )
                answer = rag.generate(prompt, sources)
                st.markdown(answer)
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- **{s['source']}** (score {s['score']:.2f})")
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:  # noqa: BLE001 - surface any runtime error to the user
                st.error(f"Error: {exc}\n\nIs the GROQ_API_KEY set (Streamlit secrets / .env)?")
