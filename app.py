"""
Streamlit chat UI for the Operations Management RAG chatbot.

Run:  streamlit run app.py
"""
import streamlit as st

import config
import rag

st.set_page_config(page_title="Ops Management Study Bot", page_icon="📘")

# On a fresh deployment (e.g. Streamlit Cloud) the FAISS index won't exist yet.
# Build it once from the notes in data/ before serving any questions.
@st.cache_resource(show_spinner="Building search index from notes…")
def _ensure_index():
    if not config.INDEX_FILE.exists() or not config.CHUNKS_FILE.exists():
        import ingest

        ingest.build_index()
    return True


_ensure_index()

st.title("📘 Operations Management Study Bot")
st.caption("Answers grounded in your course notes — local embeddings + Groq LLM.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, config.TOP_K)
    st.markdown(f"**Embedding:** `{config.EMBED_MODEL.split('/')[-1]}`")
    st.markdown(f"**LLM (Groq):** `{config.LLM_MODEL}`")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}** (score {s['score']:.2f})")

# Handle a new question.
if prompt := st.chat_input("Ask about operations management…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = rag.answer(prompt, top_k=top_k)
                st.markdown(result["answer"])
                with st.expander("Sources"):
                    for s in result["sources"]:
                        st.markdown(f"- **{s['source']}** (score {s['score']:.2f})")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )
            except FileNotFoundError:
                st.error("Index not found. Run `python ingest.py` first.")
            except Exception as exc:  # noqa: BLE001 - surface any runtime error to the user
                st.error(f"Error: {exc}\n\nIs your GROQ_API_KEY set? See `.env.example`.")
