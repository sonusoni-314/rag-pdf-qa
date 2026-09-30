"""
Module 12: Web UI (Streamlit)

Thin UI layer over the same pipeline functions the CLI uses (Module 11).
No RAG logic lives here — just upload handling, a chat box, and displaying
the answer + citations that pipeline.py already produces.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st

from config import UPLOAD_DIR
from src.pipeline import ingest_pdf, answer_question
from src.vector_store import VectorStore

st.set_page_config(page_title="RAG PDF Q&A", page_icon="📄", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --ink: #101216;
    --panel: #181b21;
    --panel-soft: #20242c;
    --line: #303641;
    --paper: #f5f1e8;
    --muted: #9ba3af;
    --amber: #f2a93b;
    --teal: #56c4b1;
}

.stApp {
    background: var(--ink);
    color: var(--paper);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: #15181d;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.4rem;
}

h1, h2, h3, [data-testid="stMarkdownContainer"] strong {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0;
}

h1 {
    font-size: clamp(2.2rem, 4vw, 4.6rem) !important;
    line-height: 0.98 !important;
    max-width: 780px;
    margin: 0.5rem 0 0.8rem !important;
}

[data-testid="stCaptionContainer"] {
    color: var(--muted);
    font-size: 0.96rem;
}

.brand-kicker {
    color: var(--amber);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 1.5rem;
}

.brand-rule {
    width: 54px;
    height: 4px;
    background: var(--amber);
    margin: 1.5rem 0 1.8rem;
}

.sidebar-brand {
    color: var(--paper);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1.05;
}

.sidebar-note {
    color: var(--muted);
    font-size: 0.83rem;
    line-height: 1.45;
    margin: 0.55rem 0 1.8rem;
}

[data-testid="stFileUploader"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.35rem;
}

[data-testid="stChatMessage"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    margin: 0.9rem 0;
    padding: 1.15rem 1.25rem;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: var(--paper);
    font-size: 1rem;
    line-height: 1.7;
}

[data-testid="stChatMessage"] ul {
    margin-bottom: 0.35rem;
}

[data-testid="stChatMessage"] code {
    color: var(--amber);
    background: #29251d;
}

[data-testid="stChatInput"] {
    border-color: var(--line);
}

[data-testid="stChatInput"] textarea {
    background: var(--panel);
    color: var(--paper);
    font-family: 'DM Sans', sans-serif;
}

.doc-count {
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 1.4rem;
    padding-top: 1rem;
}

.empty-state {
    border: 1px dashed var(--line);
    border-radius: 10px;
    color: var(--muted);
    margin-top: 2.5rem;
    padding: 2rem;
    text-align: center;
}

@media (max-width: 700px) {
    h1 { font-size: 2.5rem !important; }
    [data-testid="stChatMessage"] { padding: 0.95rem; }
}
</style>
""", unsafe_allow_html=True)

# Vector store + chat history persist across reruns within one browser session
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

st.markdown('<div class="brand-kicker">Private document intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-rule"></div>', unsafe_allow_html=True)
st.title("RAG PDF Q&A")
st.caption("Search your documents locally and get answers anchored to the pages that support them.")

# ---------- Sidebar: upload + ingest ----------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Document desk</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-note">Build a searchable workspace from your PDFs. Your files stay on this machine.</div>', unsafe_allow_html=True)
    st.header("Add a document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name not in st.session_state.ingested_files:
            with st.spinner(f"Processing {uploaded_file.name}... (classify → extract → chunk → embed)"):
                n_chunks = ingest_pdf(save_path, st.session_state.vector_store, verbose=False)
            st.session_state.ingested_files.append(uploaded_file.name)
            st.success(f"Ingested {uploaded_file.name} ({n_chunks} chunks)")
        else:
            st.info(f"{uploaded_file.name} already ingested")

    if st.session_state.ingested_files:
        st.subheader("Ingested documents")
        for fname in st.session_state.ingested_files:
            st.write(f"- {fname}")

    st.markdown(
        f'<div class="doc-count">{st.session_state.vector_store.count()} searchable chunks</div>',
        unsafe_allow_html=True,
    )

# ---------- Main: chat interface ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your uploaded PDF(s)...")

if not st.session_state.messages and st.session_state.vector_store.count() == 0:
    st.markdown(
        '<div class="empty-state">Upload a PDF to begin a grounded conversation.</div>',
        unsafe_allow_html=True,
    )

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if st.session_state.vector_store.count() == 0:
            answer = "Please upload a PDF first — I don't have any documents to search yet."
        else:
            with st.spinner("Retrieving relevant chunks and generating an answer..."):
                answer = answer_question(question, st.session_state.vector_store)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
