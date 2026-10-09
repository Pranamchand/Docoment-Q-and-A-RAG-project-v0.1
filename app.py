"""
Streamlit interface for the PDF-based RAG application.
Reuses embedding model, LLM, prompt, and retriever settings from the project
without importing main.py (which runs a CLI loop at import time).
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# ── Environment ─────────────────────────────────────────────────────────────
load_dotenv()

_REQUIRED_KEY = "OPENROUTER_API_KEY"
_missing_key = not os.getenv(_REQUIRED_KEY)

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Book Q&A",
    page_icon="📖",
    layout="centered",
)

# ── Custom CSS — minimalist, restrained palette ─────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    /* ---- globals ---- */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* hide default Streamlit header / footer for a cleaner look */
    #MainMenu, header, footer {visibility: hidden;}

    /* ---- main container ---- */
    .block-container {
        max-width: 740px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }

    /* ---- header area ---- */
    .app-header {
        text-align: center;
        margin-bottom: 1.8rem;
    }
    .app-header h1 {
        font-weight: 600;
        font-size: 1.65rem;
        color: #ffffff;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .app-header p {
        font-size: 0.92rem;
        color: #6b7280;
        margin: 0;
    }

    /* ---- chat messages ---- */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.6rem !important;
        font-size: 0.95rem;
    }

    /* ---- file uploader ---- */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    /* ---- divider ---- */
    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1.4rem 0;
    }

    /* ---- status pill ---- */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .status-ready {
        background: #ecfdf5;
        color: #047857;
    }
    .status-waiting {
        background: #f3f4f6;
        color: #6b7280;
    }

    /* ---- upload section ---- */
    .upload-section {
        border: 1.5px dashed #d1d5db;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
        transition: border-color 0.2s;
    }
    .upload-section:hover {
        border-color: #9ca3af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-header">'
    "<h1>📖 Book Q&A</h1>"
    "<p>Upload a PDF book, then ask questions about it.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Early exit if API key is missing ────────────────────────────────────────
if _missing_key:
    st.error(
        f"**{_REQUIRED_KEY}** is not set.  \n"
        "Add it to your `.env` file and restart the app.",
        icon="🔑",
    )
    st.stop()


# ── Lazy-loaded heavy dependencies (cached once per process) ────────────────
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(model="openrouter/free")


@st.cache_resource(show_spinner=False)
def get_prompt_template():
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful AI assistant.\n"
                    "Use ONLY the provided context to answer the question.\n"
                    "If the answer is not present in the context, "
                    'say: "I could not find the answer in the uploaded book."'
                ),
            ),
            (
                "human",
                "Context: {context}\n\nQuestion: {question}",
            ),
        ]
    )


# ── PDF processing (one vectorstore per file, kept in session state) ────────
def build_vectorstore(uploaded_file):
    """Parse a PDF, chunk it, embed it, and return an in-memory Chroma store."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    # Write the uploaded bytes to a temp file so PyPDFLoader can read it.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
        if not docs:
            raise ValueError("The PDF contained no extractable text.")
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)

    # In-memory collection (no persist_directory) — avoids touching chroma_DB.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
    )
    return vectorstore


# ── Session-state defaults ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "book_name" not in st.session_state:
    st.session_state.book_name = None

# ── File upload — inline in main area ───────────────────────────────────────
_book_loaded = st.session_state.book_name is not None

if _book_loaded:
    # Compact status bar when a book is already loaded
    col_status, col_new, col_clear = st.columns([3, 1.2, 1.2])
    with col_status:
        st.markdown(
            f'<span class="status-pill status-ready">📗 {st.session_state.book_name}</span>',
            unsafe_allow_html=True,
        )
    with col_new:
        if st.button("New book", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.book_name = None
            st.session_state.messages = []
            st.rerun()
    with col_clear:
        if st.session_state.messages and st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    st.markdown("")  # spacer
else:
    # Full uploader when no book is loaded
    uploaded = st.file_uploader(
        "Upload a PDF book",
        type=["pdf"],
        label_visibility="visible",
    )

    if uploaded is not None:
        if uploaded.name != st.session_state.book_name:
            try:
                with st.spinner("Processing PDF — this may take a moment…"):
                    st.session_state.vectorstore = build_vectorstore(uploaded)
                    st.session_state.book_name = uploaded.name
                    st.session_state.messages = []
                    st.rerun()
            except Exception as exc:
                st.error(f"Failed to process **{uploaded.name}**: {exc}", icon="⚠️")
                st.session_state.vectorstore = None
                st.session_state.book_name = None

# ── Chat history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ──────────────────────────────────────────────────────────────
if prompt_text := st.chat_input("Ask a question about your book…"):
    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF first.", icon="📄")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Retrieve & generate
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 2, "fetch_k": 5, "lambda_mult": 0.5},
                )
                docs = retriever.invoke(prompt_text)
                context = "\n\n".join(doc.page_content for doc in docs)

                prompt_template = get_prompt_template()
                final_prompt = prompt_template.invoke(
                    {"context": context, "question": prompt_text}
                )

                response = get_llm().invoke(final_prompt)
                answer = response.text if hasattr(response, "text") else str(response.content)
            except Exception as exc:
                answer = f"⚠️ An error occurred: {exc}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
