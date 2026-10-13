import json
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "chat_history.json")


def render_source_cards(sources):
    if not sources:
        return
    st.markdown("**Sources cited**")
    for idx, item in enumerate(sources, start=1):
        source_name = item.get("source", "Unknown document")
        page_number = item.get("page", 1)
        snippet = item.get("snippet", "")
        st.markdown(
            f"""
            <div style="padding: 0.8rem 0.9rem; border: 1px solid #e6eaf1; border-radius: 10px; background: #f8fafc; margin-bottom: 0.6rem;">
                <div style="font-weight: 700; margin-bottom: 0.3rem;">{idx}. {source_name}</div>
                <div style="font-size: 0.8rem; color: #475569; margin-bottom: 0.4rem;">Page {page_number}</div>
                <div style="font-size: 0.9rem; color: #1f2937;">{snippet}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def load_chat_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_chat_history(messages):
    with open(HISTORY_PATH, "w", encoding="utf-8") as handle:
        json.dump(messages, handle, ensure_ascii=False, indent=2)


st.set_page_config(
    page_title="Offline Document RAG",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #0b1220;
        --panel: #111827;
        --panel-2: #1f2937;
        --border: #334155;
        --text: #e5e7eb;
        --muted: #94a3b8;
        --accent: #60a5fa;
        --accent-2: #34d399;
        --warning: #fbbf24;
    }
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
        color: var(--text);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.35);
    }
    .hero h1 {
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .hero p {
        color: var(--muted);
        margin-bottom: 0;
    }
    .metric-card {
        background: rgba(17, 24, 39, 0.9);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        min-height: 90px;
    }
    .metric-card .label {
        color: var(--muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-card .value {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }
    .metric-card .delta {
        color: var(--accent-2);
        font-size: 0.72rem;
        margin-top: 0.3rem;
    }
    .sidebar .stFileUploader, .sidebar .stButton > button, .sidebar .stForm {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid var(--border);
        border-radius: 12px;
    }
    .source-card {
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.6rem;
    }
    .source-card .title {
        font-weight: 700;
        color: #f8fafc;
    }
    .source-card .meta {
        color: var(--muted);
        font-size: 0.78rem;
        margin: 0.2rem 0 0.4rem;
    }
    .source-card .snippet {
        color: #dbeafe;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>📄 Offline Privacy-First Document RAG</h1>
        <p>Local-first document Q&A powered by FastAPI, ChromaDB, and Qwen2.5-0.5B-Instruct on CPU.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card"><div class="label">Backend</div><div class="value">FastAPI</div><div class="delta">Ready</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="label">Vector store</div><div class="value">ChromaDB</div><div class="delta">Persistent</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="label">Inference</div><div class="value">CPU</div><div class="delta">Offline</div></div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

if "last_upload_summary" not in st.session_state:
    st.session_state.last_upload_summary = None

with st.sidebar:
    st.header("📂 Document Ingestion")
    st.markdown("Upload one or more PDFs, index them locally, and ask grounded questions across the full set.")

    uploaded_files = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("Ingest PDFs", type="primary", disabled=not uploaded_files):
        with st.spinner("Processing and indexing PDF collection locally..."):
            try:
                upload_payload = [
                    (file.name, file.getvalue(), "application/pdf") for file in uploaded_files
                ]
                files = [
                    ("files", (name, content, "application/pdf")) for name, content, _ in upload_payload
                ]
                response = requests.post(f"{BACKEND_URL}/upload-pdfs", files=files, timeout=180)

                if response.status_code == 200:
                    data = response.json()
                    st.session_state.last_upload_summary = {
                        "count": len(data.get("files_uploaded", [])),
                        "chunks": data.get("chunks_added", 0),
                        "files": data.get("files", []),
                    }
                    st.success(
                        f"Successfully indexed **{len(data.get('files_uploaded', []))}** PDFs ({data.get('chunks_added')} chunks added)."
                    )
                    if st.session_state.last_upload_summary["files"]:
                        st.caption("Indexed files")
                        for item in st.session_state.last_upload_summary["files"]:
                            metadata = item.get("metadata", {})
                            if metadata:
                                st.code(
                                    f"{item.get('filename', 'document')}\n{json.dumps(metadata, ensure_ascii=False, indent=2)}",
                                    language="text",
                                )
                else:
                    error_detail = response.json().get("detail", response.text)
                    st.error(f"Upload failed: {error_detail}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Connection failed: ensure the FastAPI backend is running. Details: {exc}")

    st.markdown("---")
    st.markdown("**System Status:** 100% Offline 🟢")
    if st.session_state.last_upload_summary:
        with st.container():
            st.markdown("**Latest upload summary**")
            st.caption(f"{st.session_state.last_upload_summary['count']} PDFs indexed")
            st.caption(f"{st.session_state.last_upload_summary['chunks']} chunks added")

    if st.button("Clear chat history"):
        st.session_state.messages = []
        save_chat_history([])

    example_prompts = [
        "Summarize this document set in simple terms.",
        "What are the main findings across all uploaded PDFs?",
        "List the key highlights and action items.",
    ]
    st.markdown("**Try these prompts:**")
    for prompt in example_prompts:
        if st.button(prompt, key=f"example_{prompt}"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Generating a grounded local response..."):
                    try:
                        res = requests.post(f"{BACKEND_URL}/query", json={"question": prompt}, timeout=180)
                        if res.status_code == 200:
                            payload = res.json()
                            answer = payload.get("answer", "No answer returned.")
                            st.markdown(answer)
                            sources = payload.get("sources", [])
                            if sources:
                                with st.expander("Citations"):
                                    render_source_cards(sources)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            save_chat_history(st.session_state.messages)
                        else:
                            error_msg = res.json().get("detail", res.text)
                            st.error(f"Server error: {error_msg}")
                            st.session_state.messages.append({"role": "assistant", "content": f"Server error: {error_msg}"})
                            save_chat_history(st.session_state.messages)
                    except requests.exceptions.RequestException as exc:
                        error_msg = f"Failed to connect to backend server at `{BACKEND_URL}`. Make sure it is running."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        save_chat_history(st.session_state.messages)

st.header("💬 Knowledge Base Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about your uploaded documents...", key="chat_input")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_history(st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating a grounded local response..."):
            try:
                res = requests.post(f"{BACKEND_URL}/query", json={"question": prompt}, timeout=180)
                if res.status_code == 200:
                    payload = res.json()
                    answer = payload.get("answer", "No answer returned.")
                    st.markdown(answer)
                    sources = payload.get("sources", [])
                    if sources:
                        with st.expander("Citations"):
                            render_source_cards(sources)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    save_chat_history(st.session_state.messages)
                else:
                    error_msg = res.json().get("detail", res.text)
                    st.error(f"Server error: {error_msg}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Server error: {error_msg}"})
                    save_chat_history(st.session_state.messages)
            except requests.exceptions.RequestException as exc:
                error_msg = f"Failed to connect to backend server at `{BACKEND_URL}`. Make sure it is running."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                save_chat_history(st.session_state.messages)
