"""Streamlit entry point for the live demo (Streamlit Community Cloud runs this file).

Secrets (GROQ_API_KEY, model names, ...) come from Streamlit's secrets store and are copied into
environment variables before the app's config is loaded, so the rest of the code is unchanged.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Polygraph — Agentic RAG with Evals", page_icon="🔄", layout="wide")

try:  # Streamlit Cloud secrets -> env vars (no-op locally if there is no secrets file)
    for key, value in st.secrets.items():
        if isinstance(value, str | int | float | bool):
            os.environ.setdefault(key, str(value))
except Exception:  # noqa: BLE001 - no secrets file: fall back to .env / environment
    pass

from rag_agent.config import get_settings  # noqa: E402
from rag_agent.graph import Agent  # noqa: E402
from rag_agent.ingest import load_pdf  # noqa: E402
from rag_agent.llm import LLM, QuotaExhausted, RateLimited  # noqa: E402
from rag_agent.retrieval import Retriever  # noqa: E402
from rag_agent.ui import (  # noqa: E402
    BUSY_MSG,
    EXAMPLES,
    INTRO,
    MAX_QUESTION_CHARS,
    QUOTA_MSG,
    eval_results_md,
    render_answer,
    render_trace,
)

log = logging.getLogger("streamlit_app")
SESSION_LIMIT = int(os.getenv("SESSION_QUESTION_LIMIT", "15"))


@st.cache_resource(show_spinner="Loading the knowledge base and local models (first visit only)…")
def kb_agent() -> Agent:
    """One shared agent over the prebuilt Docker-docs index (fail-fast retries for the UI)."""
    s = get_settings().with_overrides(max_retries=2)
    retriever = Retriever.from_index(s)
    retriever.retrieve("warm-up query", s)  # load embedder + reranker now, not on the first question
    return Agent(retriever, s, LLM(s))


def ask(agent: Agent, question: str) -> None:
    """Run the agent and render the answer plus its trace."""
    question = (question or "").strip()
    if not question:
        return
    if len(question) > MAX_QUESTION_CHARS:
        st.warning(f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
        return
    if st.session_state.get("count", 0) >= SESSION_LIMIT:
        st.warning(f"Session limit reached ({SESSION_LIMIT} questions). Reload the page to start a new session.")
        return
    try:
        with st.spinner("Retrieving, grading, answering…"):
            trace = agent.run(question)
    except QuotaExhausted:
        st.info(QUOTA_MSG)
        return
    except RateLimited:
        st.info(BUSY_MSG)
        return
    except Exception:  # noqa: BLE001 - never show a stack trace to visitors
        log.exception("agent run failed")
        st.error("Something went wrong. Please try again.")
        return
    st.session_state["count"] = st.session_state.get("count", 0) + 1
    st.markdown(render_answer(trace))
    with st.expander("Agent trace", expanded=False):
        st.markdown(render_trace(trace))


st.title("🔄 Polygraph — self-reflective agentic RAG with evals")
st.markdown(INTRO)

tab_ask, tab_eval, tab_pdf = st.tabs(["Ask", "Eval Results", "Upload your PDF (optional)"])

with tab_ask:
    picked = st.pills("Examples (the last two are not in the knowledge base)", EXAMPLES, key="example")
    with st.form("ask_form"):
        question = st.text_area("Question", value=picked or "", max_chars=MAX_QUESTION_CHARS, height=90,
                                placeholder="e.g. How do I limit how much memory a container can use?")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted:
        ask(kb_agent(), question)

with tab_eval:
    st.markdown(eval_results_md())

with tab_pdf:
    st.caption("Index a PDF for this session only (kept in memory, never stored).")
    upload = st.file_uploader("PDF", type=["pdf"])
    if upload is not None and st.session_state.get("pdf_name") != upload.name:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(upload.getbuffer())
            tmp.flush()
            try:
                chunks = load_pdf(tmp.name, get_settings())[:2000]
            except Exception:  # noqa: BLE001
                chunks = []
        if chunks:
            s = get_settings().with_overrides(max_retries=2)
            st.session_state["pdf_agent"] = Agent(Retriever.from_chunks(chunks, s), s, LLM(s))
            st.session_state["pdf_name"] = upload.name
            st.success(f"Indexed {len(chunks)} chunks from {upload.name}.")
        else:
            st.error("Could not extract text from this PDF.")
    if st.session_state.get("pdf_agent") is not None:
        with st.form("pdf_form"):
            pq = st.text_input("Question about your PDF", max_chars=MAX_QUESTION_CHARS)
            if st.form_submit_button("Ask"):
                ask(st.session_state["pdf_agent"], pq)
