"""Streamlit entry point for the live demo (Streamlit Community Cloud runs this file).

Configuration: non-secret settings come from config/demo.env (committed); secrets such as
GROQ_API_KEY come from Streamlit's secrets store (or a local .env). Both are copied into
environment variables before the app's config is loaded, so the rest of the code is unchanged.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

st.set_page_config(page_title="Polygraph · Agentic RAG with evals", page_icon="🔄", layout="wide",
                   initial_sidebar_state="collapsed")

try:  # Streamlit Cloud secrets -> env vars (no-op locally if there is no secrets file)
    for key, value in st.secrets.items():
        if isinstance(value, str | int | float | bool):
            os.environ.setdefault(key, str(value))
except Exception:  # noqa: BLE001 - missing or malformed secrets: handled below with a clear message
    pass
load_dotenv(ROOT / "config" / "demo.env", override=False)  # non-secret defaults

from rag_agent.config import get_settings  # noqa: E402
from rag_agent.graph import Agent, Trace  # noqa: E402
from rag_agent.ingest import load_pdf  # noqa: E402
from rag_agent.llm import LLM, QuotaExhausted, RateLimited  # noqa: E402
from rag_agent.retrieval import Retriever  # noqa: E402
from rag_agent.ui import (  # noqa: E402
    BUSY_MSG,
    CSS,
    EXAMPLES,
    MAX_QUESTION_CHARS,
    QUOTA_MSG,
    latest_results,
    number_citations,
    trace_timeline_html,
)

log = logging.getLogger("streamlit_app")
SESSION_LIMIT = int(os.getenv("SESSION_QUESTION_LIMIT", "15"))
REPO_URL = "https://github.com/Swastigit2005/Polygraph-An-Evaluation-Framework-for-Agentic"

st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading the knowledge base and local models (first visit only)…")
def kb_agent() -> Agent:
    """One shared agent over the prebuilt Docker-docs index (fail-fast retries for the UI)."""
    s = get_settings().with_overrides(max_retries=2)
    retriever = Retriever.from_index(s)
    retriever.retrieve("warm-up query", s)  # load embedder + reranker now, not on the first question
    return Agent(retriever, s, LLM(s))


@st.cache_data(ttl=600, show_spinner=False)
def results() -> dict[str, dict]:
    """Latest complete eval result per config (cached for 10 minutes)."""
    return latest_results()


def pct(v: float | None) -> str:
    """Format a rate as a percentage, or TBD."""
    return "TBD" if v is None else f"{100 * v:.1f}%"


# --------------------------------------------------------------------------- hero
full = results().get("full")
if full:
    m = full["metrics"]
    k = full["config"]["top_k"]
    stats = [(pct(m["refusal"]["correct_refusal_rate"]), "correct refusals (unanswerable)"),
             (pct(m["refusal"]["false_refusal_rate"]), "false refusals (answerable)"),
             (pct(m["retrieval"][f"recall@{k}"]), f"retrieval recall@{k}"),
             (pct((m["grader"]["set_level"] or {}).get("precision")), "grader precision")]
    note = f"Measured on {full['n_evaluated']} golden questions (AI-reviewed set) · see Eval Results"
else:
    stats, note = [("TBD", "evaluation pending")], "No completed evaluation run yet"
stat_html = "".join(f'<div class="pg-stat"><b>{v}</b><span>{label}</span></div>' for v, label in stats)
st.markdown(f"""
<div class="pg-hero">
  <div class="pg-eyebrow">Agent evaluation demo · Docker documentation</div>
  <h1>Polygraph</h1>
  <p>A RAG agent that <b>grades</b> what it retrieved, <b>rewrites</b> the query when the context is wrong,
  and <b>refuses</b> instead of guessing — with every decision traced and measured.</p>
  <div class="pg-stats">{stat_html}</div>
  <div class="pg-note">{note}</div>
</div>
""", unsafe_allow_html=True)

configured = bool(os.getenv("GROQ_API_KEY"))
if not configured:
    st.error("**Setup needed:** this deployment has no `GROQ_API_KEY`. In Streamlit Cloud open **Manage app → "
             "Settings → Secrets** and add one line:  `GROQ_API_KEY = \"gsk_...\"`  (TOML format, with quotes), "
             "then save. The app restarts automatically.")


# --------------------------------------------------------------------------- ask
def run_question(agent: Agent, question: str) -> Trace | None:
    """Validate and run one question; returns the trace or None (after showing a message)."""
    question = (question or "").strip()
    if not question:
        st.info("Type a question or pick an example.")
        return None
    if len(question) > MAX_QUESTION_CHARS:
        st.warning(f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
        return None
    if st.session_state.get("count", 0) >= SESSION_LIMIT:
        st.warning(f"Session limit reached ({SESSION_LIMIT} questions). Reload the page to start a new session.")
        return None
    try:
        with st.spinner("Retrieving → grading → answering…"):
            trace = agent.run(question)
    except QuotaExhausted:
        st.info(QUOTA_MSG)
        return None
    except RateLimited:
        st.info(BUSY_MSG)
        return None
    except Exception:  # noqa: BLE001 - never show a stack trace to visitors
        log.exception("agent run failed")
        st.error("Something went wrong. Please try again.")
        return None
    st.session_state["count"] = st.session_state.get("count", 0) + 1
    return trace


def show_answer(trace: Trace) -> None:
    """Answer card with status badge and sources."""
    with st.container(border=True):
        if trace.refused:
            st.markdown('<span class="pg-badge pg-no">Refused · not in the knowledge base</span>',
                        unsafe_allow_html=True)
            st.markdown(f"**{trace.answer}**")
            st.caption("The grader rejected the retrieved context after every rewrite, so the agent declined "
                       "rather than answer from the model's own memory.")
            return
        st.markdown('<span class="pg-badge pg-ok">Answered from the knowledge base</span>', unsafe_allow_html=True)
        inline = bool(trace.citations) and trace.citations[0]["source"] == "inline"
        st.markdown(number_citations(trace.answer, trace.citations) if inline else trace.answer)
        if trace.citations:
            st.markdown("**Sources**")
            for n, c in enumerate(trace.citations, 1):
                ref = f"**[{n}]** " if inline else ""
                st.markdown(f"{ref}[{c['title']}]({c['url']}) · `{c['id']}`" if c["url"] else f"{ref}{c['title']}")
            if trace.citations[0]["source"] != "inline":
                st.caption("No inline citations; showing the chunks the answer was generated from.")


HOW_IT_WORKS = """
<div class="pg-howto"><ol>
<li><b>Retrieve</b> — BM25 + dense vectors fused with Reciprocal Rank Fusion, reranked by a cross-encoder.</li>
<li><b>Grade</b> — a small LLM returns a structured verdict: is the answer in this context, and which chunks?</li>
<li><b>Rewrite</b> — if not, the query is rewritten and retrieval runs again (up to 2 times).</li>
<li><b>Answer or refuse</b> — answer only from graded context with citations; otherwise refuse.</li>
</ol></div>
"""

tab_ask, tab_eval, tab_pdf = st.tabs(["Ask the agent", "Eval results", "Your PDF (optional)"])

with tab_ask:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Try an example** <span style='color:#64748b;font-size:.85rem'>— the last two are "
                    "deliberately outside the knowledge base</span>", unsafe_allow_html=True)
        ex_cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            ex_cols[i % 2].button(("🚫 " if i >= len(EXAMPLES) - 2 else "") + ex, key=f"ex{i}",
                                  use_container_width=True, on_click=st.session_state.__setitem__, args=("q", ex))
        with st.form("ask_form", border=False):
            question = st.text_area("Your question", key="q", max_chars=MAX_QUESTION_CHARS, height=96,
                                    placeholder="e.g. How do I limit how much memory a container can use?")
            submitted = st.form_submit_button("Ask", type="primary", disabled=not configured)
        if submitted:
            trace = run_question(kb_agent(), question)
            if trace is not None:
                st.session_state["last_trace"] = trace
        if st.session_state.get("last_trace") is not None:
            show_answer(st.session_state["last_trace"])
    with right:
        st.markdown("#### Agent trace")
        if st.session_state.get("last_trace") is not None:
            st.markdown(trace_timeline_html(st.session_state["last_trace"]), unsafe_allow_html=True)
        else:
            st.caption("Ask a question to see each decision the agent makes.")
            st.markdown(HOW_IT_WORKS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- eval results
with tab_eval:
    res = results()
    if not res:
        st.info("No completed evaluation run yet — all metrics are TBD.")
    else:
        from make_report import compact_trace, failure_examples
        from report import metrics_table

        n = next(iter(res.values()))["n_evaluated"]
        st.markdown(f"Results from `eval/results/` · **{n} golden questions** per configuration · golden set "
                    "reviewed by an AI (Claude), not a human · judge calibration not yet performed.")
        base, fullr = res.get("baseline"), res.get("full")
        if base and fullr:
            st.markdown("##### Full agent vs. baseline (dense retrieval, no grader)")
            bm, fm = base["metrics"], fullr["metrics"]
            k = fullr["config"]["top_k"]
            cards = [
                ("Correct refusals", fm["refusal"]["correct_refusal_rate"], bm["refusal"]["correct_refusal_rate"],
                 "normal"),
                ("False refusals", fm["refusal"]["false_refusal_rate"], bm["refusal"]["false_refusal_rate"],
                 "inverse"),
                (f"Recall@{k}", fm["retrieval"][f"recall@{k}"], base["metrics"]["retrieval"][f"recall@{k}"],
                 "normal"),
                (f"MRR@{k}", fm["retrieval"][f"mrr@{k}"], bm["retrieval"][f"mrr@{k}"], "normal"),
            ]
            cols = st.columns(len(cards))
            for col, (label, fv, bv, mode) in zip(cols, cards, strict=True):
                if fv is None or bv is None:
                    delta = None
                elif "MRR" in label:
                    delta = f"{fv - bv:+.3f} vs baseline"
                else:
                    delta = f"{100 * (fv - bv):+.1f} pts vs baseline"
                col.metric(label, pct(fv) if "MRR" not in label else f"{fv:.3f}", delta, delta_color=mode,
                           border=True)
        st.markdown("##### All metrics")
        st.markdown(metrics_table(res))
        st.caption("— = not computed for that configuration (grader/rewrite metrics need reflection; Ragas runs on "
                   "`baseline` and `full` only to fit the free-tier judge budget).")
        if fullr:
            st.markdown("##### Failure analysis (full agent)")
            for i, (cat, dx, rec) in enumerate(failure_examples(fullr), 1):
                with st.expander(f"{i}. {cat} — {rec['item']['question']}"):
                    st.markdown(f"**Diagnosis:** {dx}")
                    st.code(compact_trace(rec["trace"]), language="text")


# --------------------------------------------------------------------------- PDF
with tab_pdf:
    st.caption("Index a PDF for this session only (kept in memory, never stored).")
    upload = st.file_uploader("PDF", type=["pdf"], disabled=not configured)
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
        with st.form("pdf_form", border=False):
            pq = st.text_input("Question about your PDF", max_chars=MAX_QUESTION_CHARS)
            if st.form_submit_button("Ask", type="primary"):
                ptrace = run_question(st.session_state["pdf_agent"], pq)
                if ptrace is not None:
                    show_answer(ptrace)
                    with st.expander("Agent trace"):
                        st.markdown(trace_timeline_html(ptrace), unsafe_allow_html=True)

st.markdown(f"<div style='text-align:center;color:#94a3b8;font-size:.8rem;margin-top:28px'>"
            f"<a href='{REPO_URL}' style='color:#6366f1'>Source code & evaluation methodology</a> · "
            "LangGraph · Groq (gpt-oss) · ChromaDB · BM25 · sentence-transformers · Ragas · DeepEval</div>",
            unsafe_allow_html=True)
