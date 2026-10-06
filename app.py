"""Gradio entry point (Hugging Face Spaces runs this file)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

from rag_agent.config import get_settings  # noqa: E402
from rag_agent.graph import Agent, Trace  # noqa: E402
from rag_agent.ingest import load_pdf  # noqa: E402
from rag_agent.llm import LLM, QuotaExhausted, RateLimited  # noqa: E402
from rag_agent.retrieval import Retriever  # noqa: E402

log = logging.getLogger("app")

MAX_QUESTION_CHARS = 500
SESSION_LIMIT = int(os.getenv("SESSION_QUESTION_LIMIT", "15"))
BUSY_MSG = "⏳ **Demo busy** — this Space shares a free-tier Groq quota. Please try again in a minute."
QUOTA_MSG = "⏳ **Daily demo quota used up** (free tier). Please come back tomorrow, or run it locally (see README)."
RESULTS_DIR = ROOT / "eval" / "results"

EXAMPLES = [
    "How do I limit how much memory a container can use?",
    "I get 'Cannot connect to the Docker daemon' — how do I check whether the daemon is running?",
    "my builds are slow every time, how to make it faster",
    "How does Compose watch sync file changes into running containers?",
    "How do I configure an NGINX ingress controller in Kubernetes?",  # not in the knowledge base
    "How do I upgrade my Docker Hub subscription to a Team plan?",  # not in the knowledge base
]

# Fail fast in the UI: a couple of retries, then show the busy message instead of hanging.
settings = get_settings().with_overrides(max_retries=2)
llm = LLM(settings)
kb_agent = Agent(Retriever.from_index(settings), settings, llm)


# --------------------------------------------------------------------------- rendering
def render_answer(trace: Trace) -> str:
    """Answer plus a numbered source list."""
    if trace.refused:
        return f"🚫 **{trace.answer}**\n\n_The agent refused rather than answer from outside the knowledge base._"
    sources = "\n".join(
        f"{i}. [`{c['id']}`]({c['url']}) — {c['title']}" if c["url"] else f"{i}. `{c['id']}` — {c['title']}"
        for i, c in enumerate(trace.citations, 1))
    note = "" if not trace.citations or trace.citations[0]["source"] == "inline" else \
        "\n_(no inline citations; showing the chunks the answer was generated from)_"
    return f"{trace.answer}\n\n**Sources**\n{sources or '_none_'}{note}"


def render_trace(trace: Trace) -> str:
    """Step-by-step agent trace as Markdown."""
    cost = f" · ${trace.cost_usd:.5f} list price" if trace.cost_usd is not None else ""
    lines = [f"**{len(trace.steps)} steps · {trace.rewrites} rewrite(s) · {trace.latency_s:.1f}s · "
             f"{trace.total_tokens:,} tokens{cost}**", ""]
    for i, st in enumerate(trace.steps, 1):
        d = st.data
        head = f"{i}. **{st.node}** ({st.latency_s * 1000:.0f} ms)"
        if st.node == "retrieve":
            ids = ", ".join(f"`{x}`" for x in d["retrieved_ids"])
            lines.append(f"{head} — query: _{d['query']}_  \n   ids: {ids}")
        elif st.node == "grade_context":
            mark = "✅ relevant" if d["relevant"] else "❌ not relevant"
            lines.append(f"{head} — {mark}: {d['reason']}")
        elif st.node == "rewrite_query":
            lines.append(f"{head} — _{d['from']}_ → _{d['to']}_")
        elif st.node == "generate":
            cited = ", ".join(f"`{x}`" for x in d["cited_ids"]) or "none"
            lines.append(f"{head} — cited: {cited}" + (" (generator declined)" if d["refused"] else ""))
        elif st.node == "refuse":
            lines.append(f"{head} — {d['reason']}")
        else:
            lines.append(head)
    return "\n".join(lines)


def eval_results_md() -> str:
    """Latest ablation table + judge calibration, read from eval/results (never hard-coded)."""
    path = RESULTS_DIR / "ABLATIONS.md"
    if not path.exists():
        return "No evaluation results yet — all metrics **TBD**."
    return path.read_text().replace("# Ablation results", "## Ablation results", 1)


# --------------------------------------------------------------------------- handlers
def _run(question: str, agent: Agent | None, count: int) -> tuple[str, str, int]:
    question = (question or "").strip()
    if not question:
        return "*Ask a question to get started.*", "", count
    if len(question) > MAX_QUESTION_CHARS:
        return f"⚠️ Please keep questions under {MAX_QUESTION_CHARS} characters.", "", count
    if count >= SESSION_LIMIT:
        return f"⚠️ Session limit reached ({SESSION_LIMIT} questions). Refresh the page to start a new session.", \
            "", count
    if agent is None:
        return "⚠️ Upload and process a PDF first.", "", count
    try:
        trace = agent.run(question)
    except QuotaExhausted:
        return QUOTA_MSG, "", count
    except RateLimited:
        return BUSY_MSG, "", count
    except Exception:  # noqa: BLE001 - never show a stack trace to visitors
        log.exception("agent run failed")
        return "❌ Something went wrong. Please try again.", "", count
    return render_answer(trace), render_trace(trace), count + 1


def ask_kb(question: str, count: int) -> tuple[str, str, int]:
    """Answer a question against the prebuilt Docker docs index."""
    return _run(question, kb_agent, count)


def ask_pdf(question: str, agent: Agent | None, count: int) -> tuple[str, str, int]:
    """Answer a question against this session's uploaded PDF."""
    return _run(question, agent, count)


def process_pdf(file: str | None) -> tuple[Agent | None, str]:
    """Index an uploaded PDF in memory for this session."""
    if not file:
        return None, "No file selected."
    try:
        chunks = load_pdf(file, settings)[:2000]
    except Exception:  # noqa: BLE001
        log.exception("pdf processing failed")
        return None, "❌ Could not read this PDF."
    if not chunks:
        return None, "❌ No extractable text found in this PDF."
    return Agent(Retriever.from_chunks(chunks, settings), settings, llm), f"✅ Indexed {len(chunks)} chunks."


# --------------------------------------------------------------------------- UI
INTRO = """
# 🔄 Self-Reflective Agentic RAG — with an evaluation harness
Ask about **Docker** (Engine, Compose, Build, Desktop). The agent retrieves (BM25 + dense, RRF, cross-encoder
rerank), **grades** the context, **rewrites** the query if it fails, and **refuses** instead of guessing.
Open *Agent trace* to see every decision. The *Eval Results* tab shows how those decisions were measured.
"""

with gr.Blocks(title="Agentic RAG with Evals") as demo:
    gr.Markdown(INTRO)
    count = gr.State(0)
    with gr.Tab("Ask"):
        q = gr.Textbox(label="Question", max_length=MAX_QUESTION_CHARS, lines=2,
                       placeholder="e.g. How do I limit how much memory a container can use?")
        btn = gr.Button("Ask", variant="primary")
        gr.Examples(EXAMPLES, inputs=q, label="Examples (the last two are not in the knowledge base)")
        out = gr.Markdown()
        with gr.Accordion("Agent trace", open=False):
            trace_md = gr.Markdown()
        btn.click(ask_kb, [q, count], [out, trace_md, count])
        q.submit(ask_kb, [q, count], [out, trace_md, count])
    with gr.Tab("Eval Results"):
        results_md = gr.Markdown(eval_results_md())
        gr.Button("Reload", size="sm").click(eval_results_md, None, results_md)
    with gr.Tab("Upload your PDF (optional)"):
        gr.Markdown("Index a PDF for this session only (kept in memory, never stored).")
        pdf_agent = gr.State(None)
        f = gr.File(label="PDF", file_types=[".pdf"])
        status = gr.Markdown()
        gr.Button("Process").click(process_pdf, f, [pdf_agent, status])
        pq = gr.Textbox(label="Question about your PDF", max_length=MAX_QUESTION_CHARS)
        pout = gr.Markdown()
        with gr.Accordion("Agent trace", open=False):
            ptrace = gr.Markdown()
        gr.Button("Ask").click(ask_pdf, [pq, pdf_agent, count], [pout, ptrace, count])
    demo.load(eval_results_md, None, results_md)

demo.queue(default_concurrency_limit=2, max_size=20)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
