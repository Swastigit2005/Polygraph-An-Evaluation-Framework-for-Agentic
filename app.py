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

log = logging.getLogger("app")

SESSION_LIMIT = int(os.getenv("SESSION_QUESTION_LIMIT", "15"))

# Fail fast in the UI: a couple of retries, then show the busy message instead of hanging.
settings = get_settings().with_overrides(max_retries=2)
llm = LLM(settings)
kb_agent = Agent(Retriever.from_index(settings), settings, llm)


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
with gr.Blocks(title="Agentic RAG with Evals") as demo:
    gr.Markdown("# 🔄 Polygraph — self-reflective agentic RAG with evals\n" + INTRO)
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
