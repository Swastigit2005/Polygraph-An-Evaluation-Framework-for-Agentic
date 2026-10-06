"""Gradio entry point (Hugging Face Spaces runs this file)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr  # noqa: E402

from rag_agent.config import get_settings  # noqa: E402
from rag_agent.graph import answer  # noqa: E402
from rag_agent.ingest import load_pdf  # noqa: E402
from rag_agent.llm import QuotaExhausted, RateLimited  # noqa: E402
from rag_agent.retrieval import Retriever  # noqa: E402

MAX_QUESTION_CHARS = 500
BUSY_MSG = "⏳ The demo is busy (shared free-tier quota). Please try again in a minute."

settings = get_settings()
kb_retriever = Retriever.from_index(settings)


def _run(question: str, retriever: Retriever | None) -> tuple[str, str]:
    question = (question or "").strip()
    if not question:
        return "*Ask a question to get started.*", ""
    if len(question) > MAX_QUESTION_CHARS:
        return f"⚠️ Please keep questions under {MAX_QUESTION_CHARS} characters.", ""
    if retriever is None:
        return "⚠️ Upload and process a PDF first.", ""
    try:
        state = answer(question, retriever, settings)
    except (RateLimited, QuotaExhausted):
        return BUSY_MSG, ""
    sources = "\n".join(f"- `{h.chunk.id}` — {h.chunk.header()}" for h in state.get("hits", []))
    log = f"Final query: {state.get('query')}\nRewrites: {state.get('rewrites', 0)}\n"
    if state.get("verdict") is not None:
        log += f"Last grader verdict: {state['verdict'].model_dump()}\n"
    return f"{state.get('answer', '')}\n\n**Sources**\n{sources}", log


def ask_kb(question: str) -> tuple[str, str]:
    """Answer a question against the prebuilt Docker docs index."""
    return _run(question, kb_retriever)


def process_pdf(file: str | None) -> tuple[Retriever | None, str]:
    """Index an uploaded PDF in memory for this session."""
    if not file:
        return None, "No file selected."
    chunks = load_pdf(file, settings)
    if not chunks:
        return None, "❌ No extractable text found in this PDF."
    return Retriever.from_chunks(chunks, settings), f"✅ Indexed {len(chunks)} chunks."


with gr.Blocks(title="Agentic RAG with Evals") as demo:
    gr.Markdown("# Self-Reflective Agentic RAG — Docker docs")
    with gr.Tab("Ask"):
        q = gr.Textbox(label="Question", max_length=MAX_QUESTION_CHARS, lines=2)
        btn = gr.Button("Ask", variant="primary")
        out = gr.Markdown()
        with gr.Accordion("Agent trace", open=False):
            trace = gr.Textbox(lines=8, interactive=False, show_label=False)
        btn.click(ask_kb, q, [out, trace])
        q.submit(ask_kb, q, [out, trace])
    with gr.Tab("Upload your PDF (optional)"):
        pdf_state = gr.State(None)
        f = gr.File(label="PDF", file_types=[".pdf"])
        status = gr.Markdown()
        gr.Button("Process").click(process_pdf, f, [pdf_state, status])
        pq = gr.Textbox(label="Question about your PDF", max_length=MAX_QUESTION_CHARS)
        pout = gr.Markdown()
        ptrace = gr.Textbox(lines=6, interactive=False, label="Agent trace")
        gr.Button("Ask").click(_run, [pq, pdf_state], [pout, ptrace])

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
