"""Gradio entry point (Hugging Face Spaces runs this file)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr  # noqa: E402

from rag_agent.config import get_settings  # noqa: E402
from rag_agent.graph import Agent, Trace  # noqa: E402
from rag_agent.ingest import load_pdf  # noqa: E402
from rag_agent.llm import QuotaExhausted, RateLimited  # noqa: E402
from rag_agent.retrieval import Retriever  # noqa: E402

MAX_QUESTION_CHARS = 500
BUSY_MSG = "⏳ The demo is busy (shared free-tier quota). Please try again in a minute."

settings = get_settings()
kb_agent = Agent(Retriever.from_index(settings), settings)


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
    lines = [f"**{len(trace.steps)} steps · {trace.rewrites} rewrite(s) · {trace.latency_s:.1f}s · "
             f"{trace.total_tokens} tokens**", ""]
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
            lines.append(f"{head} — cited: " + (", ".join(f"`{x}`" for x in d["cited_ids"]) or "none")
                         + (" (generator declined)" if d["refused"] else ""))
        elif st.node == "refuse":
            lines.append(f"{head} — {d['reason']}")
        else:
            lines.append(head)
    return "\n".join(lines)


def _run(question: str, agent: Agent | None) -> tuple[str, str]:
    question = (question or "").strip()
    if not question:
        return "*Ask a question to get started.*", ""
    if len(question) > MAX_QUESTION_CHARS:
        return f"⚠️ Please keep questions under {MAX_QUESTION_CHARS} characters.", ""
    if agent is None:
        return "⚠️ Upload and process a PDF first.", ""
    try:
        trace = agent.run(question)
    except (RateLimited, QuotaExhausted):
        return BUSY_MSG, ""
    return render_answer(trace), render_trace(trace)


def ask_kb(question: str) -> tuple[str, str]:
    """Answer a question against the prebuilt Docker docs index."""
    return _run(question, kb_agent)


def process_pdf(file: str | None) -> tuple[Agent | None, str]:
    """Index an uploaded PDF in memory for this session."""
    if not file:
        return None, "No file selected."
    chunks = load_pdf(file, settings)
    if not chunks:
        return None, "❌ No extractable text found in this PDF."
    return Agent(Retriever.from_chunks(chunks, settings), settings), f"✅ Indexed {len(chunks)} chunks."


with gr.Blocks(title="Agentic RAG with Evals") as demo:
    gr.Markdown("# Self-Reflective Agentic RAG — Docker docs")
    with gr.Tab("Ask"):
        q = gr.Textbox(label="Question", max_length=MAX_QUESTION_CHARS, lines=2)
        btn = gr.Button("Ask", variant="primary")
        out = gr.Markdown()
        with gr.Accordion("Agent trace", open=False):
            trace_md = gr.Markdown()
        btn.click(ask_kb, q, [out, trace_md])
        q.submit(ask_kb, q, [out, trace_md])
    with gr.Tab("Upload your PDF (optional)"):
        pdf_agent = gr.State(None)
        f = gr.File(label="PDF", file_types=[".pdf"])
        status = gr.Markdown()
        gr.Button("Process").click(process_pdf, f, [pdf_agent, status])
        pq = gr.Textbox(label="Question about your PDF", max_length=MAX_QUESTION_CHARS)
        pout = gr.Markdown()
        with gr.Accordion("Agent trace", open=False):
            ptrace = gr.Markdown()
        gr.Button("Ask").click(_run, [pq, pdf_agent], [pout, ptrace])

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
