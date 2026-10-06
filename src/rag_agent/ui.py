"""Shared presentation helpers for the demo apps (Gradio and Streamlit)."""

from __future__ import annotations

from pathlib import Path

from rag_agent.graph import Trace

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "eval" / "results"
MAX_QUESTION_CHARS = 500
BUSY_MSG = "⏳ **Demo busy** — this demo shares a free-tier Groq quota. Please try again in a minute."
QUOTA_MSG = "⏳ **Daily demo quota used up** (free tier). Please come back tomorrow, or run it locally (see README)."
EXAMPLES = [
    "How do I limit how much memory a container can use?",
    "I get 'Cannot connect to the Docker daemon' — how do I check whether the daemon is running?",
    "my builds are slow every time, how to make it faster",
    "How does Compose watch sync file changes into running containers?",
    "How do I configure an NGINX ingress controller in Kubernetes?",  # not in the knowledge base
    "How do I upgrade my Docker Hub subscription to a Team plan?",  # not in the knowledge base
]
INTRO = (
    "Ask about **Docker** (Engine, Compose, Build, Desktop). The agent retrieves (BM25 + dense, RRF, "
    "cross-encoder rerank), **grades** the context, **rewrites** the query if it fails, and **refuses** instead "
    "of guessing. Open *Agent trace* to see every decision; *Eval Results* shows how those decisions were measured."
)


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
