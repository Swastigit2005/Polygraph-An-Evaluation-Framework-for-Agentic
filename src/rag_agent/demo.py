"""Glue for the web demo: example questions and the JSON the frontend renders.

The serverless function (api/ask.py) and scripts/export_demo_data.py both build responses with
`trace_payload`, so a recorded example and a live answer have exactly the same shape.
"""

from __future__ import annotations

import re
from typing import Any

from rag_agent.graph import Trace
from rag_agent.retrieval import Retriever

MAX_QUESTION_CHARS = 300
BUSY_MSG = "The demo is busy right now (it runs on a free-tier LLM quota). Please try again in a minute."
QUOTA_MSG = "Today's free demo quota is used up. Please come back tomorrow, or try one of the examples."

# kind: what a visitor should watch for. "trick" questions are deliberately outside the docs.
EXAMPLES = [
    {"kind": "normal", "question": "How do I limit how much memory a container can use?"},
    {"kind": "vague", "question": "container can't reach the internet"},
    {"kind": "trick", "question": "How do I upgrade my Docker Hub subscription to a Team plan?"},
    {"kind": "normal", "question": "How does Compose watch sync file changes into running containers?"},
    {"kind": "trick", "question": "How do I configure an NGINX ingress controller in Kubernetes?"},
]

_CITE = re.compile(r"\[([^\[\]\s]+#\d{3})\]")


def _excerpt(text: str, n: int = 220) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"[#*`>|_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def trace_payload(trace: Trace, retriever: Retriever) -> dict[str, Any]:
    """The trace as the frontend needs it: steps, answer with numbered citations, and sources."""
    sources = []
    answer = trace.answer
    for n, c in enumerate(trace.citations, 1):
        chunk = retriever.by_id.get(c["id"])
        sources.append({"n": n, "id": c["id"], "title": c["title"], "url": c["url"],
                        "excerpt": _excerpt(chunk.text) if chunk else ""})
        answer = answer.replace(f"[{c['id']}]", f"[{n}]")
    answer = _CITE.sub("", answer)  # citations of chunks that were not in the context

    seen = {i for st in trace.steps if st.node == "retrieve" for i in st.data["retrieved_ids"]}
    titles = {i: retriever.by_id[i].header() for i in seen if i in retriever.by_id}
    steps = []
    for st in trace.steps:
        data = {k: v for k, v in st.data.items() if k != "stages"}  # per-stage rankings are eval-only
        steps.append({"node": st.node, "ms": round(st.latency_s * 1000), "data": data,
                      "tokens": sum(u.total_tokens for u in st.usage)})
    return {
        "question": trace.question,
        "answer": answer,
        "refused": trace.refused,
        "sources": sources,
        "steps": steps,
        "titles": titles,
        "stats": {"seconds": round(trace.latency_s, 2), "tokens": trace.total_tokens,
                  "rewrites": trace.rewrites, "llm_calls": len(trace.usages)},
        "models": {"generator": trace.config.get("gen_model"), "grader": trace.config.get("fast_model")},
    }
