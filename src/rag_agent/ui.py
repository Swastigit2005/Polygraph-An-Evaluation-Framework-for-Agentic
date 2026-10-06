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


# --------------------------------------------------------------------------- results + rich (HTML) rendering
def latest_results() -> dict[str, dict]:
    """Newest complete, non-dry-run result per config, read from eval/results (never hard-coded)."""
    import json

    out: dict[str, dict] = {}
    for path in sorted(RESULTS_DIR.glob("*_*.json"), reverse=True):
        if "__dryrun" in path.name:
            continue
        try:
            res = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        name = res.get("config_name")
        if name and name not in out and res.get("complete") and not res.get("dry_run"):
            res["_file"] = path.name
            out[name] = res
    order = ["baseline", "hybrid", "hybrid_rerank", "full"]  # ablation order: add one component at a time
    return dict(sorted(out.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)))


def _esc(text: object) -> str:
    import html

    return html.escape(str(text), quote=True)


_STEP_STYLE = {
    "retrieve": ("Retrieve", "#2563eb"),
    "grade_pass": ("Grade · pass", "#059669"),
    "grade_fail": ("Grade · fail", "#d97706"),
    "rewrite_query": ("Rewrite query", "#7c3aed"),
    "generate": ("Generate", "#4f46e5"),
    "refuse": ("Refuse", "#dc2626"),
}


def trace_timeline_html(trace: Trace) -> str:
    """Agent trace as an HTML timeline. All model/user text is HTML-escaped."""
    cost = f" · ${trace.cost_usd:.5f}" if trace.cost_usd is not None else ""
    chips = (f'<div class="pg-chips"><span>{len(trace.steps)} steps</span><span>{trace.rewrites} rewrite(s)</span>'
             f'<span>{trace.latency_s:.1f}s</span><span>{trace.total_tokens:,} tokens{cost}</span></div>')
    rows = []
    for st in trace.steps:
        d = st.data
        key = st.node
        if key == "grade_context":
            key = "grade_pass" if d["relevant"] else "grade_fail"
        label, color = _STEP_STYLE.get(key, (st.node, "#64748b"))
        if st.node == "retrieve":
            ids = "".join(f"<code>{_esc(i)}</code>" for i in d["retrieved_ids"])
            body = f'<div class="pg-q">“{_esc(d["query"])}”</div><div class="pg-ids">{ids}</div>'
        elif st.node == "grade_context":
            body = f"<div>{_esc(d['reason'])}</div>"
        elif st.node == "rewrite_query":
            body = f'<div class="pg-q">“{_esc(d["from"])}”</div><div>→ <b>“{_esc(d["to"])}”</b></div>'
        elif st.node == "generate":
            cited = ", ".join(_esc(c) for c in d["cited_ids"]) or "no inline citations"
            body = f"<div>{'Generator declined: ' if d['refused'] else 'Cited: '}{cited}</div>"
        elif st.node == "refuse":
            body = f"<div>{_esc(d['reason'])}</div>"
        else:
            body = ""
        rows.append(f'<div class="pg-step"><div class="pg-dot" style="background:{color}"></div>'
                    f'<div class="pg-body"><div class="pg-head"><span style="color:{color}">{label}</span>'
                    f'<span class="pg-ms">{st.latency_s * 1000:.0f} ms</span></div>{body}</div></div>')
    return chips + '<div class="pg-timeline">' + "".join(rows) + "</div>"


CSS = """
<style>
#MainMenu, footer, [data-testid="stDecoration"] {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1200px;}
.pg-hero {background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 55%, #6366f1 100%); color: #fff;
  border-radius: 18px; padding: 28px 32px; margin-bottom: 18px;}
.pg-eyebrow {letter-spacing: .14em; font-size: .72rem; opacity: .8; text-transform: uppercase; font-weight: 600;}
.pg-hero h1 {color: #fff; font-size: 2.3rem; margin: 6px 0 4px; padding: 0; font-weight: 750;}
.pg-hero p {color: #e0e7ff; margin: 0; font-size: 1.02rem; max-width: 760px;}
.pg-stats {display: flex; gap: 12px; margin-top: 18px; flex-wrap: wrap;}
.pg-stat {background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18); border-radius: 12px;
  padding: 10px 14px; min-width: 150px;}
.pg-stat b {display: block; font-size: 1.35rem; color: #fff;}
.pg-stat span {font-size: .78rem; color: #c7d2fe;}
.pg-note {font-size: .74rem; color: #c7d2fe; margin-top: 10px;}
.pg-badge {display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
  margin-bottom: 8px;}
.pg-ok {background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0;}
.pg-no {background: #fff7ed; color: #b45309; border: 1px solid #fed7aa;}
.pg-chips {display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px;}
.pg-chips span {background: #eef2ff; color: #3730a3; border-radius: 999px; padding: 2px 10px; font-size: .76rem;}
.pg-timeline {border-left: 2px solid #e2e8f0; margin-left: 6px; padding-left: 0;}
.pg-step {position: relative; padding: 0 0 14px 20px;}
.pg-dot {position: absolute; left: -7px; top: 4px; width: 12px; height: 12px; border-radius: 50%;
  box-shadow: 0 0 0 3px #fff;}
.pg-head {display: flex; justify-content: space-between; font-weight: 650; font-size: .86rem;}
.pg-ms {color: #94a3b8; font-weight: 500; font-size: .76rem;}
.pg-body {font-size: .84rem; color: #334155; line-height: 1.45;}
.pg-q {color: #475569; font-style: italic;}
.pg-ids code {font-size: .7rem; background: #f1f5f9; color: #334155; border-radius: 6px; padding: 1px 6px;
  margin: 3px 4px 0 0; display: inline-block;}
.pg-howto {font-size: .88rem; color: #334155;}
div[data-testid="stButton"] button[kind="secondary"] {justify-content: flex-start; text-align: left;
  padding: 6px 12px; min-height: 0; border-radius: 10px; height: auto;}
div[data-testid="stButton"] button[kind="secondary"] p {white-space: normal; overflow: visible;
  text-overflow: clip; font-size: .82rem; text-align: left; line-height: 1.35;}
.pg-howto li {margin-bottom: 6px;}
</style>
"""


def number_citations(answer: str, citations: list[dict[str, str]]) -> str:
    """Replace inline [chunk#id] citations with [1], [2], ... matching the Sources list order."""
    for n, c in enumerate(citations, 1):
        answer = answer.replace(f"[{c['id']}]", f"**[{n}]**")
    return answer
