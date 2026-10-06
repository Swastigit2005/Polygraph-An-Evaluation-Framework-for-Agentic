"""Markdown rendering for eval results (shared by run_eval.py, make_report.py and the app)."""

from __future__ import annotations

from typing import Any

# (section, label, path into result["metrics"], format, higher_is_better)
ROWS: list[tuple[str, str, tuple[str, ...], str, bool | None]] = [
    ("Agent decisions", "Grader precision (set)", ("grader", "set_level", "precision"), "pct", True),
    ("Agent decisions", "Grader recall (set)", ("grader", "set_level", "recall"), "pct", True),
    ("Agent decisions", "Grader precision (chunk)", ("grader", "chunk_level", "precision"), "pct", True),
    ("Agent decisions", "Grader recall (chunk)", ("grader", "chunk_level", "recall"), "pct", True),
    ("Agent decisions", "Correct-refusal rate (unanswerable)", ("refusal", "correct_refusal_rate"), "pct", True),
    ("Agent decisions", "False-refusal rate (answerable)", ("refusal", "false_refusal_rate"), "pct", False),
    ("Agent decisions", "Rewrite success rate", ("rewrites", "success_rate"), "pct", True),
    ("Agent decisions", "Rewrites (answerable items)", ("rewrites", "n_rewrites"), "int", None),
    ("Agent decisions", "Steps / query (mean)", ("steps", "mean"), "f2", None),
    ("Agent decisions", "Steps / query (p95)", ("steps", "p95"), "f1", None),
    ("Agent decisions", "Loop rate (hit MAX_REWRITES)", ("steps", "loop_rate"), "pct", False),
    ("Retrieval", "Recall@k (first retrieval)", ("retrieval", "recall@{k}"), "pct", True),
    ("Retrieval", "MRR@k (first retrieval)", ("retrieval", "mrr@{k}"), "f3", True),
    ("Answer quality", "Faithfulness (Ragas)", ("answer", "faithfulness"), "f3", True),
    ("Answer quality", "Answer relevancy (Ragas)", ("answer", "answer_relevancy"), "f3", True),
    ("Answer quality", "Correctness (G-Eval, answered)", ("answer", "correctness_mean"), "f3", True),
    ("Answer quality", "End-to-end accuracy (answerable)", ("answer", "end_to_end_accuracy"), "pct", True),
    ("Ops", "Latency p50 (s)", ("latency_s", "p50"), "f2", False),
    ("Ops", "Latency p95 (s)", ("latency_s", "p95"), "f2", False),
    ("Ops", "Tokens / query (mean)", ("tokens_per_query", "mean"), "int", False),
    ("Ops", "Cost / query (USD, list price)", ("cost_per_query_usd", "mean"), "usd", False),
]


def get_path(d: dict[str, Any], path: tuple[str, ...], k: int) -> Any:
    """Follow `path` into nested dicts ('{k}' is replaced by top_k); None if missing."""
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key.format(k=k))
    return cur


def fmt(v: Any, kind: str) -> str:
    """Format a metric value; missing values render as an em dash."""
    if v is None:
        return "—"
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if kind == "int":
        return f"{round(v):,}"
    if kind == "usd":
        return f"${v:.5f}"
    return f"{v:.{int(kind[1])}f}"


def metrics_table(results: dict[str, dict[str, Any]]) -> str:
    """Markdown table: one row per metric, one column per config result."""
    names = list(results)
    head = "| Metric | " + " | ".join(f"`{n}`" for n in names) + " |"
    sep = "|---|" + "---:|" * len(names)
    lines = [head, sep]
    section = None
    for sec, label, path, kind, _ in ROWS:
        vals = [get_path(results[n]["metrics"], path, results[n]["config"]["top_k"]) for n in names]
        if all(v is None for v in vals):
            continue
        if sec != section:
            lines.append(f"| **{sec}** |" + " |" * len(names))
            section = sec
        lines.append(f"| {label} | " + " | ".join(fmt(v, kind) for v in vals) + " |")
    ns = [f"{results[n]['n_evaluated']}" for n in names]
    lines.append("| Items evaluated | " + " | ".join(ns) + " |")
    return "\n".join(lines)
