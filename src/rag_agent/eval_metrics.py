"""Pure metric functions over agent traces and golden items (no LLM calls, unit-tested).

Definitions (also documented in the README):

* Retrieval — computed on the FIRST retrieval (original question), answerable items only:
  recall@k = |gold ∩ top-k| / |gold|;  MRR = 1 / rank of the first gold chunk in top-k (0 if none).
* Grader (set level) — one observation per grade_context step. Predicted positive = verdict
  relevant. Gold positive = the graded chunks contain at least one gold chunk (always negative
  for unanswerable items). Precision / recall / F1 / accuracy over all steps.
* Grader (chunk level) — one observation per graded chunk. Predicted positive = chunk listed in
  the verdict's relevant_chunk_ids. Gold positive = chunk in relevant_chunk_ids of the item.
* Refusal — correct-refusal rate = refused / unanswerable items; false-refusal rate =
  refused / answerable items.
* Rewrite success — over rewrites on answerable items: the next grade passed AND the retrieval
  after the rewrite contained a gold chunk.
* Steps — number of node executions per query (mean, p95). Loop rate — share of queries whose
  rewrite count reached MAX_REWRITES.
* Latency — wall time, except that LLM calls served from the disk cache are charged their
  originally measured API latency, so cached re-runs report realistic latency.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

Trace = dict[str, Any]  # Trace.to_dict() output
Item = dict[str, Any]  # golden item


# --------------------------------------------------------------------------- basics
def safe_div(num: float, den: float) -> float | None:
    """num / den, or None when den == 0 (so empty buckets read as n/a, not 0)."""
    return None if den == 0 else num / den


def mean(xs: Iterable[float]) -> float | None:
    """Arithmetic mean, or None for an empty input."""
    xs = list(xs)
    return safe_div(sum(xs), len(xs))


def percentile(xs: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile (p in [0, 100]), matching numpy's default method."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def prf(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int | None]:
    """Precision, recall, F1, accuracy plus the raw confusion counts."""
    p, r = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    f1 = None if p is None or r is None or p + r == 0 else 2 * p * r / (p + r)
    return {"precision": p, "recall": r, "f1": f1, "accuracy": safe_div(tp + tn, tp + fp + fn + tn),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# --------------------------------------------------------------------------- retrieval
def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float | None:
    """|gold ∩ retrieved[:k]| / |gold|; None if there is no gold."""
    g = set(gold)
    return safe_div(len(g & set(retrieved[:k])), len(g))


def reciprocal_rank(retrieved: Sequence[str], gold: Iterable[str], k: int | None = None) -> float:
    """1 / (1-based rank of the first gold id in retrieved[:k]); 0 if none."""
    g = set(gold)
    for rank, cid in enumerate(retrieved[:k] if k else retrieved, start=1):
        if cid in g:
            return 1.0 / rank
    return 0.0


def _steps(trace: Trace, node: str) -> list[dict[str, Any]]:
    return [st for st in trace["steps"] if st["node"] == node]


def first_retrieval(trace: Trace) -> list[str]:
    """Chunk ids from the first retrieve step (the original question)."""
    steps = _steps(trace, "retrieve")
    return steps[0]["data"]["retrieved_ids"] if steps else []


# --------------------------------------------------------------------------- grader
def grader_counts(trace: Trace, item: Item) -> tuple[list[int], list[int]]:
    """Confusion counts [tp, fp, fn, tn] at set level and chunk level for one trace."""
    gold = set(item.get("relevant_chunk_ids") or [])
    answerable = not item.get("should_refuse", False)
    set_c, chunk_c = [0, 0, 0, 0], [0, 0, 0, 0]
    for st in _steps(trace, "grade_context"):
        d = st["data"]
        graded = d["graded_ids"]
        gold_pos = answerable and bool(gold & set(graded))
        pred_pos = bool(d["relevant"])
        set_c[_cell(pred_pos, gold_pos)] += 1
        marked = set(d.get("relevant_chunk_ids") or [])
        for cid in graded:
            chunk_c[_cell(cid in marked, answerable and cid in gold)] += 1
    return set_c, chunk_c


def _cell(pred: bool, gold: bool) -> int:
    """Index into [tp, fp, fn, tn]."""
    return 0 if pred and gold else 1 if pred else 2 if gold else 3


# --------------------------------------------------------------------------- rewrites / steps
def rewrite_outcomes(trace: Trace, item: Item) -> list[bool]:
    """For each rewrite: did the following grade pass AND the following retrieval include a gold chunk?"""
    gold = set(item.get("relevant_chunk_ids") or [])
    steps = trace["steps"]
    out: list[bool] = []
    for i, st in enumerate(steps):
        if st["node"] != "rewrite_query":
            continue
        nxt_ret = next((s for s in steps[i + 1:] if s["node"] == "retrieve"), None)
        nxt_grade = next((s for s in steps[i + 1:] if s["node"] == "grade_context"), None)
        hit = bool(nxt_ret and gold & set(nxt_ret["data"]["retrieved_ids"]))
        passed = bool(nxt_grade and nxt_grade["data"]["relevant"])
        out.append(hit and passed)
    return out


def effective_latency(step_or_trace_latency: float, usages: Iterable[dict[str, Any]]) -> float:
    """Wall latency plus the original API latency of calls that were served from cache."""
    return step_or_trace_latency + sum(u["latency_s"] for u in usages if u.get("cached"))


# --------------------------------------------------------------------------- aggregate
def aggregate(records: list[dict[str, Any]], max_rewrites: int, top_k: int) -> dict[str, Any]:
    """Compute all trace-derived metrics over per-item records {item, trace, judge?}."""
    ans = [r for r in records if not r["item"]["should_refuse"]]
    unans = [r for r in records if r["item"]["should_refuse"]]

    # Retrieval (first retrieval, answerable only)
    rec_k = [recall_at_k(first_retrieval(r["trace"]), r["item"]["relevant_chunk_ids"], top_k) for r in ans]
    mrr = [reciprocal_rank(first_retrieval(r["trace"]), r["item"]["relevant_chunk_ids"], top_k) for r in ans]

    # Grader
    set_tot, chunk_tot = [0, 0, 0, 0], [0, 0, 0, 0]
    for r in records:
        s_c, c_c = grader_counts(r["trace"], r["item"])
        set_tot = [a + b for a, b in zip(set_tot, s_c, strict=True)]
        chunk_tot = [a + b for a, b in zip(chunk_tot, c_c, strict=True)]
    n_grades = sum(set_tot)

    # Rewrites
    outcomes = [o for r in ans for o in rewrite_outcomes(r["trace"], r["item"])]

    # Steps, loops, ops
    steps = [len(r["trace"]["steps"]) for r in records]
    loops = [r["trace"]["rewrites"] >= max_rewrites and max_rewrites > 0 and
             any(st["node"] == "rewrite_query" for st in r["trace"]["steps"]) for r in records]
    lat = [effective_latency(r["trace"]["latency_s"], _usages(r["trace"])) for r in records]
    node_lat: dict[str, list[float]] = {}
    for r in records:
        for st in r["trace"]["steps"]:
            node_lat.setdefault(st["node"], []).append(effective_latency(st["latency_s"], st["usage"]))
    tokens = [r["trace"]["total_tokens"] for r in records]
    costs = [r["trace"]["cost_usd"] for r in records if r["trace"]["cost_usd"] is not None]

    return {
        "n_items": len(records), "n_answerable": len(ans), "n_unanswerable": len(unans),
        "retrieval": {f"recall@{top_k}": mean(x for x in rec_k if x is not None), f"mrr@{top_k}": mean(mrr)},
        "grader": {"n_grades": n_grades, "set_level": prf(*set_tot) if n_grades else None,
                   "chunk_level": prf(*chunk_tot) if n_grades else None},
        "refusal": {
            "correct_refusal_rate": mean(float(r["trace"]["refused"]) for r in unans),
            "false_refusal_rate": mean(float(r["trace"]["refused"]) for r in ans),
        },
        "rewrites": {"n_rewrites": len(outcomes), "success_rate": mean(float(o) for o in outcomes),
                     "rewrites_on_unanswerable": sum(r["trace"]["rewrites"] for r in unans)},
        "steps": {"mean": mean(steps), "p95": percentile(steps, 95), "loop_rate": mean(float(x) for x in loops)},
        "latency_s": {"p50": percentile(lat, 50), "p95": percentile(lat, 95),
                      "per_node": {n: {"p50": percentile(v, 50), "p95": percentile(v, 95)}
                                   for n, v in sorted(node_lat.items())}},
        "tokens_per_query": {"mean": mean(tokens), "p95": percentile(tokens, 95)},
        "cost_per_query_usd": {"mean": mean(costs), "total": sum(costs) if costs else None,
                               "priced_queries": len(costs)},
    }


def _usages(trace: Trace) -> list[dict[str, Any]]:
    return [u for st in trace["steps"] for u in st["usage"]]


def aggregate_judges(records: list[dict[str, Any]], correct_threshold: float = 0.5) -> dict[str, Any]:
    """Answer-quality metrics from per-item judge scores (answerable items only)."""
    ans = [r for r in records if not r["item"]["should_refuse"]]
    answered = [r for r in ans if not r["trace"]["refused"] and r.get("judge")]

    def col(name: str) -> list[float]:
        return [r["judge"][name] for r in answered if r["judge"].get(name) is not None]

    corr = col("correctness")
    n_correct = sum(1 for x in corr if x >= correct_threshold)
    return {
        "n_answered": len(answered),
        "n_ragas_scored": len(col("faithfulness")),
        "faithfulness": mean(col("faithfulness")),
        "answer_relevancy": mean(col("answer_relevancy")),
        "correctness_mean": mean(corr),
        "correct_rate_answered": safe_div(n_correct, len(corr)),
        # Refused or unjudged answerable items count as incorrect end-to-end.
        "end_to_end_accuracy": safe_div(n_correct, len(ans)),
        "judge_errors": sum(1 for r in answered if r["judge"].get("errors")),
    }
