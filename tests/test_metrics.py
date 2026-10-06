"""Metric functions, with expected values computed by hand."""

from __future__ import annotations

from typing import Any

import pytest

from rag_agent.eval_metrics import (
    aggregate,
    aggregate_judges,
    effective_latency,
    grader_counts,
    percentile,
    prf,
    recall_at_k,
    reciprocal_rank,
    rewrite_outcomes,
)


def step(node: str, latency: float = 0.1, usage: list | None = None, **data: Any) -> dict[str, Any]:
    return {"node": node, "latency_s": latency, "data": data, "usage": usage or []}


def trace(steps: list[dict[str, Any]], refused: bool = False, latency: float = 1.0) -> dict[str, Any]:
    return {"steps": steps, "refused": refused, "latency_s": latency,
            "rewrites": sum(s["node"] == "rewrite_query" for s in steps), "total_tokens": 100, "cost_usd": 0.001}


def test_recall_at_k() -> None:
    assert recall_at_k(["a", "b", "c", "d"], ["b", "x"], k=2) == 0.5
    assert recall_at_k(["a", "b", "c", "d"], ["b", "x"], k=1) == 0.0
    assert recall_at_k(["a"], [], k=1) is None


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b", "c"], ["c"], k=2) == 0.0
    assert reciprocal_rank(["a", "b"], ["a", "b"]) == 1.0


def test_percentile_matches_numpy_linear() -> None:
    xs = [1, 2, 3, 4, 10]
    assert percentile(xs, 50) == 3.0
    assert percentile(xs, 95) == pytest.approx(8.8)  # pos 3.8 -> 4 + 0.8 * (10 - 4)
    assert percentile([], 50) is None


def test_prf() -> None:
    m = prf(tp=3, fp=1, fn=2, tn=4)
    assert m["precision"] == 0.75 and m["recall"] == 0.6
    assert m["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert m["accuracy"] == 0.7
    assert prf(0, 0, 0, 5)["precision"] is None


def test_grader_counts_set_and_chunk_level() -> None:
    item = {"relevant_chunk_ids": ["g1"], "should_refuse": False}
    t = trace([
        # step 1: gold chunk retrieved but grader says no -> set FN; chunks g1 FN, x TN
        step("grade_context", relevant=False, graded_ids=["g1", "x"], relevant_chunk_ids=[]),
        # step 2: no gold retrieved, grader says yes marking y -> set FP; chunks y FP, z TN
        step("grade_context", relevant=True, graded_ids=["y", "z"], relevant_chunk_ids=["y"]),
    ])
    set_c, chunk_c = grader_counts(t, item)
    assert set_c == [0, 1, 1, 0]
    assert chunk_c == [0, 1, 1, 2]


def test_grader_counts_unanswerable_items_are_all_negative() -> None:
    item = {"relevant_chunk_ids": [], "should_refuse": True}
    t = trace([step("grade_context", relevant=False, graded_ids=["a", "b"], relevant_chunk_ids=[])])
    assert grader_counts(t, item) == ([0, 0, 0, 1], [0, 0, 0, 2])


def test_rewrite_outcomes() -> None:
    item = {"relevant_chunk_ids": ["g"], "should_refuse": False}
    t = trace([
        step("retrieve", retrieved_ids=["x"]),
        step("grade_context", relevant=False, graded_ids=["x"], relevant_chunk_ids=[]),
        step("rewrite_query"),  # -> next retrieval misses gold: failure
        step("retrieve", retrieved_ids=["y"]),
        step("grade_context", relevant=False, graded_ids=["y"], relevant_chunk_ids=[]),
        step("rewrite_query"),  # -> gold retrieved and grade passes: success
        step("retrieve", retrieved_ids=["g"]),
        step("grade_context", relevant=True, graded_ids=["g"], relevant_chunk_ids=["g"]),
        step("generate"),
    ])
    assert rewrite_outcomes(t, item) == [False, True]


def test_effective_latency_adds_original_latency_of_cached_calls() -> None:
    usage = [{"latency_s": 2.0, "cached": True}, {"latency_s": 1.0, "cached": False}]
    assert effective_latency(1.5, usage) == 3.5


def test_aggregate_refusal_rates_steps_and_loops() -> None:
    ans_item = {"relevant_chunk_ids": ["g"], "should_refuse": False}
    unans_item = {"relevant_chunk_ids": [], "should_refuse": True}
    answered = trace([step("retrieve", retrieved_ids=["g", "x"]),
                      step("grade_context", relevant=True, graded_ids=["g", "x"], relevant_chunk_ids=["g"]),
                      step("generate")])
    false_refusal = trace([step("retrieve", retrieved_ids=["x", "g"]),
                           step("grade_context", relevant=False, graded_ids=["x", "g"], relevant_chunk_ids=[]),
                           step("rewrite_query"), step("retrieve", retrieved_ids=["x"]),
                           step("grade_context", relevant=False, graded_ids=["x"], relevant_chunk_ids=[]),
                           step("refuse")], refused=True)
    correct_refusal = trace([step("retrieve", retrieved_ids=["x"]),
                             step("grade_context", relevant=False, graded_ids=["x"], relevant_chunk_ids=[]),
                             step("rewrite_query"), step("retrieve", retrieved_ids=["y"]),
                             step("grade_context", relevant=False, graded_ids=["y"], relevant_chunk_ids=[]),
                             step("refuse")], refused=True)
    records = [{"item": ans_item, "trace": answered}, {"item": ans_item, "trace": false_refusal},
               {"item": unans_item, "trace": correct_refusal}]
    m = aggregate(records, max_rewrites=1, top_k=2)
    assert m["refusal"] == {"correct_refusal_rate": 1.0, "false_refusal_rate": 0.5}
    assert m["retrieval"]["recall@2"] == 1.0
    assert m["retrieval"]["mrr@2"] == pytest.approx((1 + 0.5) / 2)
    assert m["steps"]["mean"] == pytest.approx((3 + 6 + 6) / 3)
    assert m["steps"]["loop_rate"] == pytest.approx(2 / 3)
    # grades: TP (answered), FN (false refusal, gold in set), TN (second false-refusal grade), TN, TN
    assert m["grader"]["set_level"]["tp"] == 1 and m["grader"]["set_level"]["fn"] == 1
    assert m["grader"]["set_level"]["tn"] == 3
    assert m["rewrites"]["n_rewrites"] == 1 and m["rewrites"]["success_rate"] == 0.0


def test_aggregate_judges_counts_refusals_as_incorrect_end_to_end() -> None:
    ans = {"should_refuse": False}
    records = [
        {"item": ans, "trace": {"refused": False}, "judge": {"correctness": 0.9, "faithfulness": 1.0}},
        {"item": ans, "trace": {"refused": False}, "judge": {"correctness": 0.2, "faithfulness": 0.5}},
        {"item": ans, "trace": {"refused": True}, "judge": None},
        {"item": {"should_refuse": True}, "trace": {"refused": True}, "judge": None},
    ]
    m = aggregate_judges(records)
    assert m["correct_rate_answered"] == 0.5
    assert m["end_to_end_accuracy"] == pytest.approx(1 / 3)
    assert m["faithfulness"] == 0.75 and m["n_ragas_scored"] == 2
