"""RRF fusion, tokenization and BM25 retrieval."""

from __future__ import annotations

import pytest

from rag_agent.retrieval import Retriever, rrf_fuse, tokenize


def test_rrf_hand_computed() -> None:
    # k=60. a: 1/61 + 1/62; b: 1/62 + 1/61; c: 1/63; d: 1/63.
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 63)


def test_rrf_order_and_tie_break_by_first_appearance() -> None:
    order = [i for i, _ in rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)]
    assert order == ["a", "b", "c", "d"]  # a/b tie -> a first seen; c/d tie -> c first seen


def test_rrf_rewards_agreement_over_single_top_rank() -> None:
    # x is ranked 2nd by both lists; y is 1st in one list only.
    order = [i for i, _ in rrf_fuse([["y", "x"], ["z", "x"]], k=1)]
    assert order[0] == "x"  # 1/3 + 1/3 = 0.667 > 1/2


def test_tokenize_keeps_docker_terms_and_drops_stopwords() -> None:
    assert tokenize("How do I edit daemon.json for docker-compose?") == ["edit", "daemon.json", "docker-compose"]


def test_bm25_retrieval_finds_lexical_match(retriever: Retriever) -> None:
    # In this 4-doc corpus "limit"/"container" occur in half the docs, so their BM25 IDF is 0;
    # "memory" and "cpu" are the discriminative terms.
    res = retriever.retrieve("limit container memory and cpu")
    assert res.ids[0] in {"engine/memory#000", "engine/cpu#000"}
    assert set(res.ids) == {"engine/memory#000", "engine/cpu#000"}  # top_k = 2
    assert set(res.stages) == {"bm25"}


def test_bm25_drops_zero_score_chunks(retriever: Retriever) -> None:
    assert retriever.bm25("kubernetes", 5) == []
