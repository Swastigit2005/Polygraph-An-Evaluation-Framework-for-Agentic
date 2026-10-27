"""Web demo glue: payload shape and API input checks (mocked LLM, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

from test_graph import make_agent

from rag_agent.config import Settings
from rag_agent.demo import MAX_QUESTION_CHARS, trace_payload
from rag_agent.retrieval import Retriever

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))


def test_payload_numbers_citations_and_lists_sources(retriever: Retriever, settings: Settings) -> None:
    agent, _ = make_agent(retriever, settings, [True], answer="Use --memory [engine/memory#000] [made/up#001].")
    p = trace_payload(agent.run("how to cap memory?"), retriever)
    assert p["answer"] == "Use --memory [1] ."
    assert [s["id"] for s in p["sources"]] == ["engine/memory#000"]
    assert p["sources"][0]["n"] == 1 and p["sources"][0]["url"] == "https://x/engine/memory#000"
    assert [s["node"] for s in p["steps"]] == ["retrieve", "grade_context", "generate"]
    assert all("stages" not in s["data"] for s in p["steps"])
    assert "engine/memory#000" in p["titles"]
    assert p["stats"]["llm_calls"] == 2 and not p["refused"]


def test_payload_for_refusal(retriever: Retriever, settings: Settings) -> None:
    agent, _ = make_agent(retriever, settings, [False, False, False], max_rewrites=2)
    p = trace_payload(agent.run("kubernetes ingress?"), retriever)
    assert p["refused"] and p["sources"] == [] and p["stats"]["rewrites"] == 2


def test_api_rejects_bad_questions_before_loading_models() -> None:
    import ask

    assert ask.answer("   ")[0] == 400
    assert ask.answer("x" * (MAX_QUESTION_CHARS + 1))[0] == 400
    assert ask._agent is None  # validation never touched the index or the LLM


def test_api_per_ip_limit() -> None:
    import ask

    ip = "203.0.113.7"
    assert all(ask._allowed(ip) for _ in range(ask.PER_IP_LIMIT))
    assert not ask._allowed(ip)
