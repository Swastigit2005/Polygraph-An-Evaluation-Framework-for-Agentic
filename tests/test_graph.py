"""Graph routing with a mocked LLM: no network."""

from __future__ import annotations

from typing import Any

from conftest import FakeClient, grade_json

from rag_agent.config import Settings
from rag_agent.graph import REFUSAL_TEXT, Agent, is_refusal, normalize_citations
from rag_agent.llm import LLM
from rag_agent.retrieval import Retriever


def make_agent(retriever: Retriever, settings: Settings, grades: list[bool], answer: str = "ok",
               **overrides: Any) -> tuple[Agent, FakeClient]:
    """Agent whose grader returns `grades` in order; rewrites return a fixed query."""
    queue = list(grades)

    def responder(p: dict[str, Any]) -> str:
        if p.get("response_format"):  # grader (structured output)
            rel = queue.pop(0)
            return grade_json(rel, ["engine/memory#000"] if rel else [])
        if p["model"] == "FAST":  # rewriter
            return "docker container memory limit"
        return answer  # generator

    client = FakeClient(responder)
    s = settings.with_overrides(**overrides)
    return Agent(retriever, s, LLM(s, client=client)), client


def nodes(trace: Any) -> list[str]:
    return [st.node for st in trace.steps]


def test_grader_fails_three_times_reaches_refuse(retriever: Retriever, settings: Settings) -> None:
    agent, client = make_agent(retriever, settings, [False, False, False], max_rewrites=2)
    t = agent.run("how to cap memory?")
    assert nodes(t) == ["retrieve", "grade_context", "rewrite_query", "retrieve", "grade_context",
                        "rewrite_query", "retrieve", "grade_context", "refuse"]
    assert t.refused and t.answer == REFUSAL_TEXT and t.citations == []
    assert all(c["model"] != "GEN" for c in client.calls)  # never answers from parametric knowledge
    assert t.rewrites == 2


def test_grader_passes_after_one_rewrite(retriever: Retriever, settings: Settings) -> None:
    agent, _ = make_agent(retriever, settings, [False, True], answer="Use --memory [engine/memory#000].")
    t = agent.run("my container eats all the RAM")
    assert nodes(t) == ["retrieve", "grade_context", "rewrite_query", "retrieve", "grade_context", "generate"]
    assert not t.refused
    assert [c["id"] for c in t.citations] == ["engine/memory#000"]
    assert t.citations[0]["source"] == "inline"
    assert t.nodes("rewrite_query")[0].data["to"] == "docker container memory limit"


def test_max_rewrites_zero_refuses_immediately(retriever: Retriever, settings: Settings) -> None:
    agent, _ = make_agent(retriever, settings, [False], max_rewrites=0)
    assert nodes(agent.run("q")) == ["retrieve", "grade_context", "refuse"]


def test_no_reflection_goes_straight_to_generate(retriever: Retriever, settings: Settings) -> None:
    agent, client = make_agent(retriever, settings, [], answer="Use --memory.", use_reflection=False)
    t = agent.run("limit memory")
    assert nodes(t) == ["retrieve", "generate"]
    assert not any(c.get("response_format") for c in client.calls)
    # No inline citation: falls back to the retrieved context.
    assert t.citations and t.citations[0]["source"] == "context"


def test_generator_refusal_is_detected(retriever: Retriever, settings: Settings) -> None:
    agent, _ = make_agent(retriever, settings, [], answer=f'"{REFUSAL_TEXT}"', use_reflection=False)
    t = agent.run("kubernetes ingress?")
    assert t.refused and t.answer == REFUSAL_TEXT


def test_trace_accounts_tokens_and_cost(retriever: Retriever, settings: Settings) -> None:
    agent, _ = make_agent(retriever, settings, [True])
    t = agent.run("limit memory")
    # FakeRaw: 100 prompt + 20 completion tokens per call; grader on FAST, generator on GEN.
    assert t.total_tokens == 240
    assert t.cost_usd == round((100 * 0.5 + 20 * 1) / 1e6 + (100 * 1 + 20 * 2) / 1e6, 8)
    d = t.to_dict()
    assert d["rewrites"] == 0 and d["steps"][1]["data"]["relevant"] is True


def test_is_refusal_and_citation_normalisation() -> None:
    assert is_refusal("I couldn’t find this in the knowledge base")
    assert not is_refusal("Use the --memory flag.")
    assert normalize_citations("see 【a/b#001】") == "see [a/b#001]"
