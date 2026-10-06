"""LangGraph agent: retrieve -> grade -> (rewrite -> retrieve | generate).

Behaviour is controlled by `Settings`: retrieval mode / reranker (inside `retrieve`),
`use_reflection` (skip grading entirely when False) and `max_rewrites`.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from rag_agent.config import Settings, get_settings
from rag_agent.llm import LLM, get_llm
from rag_agent.retrieval import Retrieved, Retriever


class GradeVerdict(BaseModel):
    """Structured output of the context grader."""

    relevant: bool = Field(description="True only if the context contains the information needed to answer")
    reason: str = Field(description="One sentence justifying the verdict")


class AgentState(TypedDict, total=False):
    """State shared between graph nodes."""

    question: str
    query: str  # current search query (original or rewritten)
    hits: list[Retrieved]
    verdict: GradeVerdict | None
    rewrites: int
    answer: str


GRADER_SYSTEM = (
    "You are a strict retrieval grader for a Docker documentation assistant. Decide whether the "
    "retrieved context contains the specific information needed to answer the user's question. "
    "Topical similarity is not enough: if the answer is not in the context, mark it not relevant."
)
REWRITE_SYSTEM = (
    "You rewrite search queries for a Docker documentation search engine. Given the user's question "
    "and why the previous retrieval failed, write one concise, specific search query using Docker "
    "terminology. Output only the query text."
)
GENERATE_SYSTEM = (
    "You are a Docker support assistant. Answer using ONLY the provided context. Do not use outside "
    "knowledge. If the context does not contain the answer, say so plainly."
)


def format_context(hits: list[Retrieved]) -> str:
    """Render hits as numbered, attributed context blocks for prompts."""
    return "\n\n".join(f"[{h.chunk.id}] {h.chunk.header()}\n{h.chunk.text}" for h in hits)


def build_graph(retriever: Retriever, settings: Settings | None = None, llm: LLM | None = None) -> Any:
    """Compile the agent graph bound to a retriever, settings and LLM client."""
    s = settings or get_settings()
    llm = llm or get_llm()

    def retrieve(state: AgentState) -> AgentState:
        query = state.get("query") or state["question"]
        return {"query": query, "hits": retriever.retrieve(query, s).hits}

    def grade_context(state: AgentState) -> AgentState:
        user = f"Question: {state['question']}\n\nRetrieved context:\n{format_context(state['hits'])}"
        verdict, _ = llm.complete_structured(
            [{"role": "system", "content": GRADER_SYSTEM}, {"role": "user", "content": user}],
            s.fast_model, GradeVerdict, max_tokens=200)
        return {"verdict": verdict}

    def rewrite_query(state: AgentState) -> AgentState:
        verdict = state["verdict"]
        user = (f"Question: {state['question']}\nPrevious query: {state['query']}\n"
                f"Why it failed: {verdict.reason if verdict else 'n/a'}")
        res = llm.complete([{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": user}],
                           s.fast_model, max_tokens=60)
        new_query = res.text.strip().strip('"').splitlines()[0] if res.text.strip() else state["question"]
        return {"query": new_query, "rewrites": state.get("rewrites", 0) + 1}

    def generate(state: AgentState) -> AgentState:
        user = f"Question: {state['question']}\n\nContext:\n{format_context(state['hits'])}"
        msgs = [{"role": "system", "content": GENERATE_SYSTEM}, {"role": "user", "content": user}]
        res = llm.complete(msgs, s.gen_model)
        return {"answer": res.text}

    def route_after_grade(state: AgentState) -> str:
        verdict = state["verdict"]
        if verdict is not None and verdict.relevant:
            return "generate"
        if state.get("rewrites", 0) < s.max_rewrites:
            return "rewrite_query"
        return "generate"  # Phase 2 replaces this fallback with an explicit refuse node

    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    if s.use_reflection:
        g.add_node("grade_context", grade_context)
        g.add_node("rewrite_query", rewrite_query)
        g.add_edge("retrieve", "grade_context")
        g.add_conditional_edges("grade_context", route_after_grade, ["generate", "rewrite_query"])
        g.add_edge("rewrite_query", "retrieve")
    else:
        g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


def answer(question: str, retriever: Retriever, settings: Settings | None = None,
           llm: LLM | None = None) -> AgentState:
    """Convenience wrapper: run the graph once and return the final state."""
    graph = build_graph(retriever, settings, llm)
    return graph.invoke({"question": question, "rewrites": 0, "verdict": None})
