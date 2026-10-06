"""LangGraph agent: retrieve -> grade_context -> (rewrite_query -> retrieve | generate | refuse).

Each node appends a structured `StepRecord` (what it decided, why, latency, LLM usage) to the
run's trace. The trace is what the evaluation harness scores, and what the UI shows.

Behaviour is controlled by `Settings`: retrieval mode / reranker (inside `retrieve`),
`use_reflection` (when False: retrieve -> generate, no grading, no refuse node) and
`max_rewrites` (grader failures allowed before refusing).
"""

from __future__ import annotations

import operator
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from rag_agent.config import Settings, get_settings
from rag_agent.llm import LLM, LLMResult, Usage, get_llm
from rag_agent.retrieval import Retrieved, Retriever

REFUSAL_TEXT = "I couldn't find this in the knowledge base."


# --------------------------------------------------------------------------- records
class GradeVerdict(BaseModel):
    """Structured output of the context grader."""

    relevant: bool = Field(description="True only if the context contains the information needed to answer")
    reason: str = Field(description="One sentence justifying the verdict")
    relevant_chunk_ids: list[str] = Field(
        default_factory=list, description="Ids of the chunks that contain answer-relevant information")


@dataclass
class StepRecord:
    """One node execution: what the agent did and decided at that step."""

    node: str
    latency_s: float
    data: dict[str, Any]
    usage: list[Usage] = field(default_factory=list)


@dataclass
class Trace:
    """Everything that happened during one agent run."""

    question: str
    steps: list[StepRecord]
    answer: str
    refused: bool
    citations: list[dict[str, str]]
    config: dict[str, Any]
    latency_s: float = 0.0

    @property
    def usages(self) -> list[Usage]:
        """All LLM usage records in call order."""
        return [u for st in self.steps for u in st.usage]

    @property
    def total_tokens(self) -> int:
        """Prompt + completion tokens over all LLM calls."""
        return sum(u.total_tokens for u in self.usages)

    @property
    def cost_usd(self) -> float | None:
        """Summed list-price cost; None if any call's model has no configured price."""
        costs = [u.cost_usd for u in self.usages]
        return None if any(c is None for c in costs) else round(sum(costs), 8)  # type: ignore[arg-type]

    @property
    def rewrites(self) -> int:
        """Number of query rewrites performed."""
        return sum(1 for st in self.steps if st.node == "rewrite_query")

    def nodes(self, name: str) -> list[StepRecord]:
        """All steps executed by node `name`, in order."""
        return [st for st in self.steps if st.node == name]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (for eval results and the UI)."""
        d = asdict(self)
        d.update(total_tokens=self.total_tokens, cost_usd=self.cost_usd, rewrites=self.rewrites,
                 prompt_tokens=sum(u.prompt_tokens for u in self.usages),
                 completion_tokens=sum(u.completion_tokens for u in self.usages))
        return d


class AgentState(TypedDict, total=False):
    """State shared between graph nodes."""

    question: str
    query: str  # current search query (original or rewritten)
    hits: list[Retrieved]
    verdict: GradeVerdict | None
    rewrites: int
    answer: str
    refused: bool
    citations: list[dict[str, str]]
    steps: Annotated[list[StepRecord], operator.add]


# --------------------------------------------------------------------------- prompts
GRADER_SYSTEM = (
    "You are a strict retrieval grader for a Docker documentation assistant. Decide whether the "
    "retrieved context contains the specific information needed to answer the user's question. "
    "Topical similarity is not enough: if the answer is not stated in the context, set relevant=false. "
    "List the ids (the text inside [brackets]) of every chunk that contains answer-relevant information."
)
REWRITE_SYSTEM = (
    "You rewrite search queries for a Docker documentation search engine. Given the user's question "
    "and why the previous retrieval failed, write one concise, specific search query using Docker "
    "terminology. Do not add facts that are not implied by the question. Output only the query text."
)
GENERATE_SYSTEM = (
    "You are a Docker support assistant. Answer using ONLY the provided context; never use outside "
    "knowledge. Cite the chunk ids you used inline in square brackets, e.g. [engine/install/ubuntu#003]. "
    f"If the context does not contain the answer, reply with exactly: {REFUSAL_TEXT}"
)

# Models sometimes cite with fullwidth lenticular brackets 【id】; accept and normalise both.
_CITE = re.compile(r"[\[【]([^\[\]【】\s]+#\d{3})[\]】]")


def normalize_citations(answer: str) -> str:
    """Rewrite 【chunk#001】 citations to [chunk#001]."""
    return _CITE.sub(lambda m: f"[{m.group(1)}]", answer)


def format_context(hits: list[Retrieved]) -> str:
    """Render hits as id-tagged, attributed context blocks for prompts."""
    return "\n\n".join(f"[{h.chunk.id}] {h.chunk.header()}\n{h.chunk.text}" for h in hits)


def is_refusal(text: str) -> bool:
    """True if a generated answer is the standard refusal (allowing trivial punctuation/quote drift)."""
    norm = text.strip().strip("\"'*` ").lower().replace("’", "'")
    return norm.startswith(REFUSAL_TEXT.lower().rstrip("."))


def extract_citations(answer: str, hits: list[Retrieved]) -> list[str]:
    """Chunk ids cited inline that were actually retrieved, in order of first citation."""
    retrieved = {h.chunk.id for h in hits}
    seen: list[str] = []
    for cid in _CITE.findall(answer):
        if cid in retrieved and cid not in seen:
            seen.append(cid)
    return seen


def citation_records(ids: list[str], hits: list[Retrieved], source: str) -> list[dict[str, str]]:
    """Expand chunk ids into {id, title, url, source} records for display."""
    by_id = {h.chunk.id: h.chunk for h in hits}
    return [{"id": i, "title": by_id[i].header(), "url": by_id[i].url, "source": source} for i in ids]


NodeFn = Callable[["AgentState"], tuple["AgentState", dict[str, Any], list[LLMResult]]]


# --------------------------------------------------------------------------- agent
class Agent:
    """Compiled agent graph bound to a retriever, settings and LLM client."""

    def __init__(self, retriever: Retriever, settings: Settings | None = None, llm: LLM | None = None) -> None:
        self.retriever = retriever
        self.s = settings or get_settings()
        self.llm = llm or get_llm()
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def _timed(self, name: str, fn: NodeFn) -> Callable[[AgentState], AgentState]:
        def node(state: AgentState) -> AgentState:
            t0 = time.perf_counter()
            update, data, results = fn(state)
            rec = StepRecord(node=name, latency_s=round(time.perf_counter() - t0, 4), data=data,
                             usage=[r.usage for r in results])
            return {**update, "steps": [rec]}
        return node

    def _retrieve(self, state: AgentState) -> tuple[AgentState, dict[str, Any], list[LLMResult]]:
        query = state.get("query") or state["question"]
        res = self.retriever.retrieve(query, self.s)
        data = {"query": query, "retrieved_ids": res.ids, "stages": res.stages,
                "scores": [round(h.score, 4) for h in res.hits]}
        return {"query": query, "hits": res.hits}, data, []

    def _grade(self, state: AgentState) -> tuple[AgentState, dict[str, Any], list[LLMResult]]:
        hits = state["hits"]
        user = f"Question: {state['question']}\n\nRetrieved context:\n{format_context(hits)}"
        verdict, res = self.llm.complete_structured(
            [{"role": "system", "content": GRADER_SYSTEM}, {"role": "user", "content": user}],
            self.s.fast_model, GradeVerdict, max_tokens=600, reasoning_effort=self.s.fast_reasoning_effort)
        retrieved = {h.chunk.id for h in hits}
        verdict.relevant_chunk_ids = [c for c in verdict.relevant_chunk_ids if c in retrieved]
        data = {"relevant": verdict.relevant, "reason": verdict.reason,
                "relevant_chunk_ids": verdict.relevant_chunk_ids, "graded_ids": [h.chunk.id for h in hits]}
        return {"verdict": verdict}, data, [res]

    def _rewrite(self, state: AgentState) -> tuple[AgentState, dict[str, Any], list[LLMResult]]:
        verdict = state.get("verdict")
        user = (f"Question: {state['question']}\nPrevious query: {state['query']}\n"
                f"Why it failed: {verdict.reason if verdict else 'n/a'}")
        msgs = [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": user}]
        res = self.llm.complete(msgs, self.s.fast_model, max_tokens=400,
                                reasoning_effort=self.s.fast_reasoning_effort)
        lines = [ln.strip().strip('"') for ln in res.text.splitlines() if ln.strip()]
        new_query = lines[0] if lines else state["question"]
        n = state.get("rewrites", 0) + 1
        return {"query": new_query, "rewrites": n}, {"from": state["query"], "to": new_query, "attempt": n}, [res]

    def _generate(self, state: AgentState) -> tuple[AgentState, dict[str, Any], list[LLMResult]]:
        hits = state["hits"]
        user = f"Question: {state['question']}\n\nContext:\n{format_context(hits)}"
        msgs = [{"role": "system", "content": GENERATE_SYSTEM}, {"role": "user", "content": user}]
        res = self.llm.complete(msgs, self.s.gen_model, reasoning_effort=self.s.gen_reasoning_effort)
        refused = is_refusal(res.text)
        answer = REFUSAL_TEXT if refused else normalize_citations(res.text)
        cited = [] if refused else extract_citations(answer, hits)
        if refused:
            citations, source = [], "none"
        elif cited:
            citations, source = citation_records(cited, hits, "inline"), "inline"
        else:
            # No inline citations: fall back to the chunks the grader marked relevant, else all context.
            verdict = state.get("verdict")
            ids = (verdict.relevant_chunk_ids if verdict and verdict.relevant_chunk_ids else
                   [h.chunk.id for h in hits])
            source = "grader" if verdict and verdict.relevant_chunk_ids else "context"
            citations = citation_records(ids, hits, source)
        data = {"refused": refused, "cited_ids": cited, "citation_source": source,
                "context_ids": [h.chunk.id for h in hits]}
        return {"answer": answer, "refused": refused, "citations": citations}, data, [res]

    def _refuse(self, state: AgentState) -> tuple[AgentState, dict[str, Any], list[LLMResult]]:
        verdict = state.get("verdict")
        data = {"reason": f"grader rejected context after {state.get('rewrites', 0)} rewrite(s)",
                "last_grader_reason": verdict.reason if verdict else None}
        return {"answer": REFUSAL_TEXT, "refused": True, "citations": []}, data, []

    def _route_after_grade(self, state: AgentState) -> str:
        verdict = state.get("verdict")
        if verdict is not None and verdict.relevant:
            return "generate"
        if state.get("rewrites", 0) < self.s.max_rewrites:
            return "rewrite_query"
        return "refuse"

    def _build(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("retrieve", self._timed("retrieve", self._retrieve))
        g.add_node("generate", self._timed("generate", self._generate))
        g.add_edge(START, "retrieve")
        if self.s.use_reflection:
            g.add_node("grade_context", self._timed("grade_context", self._grade))
            g.add_node("rewrite_query", self._timed("rewrite_query", self._rewrite))
            g.add_node("refuse", self._timed("refuse", self._refuse))
            g.add_edge("retrieve", "grade_context")
            g.add_conditional_edges("grade_context", self._route_after_grade,
                                    ["generate", "rewrite_query", "refuse"])
            g.add_edge("rewrite_query", "retrieve")
            g.add_edge("refuse", END)
        else:
            g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)
        return g.compile()

    # ------------------------------------------------------------------ public
    def config_snapshot(self) -> dict[str, Any]:
        """The settings that determine agent behaviour (recorded in every trace)."""
        s = self.s
        return {"retrieval_mode": s.retrieval_mode, "use_reranker": s.use_reranker,
                "use_reflection": s.use_reflection, "max_rewrites": s.max_rewrites, "top_k": s.top_k,
                "candidate_k": s.candidate_k, "gen_model": s.gen_model, "fast_model": s.fast_model,
                "embed_model": s.embed_model, "rerank_model": s.rerank_model if s.use_reranker else None}

    def run(self, question: str, callbacks: list[Any] | None = None) -> Trace:
        """Answer `question` and return the full trace."""
        t0 = time.perf_counter()
        config: dict[str, Any] = {"recursion_limit": 4 * (self.s.max_rewrites + 2)}
        if callbacks:
            config["callbacks"] = callbacks
        final = self.graph.invoke({"question": question, "rewrites": 0, "verdict": None, "steps": []}, config)
        return Trace(question=question, steps=final.get("steps", []), answer=final.get("answer", ""),
                     refused=bool(final.get("refused")), citations=final.get("citations", []),
                     config=self.config_snapshot(), latency_s=round(time.perf_counter() - t0, 4))
