"""LLM-as-judge metrics: Ragas faithfulness / answer relevancy and a DeepEval G-Eval correctness metric.

Both libraries default to OpenAI. Here every judge call is routed through our own `LLM` wrapper
(Groq, rate-limited, disk-cached, quota-aware) and embeddings come from the local
sentence-transformers model, so no OpenAI key is ever needed. `assert_no_openai()` makes any
accidental fallback fail loudly.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, TypeVar

from pydantic import BaseModel

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from deepeval.metrics import GEval  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # noqa: E402
from ragas.embeddings.base import BaseRagasEmbedding  # noqa: E402
from ragas.llms.base import InstructorBaseRagasLLM  # noqa: E402
from ragas.metrics.collections import AnswerRelevancy, Faithfulness  # noqa: E402

from rag_agent.config import Settings  # noqa: E402
from rag_agent.llm import LLM, QuotaExhausted, Usage  # noqa: E402
from rag_agent.retrieval import get_embedder  # noqa: E402

T = TypeVar("T", bound=BaseModel)

CORRECTNESS_STEPS = [
    "Identify the key facts in the expected output that answer the input question.",
    "Check whether the actual output states those key facts correctly; paraphrasing, different "
    "formatting and different but equivalent commands are acceptable.",
    "Heavily penalize any statement in the actual output that contradicts the expected output.",
    "Penalize omission of key facts from the expected output; do not penalize additional correct, "
    "relevant detail.",
    "If the actual output declines to answer or says the information is unavailable, give the lowest score.",
]


def assert_no_openai() -> None:
    """Refuse to run evals with an OpenAI key present, so a silent OpenAI fallback cannot happen."""
    if os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is set; unset it. All judges must run on Groq.")


class _UsageLog:
    """Collects Usage records from judge calls for token/cost accounting."""

    def __init__(self) -> None:
        self.usages: list[Usage] = []

    def drain(self) -> list[Usage]:
        out, self.usages = self.usages, []
        return out


class GroqRagasLLM(InstructorBaseRagasLLM):
    """Ragas structured-output LLM backed by our Groq wrapper."""

    def __init__(self, llm: LLM, model: str, log: _UsageLog, reasoning_effort: str | None = None) -> None:
        self.llm, self.model, self.log, self.reasoning_effort = llm, model, log, reasoning_effort

    def generate(self, prompt: str, response_model: type[T]) -> T:
        """Return `response_model` parsed from the judge's structured output."""
        obj, res = self.llm.complete_structured([{"role": "user", "content": prompt}], self.model, response_model,
                                                max_tokens=2048, reasoning_effort=self.reasoning_effort)
        self.log.usages.append(res.usage)
        return obj

    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        """Async wrapper (the rate limiter is thread-safe)."""
        return await asyncio.to_thread(self.generate, prompt, response_model)


class LocalRagasEmbeddings(BaseRagasEmbedding):
    """Ragas embeddings backed by the local sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        """Embed one text (normalised)."""
        return get_embedder(self.model_name).encode([text], normalize_embeddings=True)[0].tolist()

    def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Embed many texts (normalised)."""
        return get_embedder(self.model_name).encode(texts, normalize_embeddings=True).tolist()

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        """Async single-text embedding."""
        return self.embed_text(text)

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Async batch embedding."""
        return self.embed_texts(texts)


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """DeepEval custom model backed by our Groq wrapper."""

    def __init__(self, llm: LLM, model: str, log: _UsageLog, reasoning_effort: str | None = None) -> None:
        # Note: DeepEvalBaseLLM.__init__ sets self.model = self.load_model(), so keep the id elsewhere.
        self.llm, self.model_id, self.log, self.reasoning_effort = llm, model, log, reasoning_effort
        super().__init__(model_name=model)

    def load_model(self) -> Any:
        """DeepEval hook; the client is already loaded."""
        return self

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        """Return a parsed `schema` instance when given, else plain text."""
        msgs = [{"role": "user", "content": prompt}]
        if schema is not None:
            obj, res = self.llm.complete_structured(msgs, self.model_id, schema, max_tokens=1024,
                                                    reasoning_effort=self.reasoning_effort)
            self.log.usages.append(res.usage)
            return obj
        res = self.llm.complete(msgs, self.model_id, max_tokens=1024, reasoning_effort=self.reasoning_effort)
        self.log.usages.append(res.usage)
        return res.text

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        """Async wrapper."""
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        """Model id shown in DeepEval reports."""
        return self.model_id


class Judge:
    """Scores one answer with faithfulness, answer relevancy and correctness."""

    def __init__(self, s: Settings, llm: LLM | None = None, relevancy_strictness: int = 1) -> None:
        assert_no_openai()
        model = s.judge_model or s.fast_model
        if not model:
            raise RuntimeError("Set JUDGE_MODEL (ideally a different model family from GEN_MODEL)")
        self.model = model
        self.llm = llm or LLM(s)
        self.log = _UsageLog()
        effort = s.judge_reasoning_effort or None
        ragas_llm = GroqRagasLLM(self.llm, model, self.log, effort)
        self.faithfulness = Faithfulness(llm=ragas_llm)
        self.relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=LocalRagasEmbeddings(s.embed_model),
                                         strictness=relevancy_strictness)
        self.correctness = GEval(
            name="Correctness",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT],
            evaluation_steps=CORRECTNESS_STEPS,
            model=GroqDeepEvalLLM(self.llm, model, self.log, effort),
            async_mode=False,
        )

    def correctness_score(self, question: str, answer: str, reference: str) -> tuple[float, str]:
        """G-Eval correctness in [0, 1] plus the judge's reason."""
        tc = LLMTestCase(input=question, actual_output=answer, expected_output=reference)
        self.correctness.measure(tc, _show_indicator=False)
        return float(self.correctness.score), str(self.correctness.reason or "")

    def score(self, question: str, answer: str, contexts: list[str], reference: str | None,
              ragas: bool = True) -> dict[str, Any]:
        """Run all judge metrics; failures are recorded per metric instead of aborting the run.

        Quota exhaustion is re-raised so the caller can stop and resume later; other errors
        (including per-minute rate limiting after all retries) leave that metric as None.
        """
        out: dict[str, Any] = {"judge_model": self.model, "errors": {}}
        jobs = {
            "faithfulness": lambda: asyncio.run(self.faithfulness.ascore(
                user_input=question, response=answer, retrieved_contexts=contexts)).value,
            "answer_relevancy": lambda: asyncio.run(self.relevancy.ascore(user_input=question, response=answer)).value,
        }
        for name, fn in (jobs.items() if ragas else ()):
            try:
                out[name] = float(fn())
            except QuotaExhausted:
                raise
            except Exception as exc:  # noqa: BLE001 - judge parsing failures should not kill a run
                out[name], out["errors"][name] = None, repr(exc)[:300]
        if reference:
            try:
                out["correctness"], out["correctness_reason"] = self.correctness_score(question, answer, reference)
            except QuotaExhausted:
                raise
            except Exception as exc:  # noqa: BLE001
                out["correctness"], out["errors"]["correctness"] = None, repr(exc)[:300]
        usages = self.log.drain()
        out["tokens"] = sum(u.total_tokens for u in usages)
        costs = [u.cost_usd for u in usages]
        out["cost_usd"] = None if any(c is None for c in costs) else sum(costs)  # type: ignore[arg-type]
        if not out["errors"]:
            del out["errors"]
        return out
