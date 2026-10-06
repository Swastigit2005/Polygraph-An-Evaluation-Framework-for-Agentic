"""Shared fixtures: a fake Groq client and a tiny in-memory corpus (no network, no model downloads)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag_agent.config import Settings
from rag_agent.ingest import Chunk
from rag_agent.retrieval import Retriever


class FakeRaw:
    """Mimics groq's raw response wrapper."""

    def __init__(self, text: str, prompt_tokens: int = 100, completion_tokens: int = 20) -> None:
        self.text, self.headers = text, {}
        self._usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    def parse(self) -> Any:
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))],
                               usage=self._usage)


class FakeClient:
    """Fake groq.Groq: `responder(params) -> str | Exception` decides each reply; calls are recorded."""

    def __init__(self, responder: Callable[[dict[str, Any]], Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        outer = self

        class _Raw:
            @staticmethod
            def create(**params: Any) -> FakeRaw:
                outer.calls.append(params)
                out = responder(params)
                if isinstance(out, Exception):
                    raise out
                return FakeRaw(out)

        self.chat = SimpleNamespace(completions=SimpleNamespace(with_raw_response=_Raw))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings that never touch the real cache, network or models."""
    return Settings(groq_api_key="test", gen_model="GEN", fast_model="FAST", judge_model="JUDGE",
                    retrieval_mode="bm25", use_reranker=False, top_k=2, candidate_k=5,
                    cache_dir=tmp_path / "cache", use_llm_cache=False, rpm_limit=1000, tpm_limit=10**7,
                    model_prices="GEN=1/2,FAST=0.5/1")


@pytest.fixture
def chunks() -> list[Chunk]:
    """A four-chunk toy knowledge base."""
    def c(cid: str, title: str, text: str) -> Chunk:
        return Chunk(id=cid, doc_id=cid.split("#")[0], title=title, section="", url=f"https://x/{cid}", text=text)

    return [
        c("engine/memory#000", "Resource constraints", "Use the --memory flag to limit container memory."),
        c("engine/cpu#000", "CPU constraints", "Use the --cpus flag to limit how much CPU a container uses."),
        c("compose/watch#000", "Compose watch", "Compose watch syncs files into running services."),
        c("engine/logs#000", "Logging drivers", "The json-file logging driver is the default for containers."),
    ]


@pytest.fixture
def retriever(chunks: list[Chunk], settings: Settings) -> Retriever:
    """BM25-only retriever (no embedding model needed)."""
    return Retriever(chunks, settings)


def grade_json(relevant: bool, ids: list[str] | None = None, reason: str = "test") -> str:
    """Serialized GradeVerdict as the fake grader would return it."""
    return json.dumps({"relevant": relevant, "reason": reason, "relevant_chunk_ids": ids or []})
