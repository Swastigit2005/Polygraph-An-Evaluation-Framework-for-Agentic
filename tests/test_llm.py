"""LLM wrapper: rate limiter, cache, retries, quota handling, caps. No network."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from conftest import FakeClient
from pydantic import BaseModel

from rag_agent import llm as llm_mod
from rag_agent.config import Settings
from rag_agent.llm import LLM, LLMError, QuotaExhausted, RateLimited, RateLimiter, parse_prices


class Fake429(Exception):
    """Looks like groq.RateLimitError to the wrapper."""

    def __init__(self, msg: str = "rate limited", retry_after: str | None = None) -> None:
        super().__init__(msg)
        self.status_code = 429
        self.response = SimpleNamespace(headers={"retry-after": retry_after} if retry_after else {})


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(llm_mod.time, "sleep", recorded.append)
    monkeypatch.setattr(llm_mod.random, "uniform", lambda a, b: 0.0)
    return recorded


def test_parse_prices() -> None:
    assert parse_prices("openai/gpt-oss-20b=0.075/0.30, m=1/2") == {"openai/gpt-oss-20b": (0.075, 0.30),
                                                                   "m": (1.0, 2.0)}


def test_rate_limiter_blocks_until_window_frees() -> None:
    now = [0.0]
    waits: list[float] = []

    def sleep(s: float) -> None:
        waits.append(s)
        now[0] += s

    rl = RateLimiter(rpm=2, tpm=1000, clock=lambda: now[0], sleep=sleep)
    rl.acquire(10)
    rl.acquire(10)
    assert waits == []
    rl.acquire(10)  # third request in the same minute must wait ~60s
    assert sum(waits) == pytest.approx(60.05)


def test_rate_limiter_token_budget() -> None:
    now = [0.0]

    def sleep(s: float) -> None:
        now[0] += s

    rl = RateLimiter(rpm=100, tpm=100, clock=lambda: now[0], sleep=sleep)
    h = rl.acquire(80)
    rl.record_actual(h, 30)  # actual usage lower than estimated frees budget
    rl.acquire(60)
    assert now[0] == 0.0


def test_retry_honours_retry_after(settings: Settings, sleeps: list[float]) -> None:
    replies: list[Any] = [Fake429(retry_after="7"), "hello"]
    llm = LLM(settings, client=FakeClient(lambda p: replies.pop(0)))
    assert llm.complete([{"role": "user", "content": "hi"}], "GEN").text == "hello"
    assert sleeps == [7.0]


def test_long_retry_after_means_daily_quota(settings: Settings, sleeps: list[float]) -> None:
    llm = LLM(settings, client=FakeClient(lambda p: Fake429("Please try again in 7m12.5s.")))
    with pytest.raises(QuotaExhausted):
        llm.complete([{"role": "user", "content": "hi"}], "GEN")
    assert sleeps == []


def test_gives_up_after_max_retries(settings: Settings, sleeps: list[float]) -> None:
    s = settings.with_overrides(max_retries=2)
    llm = LLM(s, client=FakeClient(lambda p: Fake429()))
    with pytest.raises(RateLimited):
        llm.complete([{"role": "user", "content": "hi"}], "GEN")
    assert sleeps == [2.0, 4.0]  # exponential backoff


def test_request_too_large_is_not_retried(settings: Settings, sleeps: list[float]) -> None:
    llm = LLM(settings, client=FakeClient(lambda p: Fake429("Request too large for model on OTPM")))
    with pytest.raises(LLMError, match="too large"):
        llm.complete([{"role": "user", "content": "hi"}], "GEN")
    assert sleeps == []


def test_disk_cache_hit_skips_api_and_marks_cached(settings: Settings) -> None:
    s = settings.with_overrides(use_llm_cache=True)
    client = FakeClient(lambda p: "cached answer")
    llm = LLM(s, client=client)
    msgs = [{"role": "user", "content": "same prompt"}]
    first, second = llm.complete(msgs, "GEN"), llm.complete(msgs, "GEN")
    assert len(client.calls) == 1
    assert not first.usage.cached and second.usage.cached
    assert second.usage.prompt_tokens == first.usage.prompt_tokens


def test_per_model_output_cap(settings: Settings) -> None:
    client = FakeClient(lambda p: "x")
    llm = LLM(settings.with_overrides(model_max_output="JUDGE=900"), client=client)
    llm.complete([{"role": "user", "content": "hi"}], "JUDGE", max_tokens=2048)
    llm.complete([{"role": "user", "content": "hi"}], "GEN", max_tokens=2048)
    assert [c["max_completion_tokens"] for c in client.calls] == [900, 2048]


class Verdict(BaseModel):
    ok: bool


def test_structured_falls_back_to_json_mode_for_configured_models(settings: Settings) -> None:
    client = FakeClient(lambda p: '{"ok": true}')
    llm = LLM(settings.with_overrides(json_mode_models="JUDGE"), client=client)
    obj, _ = llm.complete_structured([{"role": "user", "content": "q"}], "JUDGE", Verdict)
    assert obj.ok is True
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert "JSON schema" in client.calls[0]["messages"][-1]["content"]
