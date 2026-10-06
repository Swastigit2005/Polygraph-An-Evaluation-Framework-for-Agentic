"""Thin Groq chat wrapper: client-side rate limiting, 429 backoff, disk cache, usage accounting.

Every LLM call in the project goes through `LLM.complete` / `LLM.complete_structured`
so that caching, retries and token/cost accounting are applied uniformly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import sqlite3
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from rag_agent.config import Settings, get_settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# A retry-after longer than this means a daily (not per-minute) limit was hit.
_QUOTA_RETRY_AFTER_S = 120.0


class LLMError(RuntimeError):
    """Base class for LLM wrapper failures."""


class RateLimited(LLMError):
    """Per-minute limits still exceeded after all retries (transient; try again shortly)."""


class QuotaExhausted(LLMError):
    """Daily quota exhausted; retrying today is pointless. Eval runs should stop and resume later."""


@dataclass
class Usage:
    """Token and cost accounting for a single LLM call."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0  # wall time of the original (uncached) API call
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResult:
    """Text output of a call plus its usage record."""

    text: str
    usage: Usage


def parse_prices(spec: str) -> dict[str, tuple[float, float]]:
    """Parse 'model=in/out,model2=in/out' (USD per 1M tokens) into a dict."""
    prices: dict[str, tuple[float, float]] = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        model, _, io = part.rpartition("=")
        inp, _, out = io.partition("/")
        prices[model.strip()] = (float(inp), float(out))
    return prices


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used only for client-side rate limiting."""
    return max(1, len(text) // 4)


class RateLimiter:
    """Sliding 60-second window limiter over requests and tokens. Thread-safe."""

    def __init__(self, rpm: int, tpm: int, clock: Any = time.monotonic, sleep: Any = time.sleep) -> None:
        self.rpm, self.tpm = rpm, tpm
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()
        self._clock, self._sleep = clock, sleep

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= 60.0:
            self._events.popleft()

    def acquire(self, tokens: int) -> float:
        """Block until a request costing `tokens` fits in the window; record it and return a handle."""
        tokens = min(tokens, self.tpm)  # a single huge request must still be allowed eventually
        while True:
            with self._lock:
                now = self._clock()
                self._prune(now)
                used = sum(t for _, t in self._events)
                if len(self._events) < self.rpm and used + tokens <= self.tpm:
                    self._events.append((now, tokens))
                    return now
                wait = 60.0 - (now - self._events[0][0]) + 0.05
            self._sleep(max(wait, 0.05))

    def record_actual(self, handle: float, actual: int) -> None:
        """Replace the estimate recorded under `handle` with the real token count."""
        with self._lock:
            for i, (ts, _) in enumerate(self._events):
                if ts == handle:
                    self._events[i] = (ts, actual)
                    return


class DiskCache:
    """SQLite-backed key/value cache for LLM responses."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached JSON value for `key`, or None."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store a JSON-serialisable value."""
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
            self._conn.commit()


def _retry_after_seconds(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        raw = headers.get(name)
        if raw:
            try:
                return float(raw) * scale
            except ValueError:
                pass
    # Groq also states the wait in the message body, e.g. "Please try again in 7m12.5s".
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", str(exc))
    if m:
        h, mi, s = (float(x) if x else 0.0 for x in m.groups())
        return h * 3600 + mi * 60 + s
    return None


class LLM:
    """Groq chat client with rate limiting, retries, caching and usage accounting."""

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.s = settings or get_settings()
        if client is None:
            self.s.require_llm()
            import groq

            # We handle retries ourselves so that we can honour long retry-after values.
            client = groq.Groq(api_key=self.s.groq_api_key, max_retries=0)
        self._client = client
        self._limiters: dict[str, RateLimiter] = {}  # Groq limits apply per model
        self._cache = DiskCache(self.s.cache_dir / "llm_cache.sqlite") if self.s.use_llm_cache else None
        self._prices = parse_prices(self.s.model_prices)
        self._no_json_schema: set[str] = set()  # models that rejected json_schema
        self.last_rate_headers: dict[str, str] = {}

    def _limiter(self, model: str) -> RateLimiter:
        if model not in self._limiters:
            self._limiters[model] = RateLimiter(self.s.rpm_limit, self.s.tpm_limit)
        return self._limiters[model]

    # ------------------------------------------------------------------ public API
    def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResult:
        """Run a chat completion and return its text and usage.

        `reasoning_effort` is forwarded only when set (reasoning models such as gpt-oss).
        """
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.s.temperature if temperature is None else temperature,
            "max_completion_tokens": max_tokens or self.s.max_output_tokens,
        }
        if response_format:
            params["response_format"] = response_format
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort

        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        if self._cache and (hit := self._cache.get(key)):
            usage = Usage(**{**hit["usage"], "cached": True})
            return LLMResult(text=hit["text"], usage=usage)

        result = self._call_with_retries(params)
        if self._cache:
            self._cache.set(key, {"text": result.text, "usage": asdict(result.usage)})
        return result

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        model: str,
        schema: type[T],
        *,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[T, LLMResult]:
        """Run a completion constrained to `schema` and return the parsed object and raw result.

        Uses Groq's `json_schema` structured outputs where the model supports it and falls
        back to JSON mode (schema described in the prompt) otherwise.
        """
        json_schema = schema.model_json_schema()
        if model not in self._no_json_schema:
            fmt = {"type": "json_schema",
                   "json_schema": {"name": schema.__name__, "schema": json_schema}}
            try:
                res = self.complete(messages, model, max_tokens=max_tokens, response_format=fmt,
                                    reasoning_effort=reasoning_effort)
                return schema.model_validate_json(res.text), res
            except LLMError:
                raise
            except ValidationError:
                log.warning("json_schema output from %s failed validation; retrying in JSON mode", model)
            except Exception as exc:  # noqa: BLE001 - provider-specific 400 for unsupported feature
                if getattr(exc, "status_code", None) != 400:
                    raise
                log.info("Model %s rejected json_schema (%s); using JSON mode", model, exc)
                self._no_json_schema.add(model)

        hint = ("Respond with a single JSON object (no prose) that validates against this JSON schema:\n"
                + json.dumps(json_schema))
        msgs = [*messages[:-1], {**messages[-1], "content": messages[-1]["content"] + "\n\n" + hint}]
        res = self.complete(msgs, model, max_tokens=max_tokens, response_format={"type": "json_object"},
                            reasoning_effort=reasoning_effort)
        return schema.model_validate_json(res.text), res

    # ------------------------------------------------------------------ internals
    def _cost(self, model: str, prompt: int, completion: int) -> float | None:
        price = self._prices.get(model)
        if price is None:
            return None
        return (prompt * price[0] + completion * price[1]) / 1_000_000

    def _call_with_retries(self, params: dict[str, Any]) -> LLMResult:
        est = estimate_tokens(json.dumps(params["messages"])) + min(params["max_completion_tokens"], 512)
        delay = 2.0
        for attempt in range(self.s.max_retries + 1):
            handle = self._limiter(params["model"]).acquire(est)
            t0 = time.perf_counter()
            try:
                raw = self._client.chat.completions.with_raw_response.create(**params)
                resp = raw.parse()
                self.last_rate_headers = {k: v for k, v in raw.headers.items() if k.startswith("x-ratelimit")}
            except Exception as exc:  # noqa: BLE001
                status = getattr(exc, "status_code", None)
                retryable = status == 429 or (status is not None and status >= 500) or \
                    type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
                if not retryable:
                    raise
                wait = _retry_after_seconds(exc) if status == 429 else None
                if wait is not None and wait > _QUOTA_RETRY_AFTER_S:
                    raise QuotaExhausted(f"Groq daily quota exhausted for {params['model']}; "
                                         f"retry after ~{wait / 60:.0f} min") from exc
                if attempt == self.s.max_retries:
                    raise RateLimited(f"Gave up after {attempt + 1} attempts: {exc}") from exc
                sleep_s = max(wait or 0.0, delay) + random.uniform(0, 0.5)
                log.warning("Groq %s (attempt %d); sleeping %.1fs", status or type(exc).__name__,
                            attempt + 1, sleep_s)
                time.sleep(sleep_s)
                delay = min(delay * 2, 60.0)
                continue

            latency = time.perf_counter() - t0
            u = resp.usage
            prompt_t = getattr(u, "prompt_tokens", 0) or 0
            completion_t = getattr(u, "completion_tokens", 0) or 0
            self._limiter(params["model"]).record_actual(handle, prompt_t + completion_t)
            text = (resp.choices[0].message.content or "").strip()
            usage = Usage(model=params["model"], prompt_tokens=prompt_t, completion_tokens=completion_t,
                          cost_usd=self._cost(params["model"], prompt_t, completion_t), latency_s=latency)
            return LLMResult(text=text, usage=usage)
        raise RateLimited("unreachable")  # pragma: no cover


_llm: LLM | None = None


def get_llm() -> LLM:
    """Return a process-wide LLM instance built from `get_settings()`."""
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm
