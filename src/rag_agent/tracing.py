"""Optional Langfuse tracing. A no-op unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.

The agent opens one `agent` observation per question, one `span` per graph node and one
`generation` per LLM call (model, prompt, output, token usage, list-price cost), so a run in
Langfuse mirrors the local Trace object. Tracing failures never break the app.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)


class _NoOp:
    """Stand-in observation when tracing is disabled."""

    def update(self, **_: Any) -> None:
        """Ignore updates."""


def enabled() -> bool:
    """True when Langfuse credentials are configured."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@lru_cache(maxsize=1)
def _client() -> Any | None:
    if not enabled():
        return None
    try:
        from langfuse import get_client

        return get_client()  # reads LANGFUSE_* env vars (LANGFUSE_HOST / LANGFUSE_BASE_URL)
    except Exception as exc:  # noqa: BLE001 - tracing must never break the app
        log.warning("Langfuse disabled: %s", exc)
        return None


@contextmanager
def observation(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """Context manager yielding a Langfuse observation (or a no-op) with `.update(...)`."""
    client = _client()
    if client is None:
        yield _NoOp()
        return
    try:
        cm = client.start_as_current_observation(name=name, as_type=as_type, **kwargs)
        obs = cm.__enter__()
    except Exception as exc:  # noqa: BLE001
        log.warning("Langfuse observation failed: %s", exc)
        yield _NoOp()
        return
    try:
        yield obs
    except BaseException as exc:
        cm.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        cm.__exit__(None, None, None)


def flush() -> None:
    """Send buffered events (call at the end of batch jobs)."""
    client = _client()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:  # noqa: BLE001
            log.warning("Langfuse flush failed: %s", exc)
