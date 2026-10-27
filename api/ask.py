"""Vercel serverless function: POST {"question": "..."} -> agent trace JSON. GET warms the instance.

Runs the same LangGraph agent and hybrid + rerank retrieval as the evaluated `full` config, with the
local models served by ONNX Runtime (PyTorch does not fit a serverless bundle).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / "config" / "demo.env", override=False)  # non-secret settings; GROQ_API_KEY is a host secret
os.environ.setdefault("CACHE_DIR", "/tmp/polygraph-cache")  # the only writable path on Vercel

from rag_agent.config import get_settings  # noqa: E402
from rag_agent.demo import BUSY_MSG, MAX_QUESTION_CHARS, QUOTA_MSG, trace_payload  # noqa: E402
from rag_agent.graph import Agent  # noqa: E402
from rag_agent.llm import LLM, QuotaExhausted, RateLimited  # noqa: E402
from rag_agent.retrieval import Retriever  # noqa: E402

log = logging.getLogger("api.ask")
PER_IP_LIMIT = int(os.getenv("DEMO_PER_IP_LIMIT", "12"))  # questions per IP per 10 minutes, per instance
_WINDOW_S = 600.0

_agent: Agent | None = None
_lock = threading.Lock()
_hits: dict[str, deque[float]] = defaultdict(deque)


def get_agent() -> Agent:
    """Load the index and models once per warm instance."""
    global _agent
    with _lock:
        if _agent is None:
            s = get_settings().with_overrides(max_retries=2)  # fail fast instead of hanging the request
            retriever = Retriever.from_index(s)
            retriever.retrieve("warm-up query", s)  # load the embedder and reranker now
            _agent = Agent(retriever, s, LLM(s))
        return _agent


def _allowed(ip: str) -> bool:
    now = time.monotonic()
    q = _hits[ip]
    while q and now - q[0] > _WINDOW_S:
        q.popleft()
    if len(q) >= PER_IP_LIMIT:
        return False
    q.append(now)
    return True


def answer(question: str, ip: str = "") -> tuple[int, dict]:
    """(HTTP status, JSON body) for one question."""
    question = " ".join((question or "").split())
    if not question:
        return 400, {"error": "Please type a question."}
    if len(question) > MAX_QUESTION_CHARS:
        return 400, {"error": f"Please keep questions under {MAX_QUESTION_CHARS} characters."}
    if not _allowed(ip):
        return 429, {"error": "You've asked a lot of questions in a short time. Please wait a few minutes."}
    try:
        agent = get_agent()
        return 200, trace_payload(agent.run(question), agent.retriever)
    except QuotaExhausted:
        return 503, {"error": QUOTA_MSG, "busy": True}
    except RateLimited:
        return 503, {"error": BUSY_MSG, "busy": True}
    except Exception:  # noqa: BLE001 - never show a stack trace to visitors
        log.exception("agent run failed")
        return 500, {"error": "Something went wrong on our side. Please try again.", "busy": True}


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel looks for this exact name
    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        try:
            get_agent()
            self._send(200, {"ok": True})
        except Exception:  # noqa: BLE001
            log.exception("warm-up failed")
            self._send(500, {"ok": False})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = min(int(self.headers.get("Content-Length") or 0), 10_000)
            question = json.loads(self.rfile.read(length) or b"{}").get("question", "")
        except (ValueError, AttributeError):
            self._send(400, {"error": "Invalid request."})
            return
        ip = (self.headers.get("x-forwarded-for") or self.client_address[0]).split(",")[0].strip()
        self._send(*answer(str(question), ip))
