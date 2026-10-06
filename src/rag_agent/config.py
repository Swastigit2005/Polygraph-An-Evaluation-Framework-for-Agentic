"""All runtime settings, read from environment variables (optionally via a .env file).

Nothing in the codebase hard-codes a model name, k, threshold or feature flag;
every such knob lives here so that eval ablations can be expressed purely as
environment overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

RetrievalMode = Literal["dense", "bm25", "hybrid"]


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw in (None, "") else int(raw)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw in (None, "") else float(raw)


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    """Immutable bag of settings. Use `get_settings()` or `Settings.from_env()`."""

    # --- LLM provider (Groq) ---
    groq_api_key: str = field(repr=False, default="")
    gen_model: str = ""  # strong model for answer generation
    fast_model: str = ""  # small model for grading / rewriting
    judge_model: str = ""  # eval judge; ideally differs from gen_model
    temperature: float = 0.0
    # Forwarded as `reasoning_effort` to reasoning models; empty = provider default.
    gen_reasoning_effort: str = "low"
    fast_reasoning_effort: str = "low"
    max_output_tokens: int = 1024
    # Client-side rate limits, applied per model (Groq limits are per model).
    rpm_limit: int = 30
    tpm_limit: int = 6000
    max_retries: int = 6
    # USD per 1M tokens as "model=in/out,model=in/out"; unknown models report cost=None.
    model_prices: str = ""

    # --- Embeddings / reranker (local) ---
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_query_prompt: str = "Represent this sentence for searching relevant passages: "
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Retrieval / agent behaviour ---
    retrieval_mode: RetrievalMode = "hybrid"
    use_reranker: bool = True
    use_reflection: bool = True
    max_rewrites: int = 2
    top_k: int = 4
    candidate_k: int = 20  # pool size per retriever before fusion / rerank
    rrf_k: int = 60

    # --- Chunking ---
    chunk_size: int = 1200  # characters
    chunk_overlap: int = 150

    # --- Paths ---
    corpus_dir: Path = ROOT / "data" / "corpus" / "docker-docs"
    index_dir: Path = ROOT / "data" / "index"
    cache_dir: Path = ROOT / ".cache"
    use_llm_cache: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the current process environment."""
        mode = _str("RETRIEVAL_MODE", "hybrid").lower()
        if mode not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"RETRIEVAL_MODE must be dense|bm25|hybrid, got {mode!r}")
        return cls(
            groq_api_key=_str("GROQ_API_KEY"),
            gen_model=_str("GEN_MODEL"),
            fast_model=_str("FAST_MODEL"),
            judge_model=_str("JUDGE_MODEL"),
            temperature=_float("LLM_TEMPERATURE", 0.0),
            gen_reasoning_effort=_str("GEN_REASONING_EFFORT", "low"),
            fast_reasoning_effort=_str("FAST_REASONING_EFFORT", "low"),
            max_output_tokens=_int("LLM_MAX_OUTPUT_TOKENS", 1024),
            rpm_limit=_int("GROQ_RPM_LIMIT", 30),
            tpm_limit=_int("GROQ_TPM_LIMIT", 6000),
            max_retries=_int("LLM_MAX_RETRIES", 6),
            model_prices=_str("MODEL_PRICES"),
            embed_model=_str("EMBED_MODEL", cls.embed_model),
            embed_query_prompt=os.getenv("EMBED_QUERY_PROMPT", cls.embed_query_prompt),
            rerank_model=_str("RERANK_MODEL", cls.rerank_model),
            retrieval_mode=mode,  # type: ignore[arg-type]
            use_reranker=_bool("USE_RERANKER", True),
            use_reflection=_bool("USE_REFLECTION", True),
            max_rewrites=_int("MAX_REWRITES", 2),
            top_k=_int("TOP_K", 4),
            candidate_k=_int("CANDIDATE_K", 20),
            rrf_k=_int("RRF_K", 60),
            chunk_size=_int("CHUNK_SIZE", 1200),
            chunk_overlap=_int("CHUNK_OVERLAP", 150),
            index_dir=Path(_str("INDEX_DIR") or cls.index_dir),
            cache_dir=Path(_str("CACHE_DIR") or cls.cache_dir),
            use_llm_cache=_bool("USE_LLM_CACHE", True),
        )

    def with_overrides(self, **kwargs: object) -> Settings:
        """Return a copy with some fields replaced (used by ablations and tests)."""
        return replace(self, **kwargs)  # type: ignore[arg-type]

    def require_llm(self) -> None:
        """Fail loudly if the Groq key or model names are missing."""
        missing = [n for n, v in (("GROQ_API_KEY", self.groq_api_key), ("GEN_MODEL", self.gen_model),
                                  ("FAST_MODEL", self.fast_model)) if not v]
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)} (see .env.example)")


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide settings, loading them from the environment once."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
