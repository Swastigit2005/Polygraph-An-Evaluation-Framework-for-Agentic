"""Dense, BM25 and hybrid (Reciprocal Rank Fusion) retrieval, with optional cross-encoder rerank."""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from rag_agent.config import Settings, get_settings
from rag_agent.ingest import COLLECTION, Chunk, read_chunks

_TOKEN = re.compile(r"[a-z0-9][a-z0-9._-]*")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it of on or that the this to "
    "what when where which who why with you your my me we".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer for BM25 that keeps tokens like 'docker-compose' and 'daemon.json'."""
    return [t.strip("._-") for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and t.strip("._-")]


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank_d), rank starting at 1.

    Returns (id, score) sorted by descending score; ties broken by first appearance.
    """
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            order.setdefault(doc_id, len(order))
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


@dataclass
class Retrieved:
    """A chunk returned by retrieval together with the scores that put it there."""

    chunk: Chunk
    score: float
    scores: dict[str, float] = field(default_factory=dict)  # per-stage scores, e.g. dense/bm25/rrf/rerank


@dataclass
class RetrievalResult:
    """Final top-k plus per-stage candidate rankings (ids), for tracing and retrieval metrics."""

    hits: list[Retrieved]
    stages: dict[str, list[str]]

    @property
    def ids(self) -> list[str]:
        """Ids of the final hits, in rank order."""
        return [h.chunk.id for h in self.hits]


@lru_cache(maxsize=2)
def get_embedder(model_name: str) -> Any:
    """Load (once) a local sentence-transformers embedding model on CPU."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


@lru_cache(maxsize=2)
def get_reranker(model_name: str) -> Any:
    """Load (once) a local cross-encoder reranker on CPU."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")


class Retriever:
    """Retrieval over a fixed set of chunks; dense vectors come from Chroma or are computed in memory."""

    def __init__(self, chunks: list[Chunk], s: Settings, collection: Any | None = None) -> None:
        from rank_bm25 import BM25Okapi

        self.s = s
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self._bm25 = BM25Okapi([tokenize(c.embed_text()) for c in chunks])
        self._collection = collection
        self._matrix = None  # in-memory vectors when no Chroma collection (e.g. uploaded PDF)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_index(cls, s: Settings | None = None) -> Retriever:
        """Open the prebuilt index in `s.index_dir`."""
        import chromadb

        s = s or get_settings()
        chunks_path = s.index_dir / "chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(f"No index at {s.index_dir}; run `python -m rag_agent.ingest` first")
        # Chroma writes to its sqlite file even on read; open a temp copy so the committed index stays clean.
        tmp = Path(tempfile.mkdtemp(prefix="rag_index_")) / "chroma"
        shutil.copytree(s.index_dir / "chroma", tmp)
        client = chromadb.PersistentClient(path=str(tmp))
        return cls(read_chunks(chunks_path), s, client.get_collection(COLLECTION, embedding_function=None))

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], s: Settings | None = None) -> Retriever:
        """Build an in-memory retriever (used for uploaded PDFs)."""
        r = cls(chunks, s or get_settings())
        r._matrix = get_embedder(r.s.embed_model).encode(
            [c.embed_text() for c in chunks], normalize_embeddings=True, batch_size=64)
        return r

    # ------------------------------------------------------------------ retrievers
    def dense(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (id, cosine similarity) by dense embedding."""
        q = get_embedder(self.s.embed_model).encode(
            [self.s.embed_query_prompt + query], normalize_embeddings=True)[0]
        k = min(k, len(self.chunks))
        if self._collection is not None:
            res = self._collection.query(query_embeddings=[q.tolist()], n_results=k, include=["distances"])
            return [(i, 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0], strict=True)]
        sims = self._matrix @ q
        top = sims.argsort()[::-1][:k]
        return [(self.chunks[i].id, float(sims[i])) for i in top]

    def bm25(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (id, BM25 score); chunks with zero score are dropped."""
        scores = self._bm25.get_scores(tokenize(query))
        top = scores.argsort()[::-1][:k]
        return [(self.chunks[i].id, float(scores[i])) for i in top if scores[i] > 0]

    def rerank(self, query: str, ids: list[str]) -> list[tuple[str, float]]:
        """Re-order candidate ids with the cross-encoder (higher = more relevant)."""
        if not ids:
            return []
        model = get_reranker(self.s.rerank_model)
        scores = model.predict([(query, self.by_id[i].embed_text()) for i in ids])
        return sorted(zip(ids, (float(x) for x in scores), strict=True), key=lambda kv: -kv[1])

    def retrieve(self, query: str, s: Settings | None = None) -> RetrievalResult:
        """Run the configured pipeline (mode, reranker, top_k) and return hits plus stage rankings."""
        s = s or self.s
        stages: dict[str, list[str]] = {}
        per_stage: dict[str, dict[str, float]] = {}

        if s.retrieval_mode in ("dense", "hybrid"):
            d = self.dense(query, s.candidate_k)
            stages["dense"] = [i for i, _ in d]
            per_stage["dense"] = dict(d)
        if s.retrieval_mode in ("bm25", "hybrid"):
            b = self.bm25(query, s.candidate_k)
            stages["bm25"] = [i for i, _ in b]
            per_stage["bm25"] = dict(b)

        if s.retrieval_mode == "hybrid":
            fused = rrf_fuse([stages["dense"], stages["bm25"]], k=s.rrf_k)[: s.candidate_k]
            stages["rrf"] = [i for i, _ in fused]
            per_stage["rrf"] = dict(fused)
            ranked = fused
        else:
            ranked = d if s.retrieval_mode == "dense" else b

        if s.use_reranker:
            rr = self.rerank(query, [i for i, _ in ranked])
            stages["rerank"] = [i for i, _ in rr]
            per_stage["rerank"] = dict(rr)
            ranked = rr

        hits = [
            Retrieved(chunk=self.by_id[i], score=sc,
                      scores={name: vals[i] for name, vals in per_stage.items() if i in vals})
            for i, sc in ranked[: s.top_k]
        ]
        return RetrievalResult(hits=hits, stages=stages)
