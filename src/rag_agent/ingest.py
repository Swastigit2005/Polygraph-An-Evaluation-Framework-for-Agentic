"""Load -> clean -> chunk -> embed -> persist (Chroma vectors + chunk store for BM25).

Usage:
    python -m rag_agent.ingest            # build data/index/ from data/corpus/docker-docs
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_agent.config import Settings, get_settings

COLLECTION = "docker_docs"
DOCS_BASE_URL = "https://docs.docker.com/"

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_PARAM = re.compile(r"\{\{%\s*param\s+\"?([\w-]+)\"?\s*%\}\}")
_INCLUDE = re.compile(r"\{\{%\s*include\s+\"([^\"]+)\"\s*%\}\}")
_TAB_OPEN = re.compile(r"\{\{<\s*tab\s+name=\"([^\"]+)\"\s*>\}\}")
_SHORTCODE = re.compile(r"\{\{[<%].*?[%>]\}\}", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_HEADING = re.compile(r"^(#{1,4})\s+(.*?)\s*(\{#[^}]*\})?\s*$")
_ATTR_LINE = re.compile(r"^\s*\{[:.#][^}]*\}\s*$", re.MULTILINE)  # Hugo/kramdown attribute lines


@dataclass
class Chunk:
    """A retrievable unit of text with provenance."""

    id: str  # stable id, e.g. "engine/daemon/troubleshoot#004"
    doc_id: str  # page id, e.g. "engine/daemon/troubleshoot"
    title: str  # page title
    section: str  # heading path inside the page, e.g. "Daemon > Check whether Docker is running"
    url: str
    text: str

    def header(self) -> str:
        """Human-readable location used when embedding and when showing context to the LLM."""
        return f"{self.title} > {self.section}" if self.section else self.title

    def embed_text(self) -> str:
        """Text used for dense embedding and BM25 (header + body)."""
        return f"{self.header()}\n{self.text}"


# --------------------------------------------------------------------------- cleaning
def split_front_matter(raw: str) -> tuple[dict, str]:
    """Return (front-matter dict, body) for a Markdown page."""
    m = _FRONT_MATTER.match(raw)
    if not m:
        return {}, raw
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), raw[m.end():]


def clean_markdown(body: str, params: dict[str, str], includes_dir: Path | None = None) -> str:
    """Expand/strip Hugo shortcodes and Markdown links, leaving readable Markdown."""

    def _include(m: re.Match[str]) -> str:
        if includes_dir is None:
            return ""
        path = includes_dir / m.group(1)
        if not path.is_file():
            return ""
        return split_front_matter(path.read_text(encoding="utf-8"))[1]

    body = _INCLUDE.sub(_include, body)
    body = _PARAM.sub(lambda m: params.get(m.group(1), m.group(1)), body)
    body = _TAB_OPEN.sub(lambda m: f"\n**{m.group(1)}:**\n", body)
    body = _SHORTCODE.sub("", body)
    body = _HTML_COMMENT.sub("", body)
    body = _ATTR_LINE.sub("", body)
    body = _LINK.sub(lambda m: m.group(1), body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def doc_id_for(rel_path: Path) -> str:
    """Map 'engine/daemon/_index.md' -> 'engine/daemon', 'engine/x.md' -> 'engine/x'."""
    parts = list(rel_path.with_suffix("").parts)
    if parts and parts[-1] == "_index":
        parts = parts[:-1]
    return "/".join(parts) or "index"


# --------------------------------------------------------------------------- chunking
def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split Markdown into (heading path, body) sections on #..#### headings, ignoring code fences."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append((" > ".join(h for _, h in stack), text))
        buf.clear()

    for line in markdown.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def chunk_page(doc_id: str, title: str, url: str, markdown: str, chunk_size: int,
               chunk_overlap: int) -> list[Chunk]:
    """Chunk one cleaned page into heading-aware, size-bounded chunks with stable ids."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for section, text in split_sections(markdown):
        for piece in splitter.split_text(text):
            if len(piece.strip()) < 40:  # drop fragments with no retrievable content
                continue
            chunks.append(Chunk(id=f"{doc_id}#{len(chunks):03d}", doc_id=doc_id, title=title,
                                section=section, url=url, text=piece.strip()))
    return chunks


def load_corpus(s: Settings) -> list[Chunk]:
    """Load and chunk every page under `s.corpus_dir/manuals`."""
    manuals = s.corpus_dir / "manuals"
    params_path = s.corpus_dir / "hugo_params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else {}
    includes = s.corpus_dir / "includes"
    chunks: list[Chunk] = []
    for path in sorted(manuals.rglob("*.md")):
        rel = path.relative_to(manuals)
        meta, body = split_front_matter(path.read_text(encoding="utf-8"))
        doc_id = doc_id_for(rel)
        title = str(meta.get("title") or meta.get("linkTitle") or doc_id)
        url = DOCS_BASE_URL + doc_id + "/" if doc_id != "index" else DOCS_BASE_URL
        cleaned = clean_markdown(body, params, includes)
        chunks.extend(chunk_page(doc_id, title, url, cleaned, s.chunk_size, s.chunk_overlap))
    return chunks


def load_pdf(path: str | Path, s: Settings) -> list[Chunk]:
    """Chunk an uploaded PDF (one section per page) for the optional upload tab."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    name = Path(path).stem
    splitter = RecursiveCharacterTextSplitter(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
    chunks: list[Chunk] = []
    for page_no, page in enumerate(reader.pages, start=1):
        for piece in splitter.split_text(page.extract_text() or ""):
            if len(piece.strip()) >= 40:
                chunks.append(Chunk(id=f"pdf:{name}#{len(chunks):03d}", doc_id=f"pdf:{name}", title=name,
                                    section=f"page {page_no}", url="", text=piece.strip()))
    return chunks


# --------------------------------------------------------------------------- persistence
def write_chunks(chunks: Iterable[Chunk], path: Path) -> None:
    """Write chunks as JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    """Read chunks written by `write_chunks`."""
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def build_index(s: Settings) -> dict:
    """Build the full on-disk index (chunks.jsonl, Chroma collection, meta.json)."""
    import chromadb

    from rag_agent.retrieval import get_embedder

    t0 = time.perf_counter()
    chunks = load_corpus(s)
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate chunk ids; check doc_id mapping")

    s.index_dir.mkdir(parents=True, exist_ok=True)
    write_chunks(chunks, s.index_dir / "chunks.jsonl")

    embedder = get_embedder(s.embed_model)
    vectors = embedder.encode([c.embed_text() for c in chunks], batch_size=64, normalize_embeddings=True,
                              show_progress_bar=True)

    chroma_dir = s.index_dir / "chroma"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None)
    batch = 1000
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        col.add(ids=[c.id for c in part], embeddings=vectors[i:i + batch].tolist(),
                metadatas=[{"doc_id": c.doc_id, "title": c.title} for c in part])

    manifest_path = s.corpus_dir.parent / "manifest.json"
    meta = {
        "num_chunks": len(chunks),
        "num_docs": len({c.doc_id for c in chunks}),
        "embed_model": s.embed_model,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "corpus": json.loads(manifest_path.read_text()) if manifest_path.exists() else None,
        "build_seconds": round(time.perf_counter() - t0, 1),
    }
    (s.index_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    """CLI entry point: build the index from the configured corpus."""
    argparse.ArgumentParser(description="Build the retrieval index").parse_args()
    meta = build_index(get_settings())
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
