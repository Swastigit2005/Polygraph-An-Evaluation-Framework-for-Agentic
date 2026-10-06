"""Draft golden-set items with an LLM. Every drafted item has "verified": false.

Usage:
    python eval/draft_golden.py [--answerable 45] [--vague 12] [--unanswerable 30] [--limit N]

Method (documented in the README):
  * Answerable / vague: sample seed chunks stratified across documents; the generator model
    writes a question + reference answer grounded in the seed chunk. Gold relevant_chunk_ids
    are labelled by a *different* model over a pool of candidates (top dense + top BM25 hits
    for the question, plus the seed), TREC-style pooling.
  * Unanswerable: the model proposes plausible in-domain questions outside the corpus scope;
    each is checked against a retrieval pool and dropped if any pooled chunk answers it.
Drafts must be reviewed with eval/review_golden.py; only verified items count in metrics.
The run is resumable: existing ids in the output file are skipped.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_agent.config import get_settings  # noqa: E402
from rag_agent.ingest import Chunk  # noqa: E402
from rag_agent.llm import LLM, QuotaExhausted  # noqa: E402
from rag_agent.retrieval import Retriever  # noqa: E402

OUT = Path(__file__).resolve().parent / "golden_set.jsonl"
SEED = 13
POOL_PER_RETRIEVER = 5
SNIPPET_CHARS = 700


class DraftQA(BaseModel):
    """Generator output for one seed chunk."""

    skip: bool = Field(description="True if the excerpt is navigation/boilerplate and cannot support a question")
    question: str = ""
    reference_answer: str = ""


class PoolLabels(BaseModel):
    """Labeller output: which pooled chunks support the reference answer."""

    relevant_ids: list[str]


class AnswerCheck(BaseModel):
    """Labeller output for an unanswerable candidate."""

    answerable: bool
    supporting_ids: list[str] = Field(default_factory=list)


class Unanswerables(BaseModel):
    """Generator output: candidate out-of-scope questions."""

    questions: list[str]


SPECIFIC_PROMPT = """Below is an excerpt from the Docker documentation.
Write ONE realistic question that a developer or IT-support engineer would ask, whose answer is fully
contained in the excerpt. Phrase it the way a user would (do not copy headings verbatim).
Then write a reference answer of 1-3 sentences that uses ONLY facts stated in the excerpt.
Set skip=true if the excerpt is navigation, a link list, a changelog fragment or too thin to support
a useful question.

Excerpt [{cid}] ({header}):
{text}"""

VAGUE_PROMPT = """Below is an excerpt from the Docker documentation.
Write ONE deliberately vague, underspecified question, as a frustrated user might type it: describe the
symptom or goal in plain words, avoid Docker command names, flags and exact terminology, and leave
out context. The answer must still be contained in the excerpt. Then write a reference answer of
1-3 sentences that uses ONLY facts stated in the excerpt.
Set skip=true if the excerpt is navigation, a link list or too thin to support a useful question.

Excerpt [{cid}] ({header}):
{text}"""

LABEL_PROMPT = """Question: {question}
Reference answer: {answer}

Below are candidate documentation chunks. Return the ids of EVERY chunk that contains information
needed to produce the reference answer (fully or partially). Do not include chunks that are merely
on the same topic.

{pool}"""

UNANSWERABLE_PROMPT = """You are building a test set for a support assistant whose knowledge base covers ONLY these
parts of the Docker documentation: Docker Engine (install, daemon, containers, storage, networking,
logging, security; NOT Swarm), Docker Compose, Docker Build/BuildKit/Bake, and Docker Desktop
setup/troubleshooting/settings.

Write {n} realistic, specific questions that a DevOps/IT-support user might plausibly ask this assistant,
but that this knowledge base does NOT answer. Mix: Kubernetes administration, Docker Swarm services,
Docker Hub billing/plans/organizations, Docker Scout, Podman/containerd/CRI-O specifics, cloud services
(ECS, AKS, Cloud Run), CI providers' own settings, OS-level troubleshooting unrelated to Docker, and
exact version-history questions. Each must sound in-domain, not absurd. One question per list item."""

CHECK_PROMPT = """Question: {question}

Do ANY of the candidate documentation chunks below contain the information needed to answer this
question? Be strict: a chunk that only mentions the topic does not count.

{pool}"""


def _pool_text(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.id}] {c.header()}\n{c.text[:SNIPPET_CHARS]}" for c in chunks)


def _pool(retriever: Retriever, question: str, extra: list[str] | None = None) -> list[Chunk]:
    ids = list(extra or [])
    for i, _ in retriever.dense(question, POOL_PER_RETRIEVER) + retriever.bm25(question, POOL_PER_RETRIEVER):
        if i not in ids:
            ids.append(i)
    return [retriever.by_id[i] for i in ids]


def sample_seeds(chunks: list[Chunk], n: int, rng: random.Random, exclude_docs: set[str]) -> list[Chunk]:
    """Pick at most one substantive chunk per document, spread across top-level sections."""
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if len(c.text) >= 400 and c.text.count("](") < 3 and c.doc_id not in exclude_docs:
            by_doc[c.doc_id].append(c)
    by_section: dict[str, list[str]] = defaultdict(list)
    for doc in sorted(by_doc):
        by_section[doc.split("/")[0]].append(doc)
    for docs in by_section.values():
        rng.shuffle(docs)
    seeds: list[Chunk] = []
    sections = sorted(by_section)
    while len(seeds) < n and any(by_section.values()):
        for sec in sections:  # round-robin across sections
            if by_section[sec] and len(seeds) < n:
                seeds.append(rng.choice(by_doc[by_section[sec].pop()]))
    return seeds


def _write(items: list[dict]) -> None:
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    tmp.replace(OUT)


def main() -> None:
    """Draft golden items and append them to eval/golden_set.jsonl."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answerable", type=int, default=45)
    ap.add_argument("--vague", type=int, default=12)
    ap.add_argument("--unanswerable", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None, help="max new items to draft this run")
    args = ap.parse_args()

    s = get_settings()
    llm = LLM(s)
    gen, labeller = s.gen_model, s.judge_model or s.gen_model
    retriever = Retriever.from_index(s)
    rng = random.Random(SEED)

    items = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()] if OUT.exists() else []
    done = {it["id"] for it in items}
    budget = args.limit if args.limit is not None else 10**9

    seeds = sample_seeds(retriever.chunks, args.answerable + args.vague + 20, rng, exclude_docs=set())
    plan = [("answerable", c) for c in seeds[: args.answerable + 10]] + \
           [("vague", c) for c in seeds[args.answerable + 10:]]
    counts = {"answerable": 0, "vague": 0}
    for it in items:
        if it["type"] in counts:
            counts[it["type"]] += 1
    target = {"answerable": args.answerable, "vague": args.vague}

    try:
        for kind, seed in plan:
            item_id = f"{kind[0]}-{seed.id}"
            if item_id in done or counts[kind] >= target[kind] or budget <= 0:
                continue
            prompt = (SPECIFIC_PROMPT if kind == "answerable" else VAGUE_PROMPT).format(
                cid=seed.id, header=seed.header(), text=seed.text)
            qa, _ = llm.complete_structured([{"role": "user", "content": prompt}], gen, DraftQA,
                                            max_tokens=800, reasoning_effort=s.gen_reasoning_effort)
            if qa.skip or not qa.question.strip():
                continue
            pool = _pool(retriever, qa.question, extra=[seed.id])
            labels, _ = llm.complete_structured(
                [{"role": "user", "content": LABEL_PROMPT.format(question=qa.question, answer=qa.reference_answer,
                                                                 pool=_pool_text(pool))}],
                labeller, PoolLabels, max_tokens=600)
            pool_ids = {c.id for c in pool}
            rel = [seed.id] + [i for i in labels.relevant_ids if i in pool_ids and i != seed.id]
            items.append({"id": item_id, "type": kind, "question": qa.question.strip(),
                          "reference_answer": qa.reference_answer.strip(), "relevant_chunk_ids": rel,
                          "should_refuse": False, "source_chunk_id": seed.id, "pool_ids": [c.id for c in pool],
                          "verified": False, "notes": ""})
            counts[kind] += 1
            budget -= 1
            _write(items)
            print(f"[{kind:10}] {item_id}: {qa.question[:90]}  rel={len(rel)}")

        n_unans = sum(1 for it in items if it["type"] == "unanswerable")
        if n_unans < args.unanswerable and budget > 0:
            cands, _ = llm.complete_structured(
                [{"role": "user", "content": UNANSWERABLE_PROMPT.format(n=args.unanswerable + 10)}],
                gen, Unanswerables, max_tokens=2500, reasoning_effort=s.gen_reasoning_effort)
            for k, q in enumerate(cands.questions):
                item_id = f"u-{k:03d}"
                if item_id in done or n_unans >= args.unanswerable or budget <= 0:
                    continue
                pool = _pool(retriever, q)
                check, _ = llm.complete_structured(
                    [{"role": "user", "content": CHECK_PROMPT.format(question=q, pool=_pool_text(pool))}],
                    labeller, AnswerCheck, max_tokens=400)
                if check.answerable:
                    print(f"[dropped   ] {q[:90]} (answerable via {check.supporting_ids})")
                    continue
                items.append({"id": item_id, "type": "unanswerable", "question": q.strip(), "reference_answer": "",
                              "relevant_chunk_ids": [], "should_refuse": True, "source_chunk_id": None,
                              "pool_ids": [c.id for c in pool], "verified": False, "notes": ""})
                n_unans += 1
                budget -= 1
                _write(items)
                print(f"[unanswer. ] {item_id}: {q[:90]}")
    except QuotaExhausted as exc:
        print(f"\nStopped: {exc}\nRe-run later; completed items are kept and will be skipped.")

    by_type = defaultdict(int)
    for it in items:
        by_type[it["type"]] += 1
    print(f"\n{len(items)} items in {OUT.name}: {dict(by_type)} (all verified=false until reviewed)")


if __name__ == "__main__":
    main()
