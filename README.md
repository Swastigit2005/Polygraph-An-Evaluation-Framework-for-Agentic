---
title: Agentic RAG with Evals
emoji: 🔄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Self-reflective RAG agent over Docker docs, with agent evals
---

# Self-Reflective Agentic RAG with an Agent Evaluation Harness

**A LangGraph RAG agent that grades its own retrieval, rewrites failed queries and refuses rather than guessing, plus an evaluation harness that measures each of those decisions, not just the final answer.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-yellow)](HF_SPACE_URL)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/Swastigit2005/agentic-rag-evals)
![CI](https://github.com/Swastigit2005/agentic-rag-evals/actions/workflows/ci.yml/badge.svg)

> Demo GIF: _placeholder._ To record one, open the Space, ask an answerable question and one of the "not in the
> knowledge base" examples, expand **Agent trace**, then record with Kap (macOS, free) or ScreenToGif (Windows)
> and save it as `assets/demo.gif`.

The knowledge base is 242 pages of the Docker documentation (Engine, Compose, Build, Desktop), giving an
IT-support / DevOps assistant. Everything runs on free tiers: Groq for LLMs, local sentence-transformers for
embeddings and reranking, and Hugging Face Spaces for hosting.

## Architecture

```mermaid
flowchart LR
    Q([question]) --> R[retrieve<br/>BM25 + dense → RRF<br/>→ cross-encoder rerank]
    R --> G{grade_context<br/>structured verdict:<br/>relevant? reason, chunk ids}
    G -- relevant --> A[generate<br/>answer only from context,<br/>inline chunk citations]
    G -- "not relevant &<br/>rewrites < MAX" --> W[rewrite_query]
    W --> R
    G -- "not relevant &<br/>rewrites = MAX" --> X[refuse<br/>'I couldn't find this<br/>in the knowledge base']
    A --> T[(trace: steps, verdicts,<br/>rewrites, ids, latency,<br/>tokens, cost)]
    X --> T
```

Every node writes a structured decision record. One run returns a `Trace`, and that trace is what the
evaluation scores and what the demo's **Agent trace** panel shows.

| Component | Choice |
|---|---|
| Orchestration | LangGraph `StateGraph` (`src/rag_agent/graph.py`) |
| Generator | `openai/gpt-oss-120b` on Groq |
| Grader and query rewriter | `openai/gpt-oss-20b` on Groq (small and fast) |
| Eval judge | `qwen/qwen3.8-27b` on Groq: a different model family from the generator, to reduce self-preference bias |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, CPU) in ChromaDB |
| Lexical retrieval | `rank_bm25`, fused with dense results via Reciprocal Rank Fusion (k = 60) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, CPU) |
| Answer metrics | Ragas (faithfulness, answer relevancy) and DeepEval G-Eval (correctness), both routed through Groq |
| Tracing | Langfuse (optional; a no-op when its keys are absent) |

All behaviour is set by environment variables (`src/rag_agent/config.py`): `RETRIEVAL_MODE=dense|bm25|hybrid`,
`USE_RERANKER`, `USE_REFLECTION`, `MAX_REWRITES` and `TOP_K`. The ablations are just different settings of these.

## Evaluation results

<!-- RESULTS:START -->
_No complete evaluation run yet: all metrics are **TBD**. This section is regenerated from `eval/results/*.json` by `eval/make_report.py`._
<!-- RESULTS:END -->

### What the metrics mean

**Agent decisions** (the core of the project):
- **Grader precision / recall (set level).** Each `grade_context` step is one prediction. Positive means the
  retrieved context contains at least one gold chunk. The chunk-level variant scores every graded chunk
  against `relevant_chunk_ids`.
- **Correct-refusal rate:** the share of unanswerable questions the agent refused.
- **False-refusal rate:** the share of answerable questions it refused.
- **Rewrite success rate:** the share of rewrites after which the grader passed and a gold chunk was retrieved.
- **Steps per query** (mean, p95) and **loop rate**, the share of queries that used all `MAX_REWRITES`.

**Retrieval:** Recall@k and MRR@k, computed on the first retrieval (the original question).

**Answers:**
- Ragas **faithfulness** (are the answer's claims supported by the retrieved context?).
- Ragas **answer relevancy**.
- DeepEval **G-Eval correctness** against the reference answer.
- **End-to-end accuracy:** correct answers divided by all answerable questions, so refusals count as wrong.

**Ops:** latency p50/p95 per query and per node, tokens per query, and cost per query at Groq list prices.
The free tier isn't billed. When a call is served from the disk cache, it's charged its originally measured
latency.

### Golden-set methodology

- **75–90 items drafted by an LLM** (`eval/draft_golden.py`) from seed chunks sampled across all four doc
  sections:
  - answerable questions, each with a grounded reference answer;
  - deliberately **vague** questions that need a rewrite;
  - **unanswerable** but plausible in-domain questions (Kubernetes administration, Swarm, Docker Hub billing,
    other CI systems).
- **Gold `relevant_chunk_ids`:** labelled by a different model (`qwen3.8-27b`) over a TREC-style pool, the top-5
  dense plus top-5 BM25 hits for each question. The grader being evaluated never labels its own gold.
- **Unanswerable checks:** each unanswerable item was checked against its retrieval pool, and dropped if any chunk
  answered it.
- **Review: done by an AI (Claude), not by a human.** Every item was read together with its gold and pooled
  chunks. Proposals are in `eval/ai_review_proposals.json`:
  - 70 kept: 40 answerable, 10 vague, 20 unanswerable;
  - 2 edited;
  - 17 rejected, mostly "unanswerable" questions the corpus partly answers, plus trivia and duplicates.
- **Which items count:** the metrics above use those 70 items (`--golden-source ai_reviewed`). A human can
  confirm or override every proposal with `eval/review_golden.py --proposals ...`; the runner then reports on
  `--golden-source verified` instead.
- **Judge calibration** (`eval/judge_calibration.py`) needs 30 human correct/incorrect labels to compute accuracy
  and Cohen's κ against the G-Eval judge. **It hasn't been done**, so the correctness numbers come from an
  uncalibrated LLM judge.

## Key design decisions and trade-offs

- **Refuse instead of falling back.** The original project generated "with the best context available" after
  three failed grades. Here, an explicit `refuse` node guarantees the generator never answers from bad context.
  The cost is false refusals when the grader is too strict, which the false-refusal rate measures directly.
- **Different models for grading, generating and judging.** A small, fast grader keeps the loop cheap. A
  different-family judge reduces self-preference bias.
- **Hybrid retrieval with RRF.** Docker questions are full of exact tokens (`daemon.json`, `--memory`, error
  strings) that BM25 matches well, and paraphrased symptoms that dense retrieval matches well. RRF combines the
  two without tuning score scales.
- **Structured grader output.** A Pydantic model through Groq structured outputs
  (`relevant`, `reason`, `relevant_chunk_ids`) replaces free-text parsing. The chunk ids make chunk-level grader
  metrics possible at no extra cost.
- **Designed for the free tier.**
  - Rate limiting per model (requests and tokens per minute).
  - Exponential backoff that honours `retry-after`.
  - Exhausted daily quotas are detected, and runs stop cleanly instead of retrying all day.
  - An SQLite response cache.
  - Resumable eval runs.
  - A per-day usage ledger (`scripts/usage_today.py`).
  - Ragas runs on a fixed 30-item subset, and only for `baseline` and `full`, to fit the judge's 200K-tokens/day
    budget.
- **Prebuilt index committed** (about 9 MB), so the demo starts instantly. Visitors don't have to upload anything.

## Failure analysis

`eval/results/ABLATIONS.md` lists up to five failure cases from the `full` config. Each has its trace and a
diagnosis: grader false negative, retrieval miss, generation error, grader false positive or rewrite loop. That
file is generated from the same results as the table above.

## Run locally

```bash
git clone https://github.com/Swastigit2005/agentic-rag-evals && cd agentic-rag-evals
uv venv -p 3.11 .venv && uv pip install -p .venv -r requirements.txt -r requirements-dev.txt -e .
cp .env.example .env                       # add GROQ_API_KEY (free at console.groq.com)
.venv/bin/python scripts/list_groq_models.py   # see which models your key can use; set them in .env
.venv/bin/python app.py                    # http://127.0.0.1:7860
.venv/bin/pytest -q                        # offline unit tests (no network)
```

Use `.venv/bin/python` explicitly if your shell aliases `python` to a system interpreter.

## Run the evaluations

```bash
.venv/bin/python eval/run_eval.py --config full --golden-source ai_reviewed --limit 5   # quick check
nohup .venv/bin/python eval/run_all.py > eval/results/run_all.log 2>&1 &                # all ablations
.venv/bin/python scripts/usage_today.py      # Groq free-tier usage today
.venv/bin/python eval/make_report.py         # rebuild ABLATIONS.md and this README's results section
```

A full ablation sweep (4 configs × 70 items, plus judges) needs more than one day of Groq free-tier quota.
`run_all.py` waits out the daily quota and resumes where it stopped.

CI (`.github/workflows/ci.yml`) runs ruff and the unit tests on every push. The **eval gate** runs on manual
dispatch, or on PRs labelled `run-evals`: it evaluates the `full` config on a fixed 20-item subset
(`eval/ci_subset.json`) and fails if any metric falls below `eval/thresholds.json`. The thresholds come from the
first full run, minus 0.10. The gate needs a `GROQ_API_KEY` repository secret.

## Deploy (Hugging Face Spaces, free CPU)

```bash
hf auth login                                   # token with "write" access from huggingface.co/settings/tokens
.venv/bin/python scripts/deploy_space.py        # creates the Space, sets secrets, uploads, verifies a live answer
```

Docker alternative: `docker build -t agentic-rag-evals . && docker run -p 7860:7860 --env-file .env agentic-rag-evals`.

## Limitations

- **The golden set was reviewed by an AI, not a human.** It's also small: 70 items, from one documentation
  corpus.
- **The correctness judge is uncalibrated.** No human labels have been collected.
- **Gold chunk labels come from pooling, so they can be incomplete.** Some "grader false positives" may actually
  be relevant chunks missing from the gold list.
- **Single run per config.** LLM outputs vary, so small differences between configs aren't significant.
- **The free-tier quota is shared with the live demo.** Under load, visitors see a "demo busy" message.
- **The corpus is a snapshot** (docker/docs@`6cf1b1c`, Oct 2026). Answers can go stale as Docker changes.

## Acknowledgements

- Base code: [Self-Reflective Agentic RAG](https://github.com/Sumanth077/Hands-On-AI-Engineering/tree/main/ai_agents/agentic_rag_system)
  from [Hands-On-AI-Engineering](https://github.com/Sumanth077/Hands-On-AI-Engineering) by Sumanth077 (MIT). See [NOTICE](NOTICE).
- Corpus: [Docker documentation](https://github.com/docker/docs) (Apache-2.0). See [data/corpus/SOURCES.md](data/corpus/SOURCES.md).
- Models: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5),
  [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2),
  and gpt-oss and Qwen models served by [Groq](https://groq.com).

## License

This project's code is MIT-licensed ([LICENSE](LICENSE)). Corpus files keep their Apache-2.0 license.
