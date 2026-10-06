"""Run the agent over the golden set for one ablation config and compute all metrics.

Usage:
    python eval/run_eval.py --config full [--limit N] [--no-judge] [--include-unverified]
    python eval/run_eval.py --config all                      # every config in ablations.yaml

Outputs eval/results/<config>_<timestamp>.json and .md. Runs are resumable: per-item records are
kept in eval/results/progress/<config>.jsonl and reused unless the item (or config) changed.
If Groq's daily quota runs out the run stops cleanly; re-run the same command later to resume.
Only items with "verified": true are used unless --include-unverified is given, in which case the
results are labelled DRY RUN and must not be reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from report import metrics_table  # noqa: E402

from rag_agent.config import Settings, get_settings  # noqa: E402
from rag_agent.eval_metrics import aggregate, aggregate_judges  # noqa: E402
from rag_agent.graph import Agent  # noqa: E402
from rag_agent.llm import LLM, QuotaExhausted  # noqa: E402
from rag_agent.retrieval import Retriever  # noqa: E402

EVAL_DIR = ROOT / "eval"
RESULTS = EVAL_DIR / "results"
PROGRESS = RESULTS / "progress"


def load_golden(include_unverified: bool, source: str = "verified",
                path: Path = EVAL_DIR / "golden_set.jsonl") -> list[dict[str, Any]]:
    """Golden items for evaluation.

    source="verified": items a human marked verified (dry runs: all non-rejected items).
    source="ai_reviewed": items accepted in eval/ai_review_proposals.json (with its edits applied).
    Results always record which source was used.
    """
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if source == "ai_reviewed":
        props = json.loads((EVAL_DIR / "ai_review_proposals.json").read_text())["proposals"]
        out = []
        for it in items:
            prop = props.get(it["id"])
            if prop and prop["decision"] != "reject":
                out.append({**it, **prop.get("edit", {})})
        return out
    return [it for it in items if it.get("verified") or (include_unverified and not it.get("rejected"))]


def load_configs(path: Path = EVAL_DIR / "ablations.yaml") -> dict[str, dict[str, Any]]:
    """Ablation configs keyed by name."""
    return yaml.safe_load(path.read_text())["configs"]


def apply_config(base: Settings, cfg: dict[str, Any]) -> Settings:
    """Settings with the config's overrides applied (unknown keys raise)."""
    overrides = {k: v for k, v in cfg.items() if k not in ("description", "ragas")}
    return base.with_overrides(**overrides)


def item_key(item: dict[str, Any], cfg_snapshot: dict[str, Any]) -> str:
    """Hash of everything that should invalidate a cached per-item record."""
    payload = {k: item.get(k) for k in ("question", "reference_answer", "relevant_chunk_ids", "should_refuse")}
    return hashlib.sha256(json.dumps([payload, cfg_snapshot], sort_keys=True).encode()).hexdigest()[:16]


def ragas_subset(items: list[dict[str, Any]], n: int) -> set[str]:
    """Deterministic subset of answerable item ids (by hash of id), identical across configs."""
    ans = [it["id"] for it in items if not it["should_refuse"]]
    return set(sorted(ans, key=lambda i: hashlib.sha256(i.encode()).hexdigest())[:n])


def git_commit() -> str | None:
    """Current git commit (short), if available."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_config(name: str, cfg: dict[str, Any], args: argparse.Namespace, base: Settings,
               retriever: Retriever, llm: LLM) -> int:
    """Evaluate one config; returns 0 when complete, 2 when stopped by quota."""
    s = apply_config(base, cfg)
    agent = Agent(retriever, s, llm)
    snapshot = agent.config_snapshot()
    judge = None
    if not args.no_judge:
        from rag_agent.judges import Judge  # heavy import; only when judging

        judge = Judge(s, llm, relevancy_strictness=args.relevancy_strictness)

    items = load_golden(args.include_unverified, args.golden_source)
    wanted: set[str] = set(args.ids.split(",")) if args.ids else set()
    if args.ids_file:
        wanted |= set(json.loads(Path(args.ids_file).read_text())["ids"])
    if wanted:
        items = [it for it in items if it["id"] in wanted]
    if args.limit is not None:
        items = items[: args.limit]
    use_ragas = judge is not None and bool(cfg.get("ragas")) and args.ragas_subset > 0
    ragas_ids = ragas_subset(items, args.ragas_subset) if use_ragas else set()
    if not items:
        print("No golden items to evaluate (are any verified? use --include-unverified for a dry run)")
        return 1

    suffix = "__dryrun" if args.include_unverified else ""
    PROGRESS.mkdir(parents=True, exist_ok=True)
    prog_path = PROGRESS / f"{name}{suffix}.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if prog_path.exists() and not args.fresh:
        for line in prog_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["key"]] = rec

    records: list[dict[str, Any]] = []
    stopped = False
    t_start = time.perf_counter()
    retriever.retrieve("warm-up query", s)  # load embedder/reranker before timing anything
    print(f"\n=== {name}: {cfg.get('description', '')} — {len(items)} items")
    for n, item in enumerate(items, 1):
        key = item_key(item, snapshot)
        needs_judge = judge is not None and not item["should_refuse"]
        wants_ragas = item["id"] in ragas_ids
        rec = done.get(key)
        judged = rec is not None and rec.get("judge") is not None and \
            (not wants_ragas or "faithfulness" in rec["judge"])
        if rec and (not needs_judge or judged or rec["trace"]["refused"]):
            records.append(rec)
            continue
        try:
            if rec is None:
                trace = agent.run(item["question"]).to_dict()
                rec = {"key": key, "id": item["id"], "item": item, "trace": trace, "judge": None}
            if needs_judge and not rec["trace"]["refused"]:
                gen = [st for st in rec["trace"]["steps"] if st["node"] == "generate"][-1]
                contexts = [retriever.by_id[c].embed_text() for c in gen["data"]["context_ids"]]
                rec["judge"] = judge.score(item["question"], rec["trace"]["answer"], contexts,
                                           item["reference_answer"], ragas=wants_ragas)
        except QuotaExhausted as exc:
            print(f"\nStopped at item {n}/{len(items)}: {exc}\nRe-run the same command later to resume.")
            stopped = True
            break
        with prog_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        records.append(rec)
        t = rec["trace"]
        verdicts = "".join("✓" if st["data"]["relevant"] else "✗" for st in t["steps"] if st["node"] == "grade_context")
        j = rec.get("judge") or {}
        print(f"[{n:3}/{len(items)}] {item['id'][:48]:48} {'REFUSED' if t['refused'] else 'answered':8} "
              f"grades={verdicts or '-':4} steps={len(t['steps'])} corr={j.get('correctness', '-')}")

    if not records:
        return 2 if stopped else 1
    metrics = aggregate(records, s.max_rewrites, s.top_k)
    if judge is not None:
        metrics["answer"] = aggregate_judges(records)
        metrics["answer"]["ragas_subset_size"] = len(ragas_ids)
        metrics["judge_tokens_total"] = sum((r.get("judge") or {}).get("tokens", 0) for r in records)
    complete = len(records) == len(items)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = {
        "config_name": name, "description": cfg.get("description", ""), "config": snapshot,
        "judge_model": judge.model if judge else None, "timestamp": ts, "git_commit": git_commit(),
        "dry_run": bool(args.include_unverified), "complete": complete,
        "golden_source": "dry run (unverified)" if args.include_unverified else args.golden_source,
        "n_target_items": len(items), "n_evaluated": len(records),
        "wall_seconds": round(time.perf_counter() - t_start, 1),
        "metrics": metrics, "records": records,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"{name}{suffix}_{ts}"
    (RESULTS / f"{stem}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    banner = "> **DRY RUN on unverified items — do not report.**\n\n" if args.include_unverified else ""
    partial = "" if complete else f"> Partial run: {len(records)}/{len(items)} items.\n\n"
    (RESULTS / f"{stem}.md").write_text(f"# {name}\n\n{banner}{partial}{metrics_table({name: out})}\n")
    print(f"\nWrote {RESULTS / stem}.json/.md ({len(records)}/{len(items)} items{'' if complete else ', PARTIAL'})")
    print(metrics_table({name: out}))
    return 0 if complete else 2


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config name from ablations.yaml, or 'all'")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N items")
    ap.add_argument("--ids", default=None, help="comma-separated golden item ids")
    ap.add_argument("--ids-file", default=None, help='JSON file {"ids": [...]} (e.g. eval/ci_subset.json)')
    ap.add_argument("--no-judge", action="store_true", help="skip Ragas/DeepEval answer metrics")
    ap.add_argument("--golden-source", choices=["verified", "ai_reviewed"], default="verified",
                    help="verified = human-verified items; ai_reviewed = items accepted by the AI pre-review")
    ap.add_argument("--include-unverified", action="store_true", help="dry run on unverified items")
    ap.add_argument("--fresh", action="store_true", help="ignore saved per-item progress")
    ap.add_argument("--ragas-subset", type=int, default=30,
                    help="answerable items scored with Ragas in configs that enable it (0 = off)")
    ap.add_argument("--relevancy-strictness", type=int, default=1, help="Ragas answer-relevancy questions per answer")
    args = ap.parse_args()

    configs = load_configs()
    names = list(configs) if args.config == "all" else [args.config]
    for n in names:
        if n not in configs:
            sys.exit(f"Unknown config {n!r}; choose from {list(configs)} or 'all'")
    base = get_settings()
    retriever = Retriever.from_index(base)
    llm = LLM(base)
    code = 0
    for n in names:
        code = run_config(n, configs[n], args, base, retriever, llm)
        if code == 2:
            break
    sys.exit(code)


if __name__ == "__main__":
    main()
