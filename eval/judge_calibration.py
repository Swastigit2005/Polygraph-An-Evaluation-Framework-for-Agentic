"""Calibrate the LLM correctness judge (G-Eval) against human labels.

Usage:
    python eval/judge_calibration.py label  [--n 30]   # you label sampled answers correct/incorrect
    python eval/judge_calibration.py report             # agreement: accuracy + Cohen's kappa

Sampling: answered, answerable items from all non-dry-run results in eval/results/, deduplicated
by (question, answer). To make kappa meaningful, up to half the sample is drawn from answers the
judge scored below the threshold (the rest at/above it), with a fixed seed. The judge's score is
never shown while labelling. Labels go to eval/results/calibration_labels.jsonl; the report goes
to eval/results/judge_calibration.json (+ .md), which the app and README read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_agent.eval_metrics import cohen_kappa  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
LABELS = RESULTS / "calibration_labels.jsonl"
REPORT = RESULTS / "judge_calibration.json"
THRESHOLD = 0.5
SEED = 7
KAPPA_TARGET = 0.6


def _key(question: str, answer: str) -> str:
    return hashlib.sha256(f"{question}\n{answer}".encode()).hexdigest()[:16]


def candidates() -> list[dict[str, Any]]:
    """All judged answers from real (non-dry-run) results, newest file wins per (question, answer)."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(RESULTS.glob("*_*.json")):
        if "__dryrun" in path.name or path.name.startswith("judge_calibration"):
            continue
        res = json.loads(path.read_text())
        if res.get("dry_run") or "records" not in res:
            continue
        for r in res["records"]:
            j = r.get("judge") or {}
            if r["item"]["should_refuse"] or r["trace"]["refused"] or j.get("correctness") is None:
                continue
            k = _key(r["item"]["question"], r["trace"]["answer"])
            out[k] = {"key": k, "item_id": r["item"]["id"], "config": res["config_name"],
                      "question": r["item"]["question"], "reference": r["item"]["reference_answer"],
                      "answer": r["trace"]["answer"], "judge_score": j["correctness"],
                      "judge_reason": j.get("correctness_reason", ""), "judge_model": j.get("judge_model")}
    return list(out.values())


def sample(cands: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Stratified sample: up to n/2 judge-incorrect, rest judge-correct; deterministic."""
    rng = random.Random(SEED)
    low = sorted((c for c in cands if c["judge_score"] < THRESHOLD), key=lambda c: c["key"])
    high = sorted((c for c in cands if c["judge_score"] >= THRESHOLD), key=lambda c: c["key"])
    rng.shuffle(low)
    rng.shuffle(high)
    take_low = min(len(low), n // 2)
    picked = low[:take_low] + high[: n - take_low]
    rng.shuffle(picked)
    return picked


def load_labels() -> dict[str, dict[str, Any]]:
    """Existing human labels keyed by (question, answer) hash."""
    if not LABELS.exists():
        return {}
    return {d["key"]: d for d in map(json.loads, LABELS.read_text().splitlines()) if d}


def label(n: int) -> None:
    """Interactive labelling loop (resumable)."""
    cands = candidates()
    if not cands:
        sys.exit("No judged answers found. Run eval/run_eval.py (verified items, with judge) first.")
    picked = sample(cands, n)
    labels = load_labels()
    todo = [c for c in picked if c["key"] not in labels]
    print(f"{len(picked)} sampled answers, {len(picked) - len(todo)} already labelled.\n"
          "For each: is the ANSWER correct with respect to the REFERENCE (paraphrase is fine; contradictions or "
          "missing key facts are not)?  c = correct, i = incorrect, s = skip, q = quit\n")
    for k, c in enumerate(todo, 1):
        print("=" * 100)
        print(f"[{k}/{len(todo)}] {c['item_id']}  (config: {c['config']})")
        for title, text in (("QUESTION", c["question"]), ("REFERENCE", c["reference"]), ("ANSWER", c["answer"])):
            print(f"\033[1m{title}\033[0m")
            print(textwrap.indent(textwrap.fill(text, 100, replace_whitespace=False), "  "))
        while True:
            ans = input("correct? [c/i/s/q] > ").strip().lower()
            if ans in {"c", "i", "s", "q"}:
                break
        if ans == "q":
            break
        if ans == "s":
            continue
        rec = {k2: c[k2] for k2 in ("key", "item_id", "config", "judge_score", "judge_model")}
        rec.update(human_correct=ans == "c", labelled_at=datetime.now(UTC).isoformat(timespec="seconds"))
        with LABELS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"\nLabels saved to {LABELS}. Run: python eval/judge_calibration.py report")


def report() -> None:
    """Agreement between judge (score >= threshold) and human labels."""
    labels = list(load_labels().values())
    if not labels:
        sys.exit("No labels yet. Run: python eval/judge_calibration.py label")
    judge = [d["judge_score"] >= THRESHOLD for d in labels]
    human = [d["human_correct"] for d in labels]
    n = len(labels)
    acc = sum(a == b for a, b in zip(judge, human, strict=True)) / n
    kappa = cohen_kappa(judge, human)
    cm = {"judge_correct_human_correct": sum(a and b for a, b in zip(judge, human, strict=True)),
          "judge_correct_human_incorrect": sum(a and not b for a, b in zip(judge, human, strict=True)),
          "judge_incorrect_human_correct": sum(not a and b for a, b in zip(judge, human, strict=True)),
          "judge_incorrect_human_incorrect": sum(not a and not b for a, b in zip(judge, human, strict=True))}
    out = {"n_labels": n, "threshold": THRESHOLD, "judge_models": sorted({d.get("judge_model") or "?" for d in labels}),
           "accuracy": acc, "cohen_kappa": kappa, "confusion": cm, "kappa_target": KAPPA_TARGET,
           "trusted": kappa is not None and kappa >= KAPPA_TARGET,
           "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
           "sampling": "stratified: up to half judge-incorrect, rest judge-correct (seed 7)"}
    REPORT.write_text(json.dumps(out, indent=2) + "\n")
    kappa_s = "undefined" if kappa is None else f"{kappa:.3f}"
    md = (f"| Judge | Labels | Accuracy | Cohen's κ |\n|---|---:|---:|---:|\n"
          f"| {', '.join(out['judge_models'])} (G-Eval correctness ≥ {THRESHOLD}) | {n} | {100 * acc:.1f}% | "
          f"{kappa_s} |\n")
    REPORT.with_suffix(".md").write_text(md)
    print(md)
    print(json.dumps(cm, indent=2))
    if not out["trusted"]:
        print(f"\n⚠️  kappa {kappa_s} is below {KAPPA_TARGET}: do not trust the correctness judge yet; "
              "inspect disagreements and refine CORRECTNESS_STEPS in src/rag_agent/judges.py.")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["label", "report"])
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    label(args.n) if args.command == "label" else report()


if __name__ == "__main__":
    main()
