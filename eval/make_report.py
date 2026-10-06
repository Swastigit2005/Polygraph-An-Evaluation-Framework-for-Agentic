"""Build eval/results/ABLATIONS.md from the latest complete, verified result of each config.

Usage:
    python eval/make_report.py

Includes the cross-config metrics table, the judge-calibration result (if any) and up to five
failure examples from the `full` config with compact traces and a rule-based diagnosis.
Nothing here is hand-entered: every number comes from eval/results/*.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
RESULTS = EVAL_DIR / "results"
sys.path.insert(0, str(EVAL_DIR))

import yaml  # noqa: E402
from report import metrics_table  # noqa: E402


def latest_complete(config: str) -> dict[str, Any] | None:
    """Newest result for `config` that is complete and not a dry run."""
    for path in sorted(RESULTS.glob(f"{config}_*.json"), reverse=True):
        if "__dryrun" in path.name:
            continue
        res = json.loads(path.read_text())
        if res.get("complete") and not res.get("dry_run"):
            res["_file"] = path.name
            return res
    return None


def compact_trace(trace: dict[str, Any]) -> str:
    """One line per step."""
    lines = []
    for st in trace["steps"]:
        d = st["data"]
        if st["node"] == "retrieve":
            lines.append(f"retrieve  q={d['query']!r} -> {', '.join(d['retrieved_ids'])}")
        elif st["node"] == "grade_context":
            lines.append(f"grade     {'PASS' if d['relevant'] else 'FAIL'}: {d['reason']}")
        elif st["node"] == "rewrite_query":
            lines.append(f"rewrite   {d['from']!r} -> {d['to']!r}")
        elif st["node"] == "generate":
            lines.append(f"generate  cited={d['cited_ids']} refused={d['refused']}")
        elif st["node"] == "refuse":
            lines.append(f"refuse    {d['reason']}")
    return "\n".join(lines)


DX = {
    "fr_grader": "A gold chunk was retrieved but the grader rejected it every time (grader false negative).",
    "fr_retrieval": "No retrieval (original or rewritten) surfaced a gold chunk; refusing was the safe outcome "
                    "of a retrieval miss.",
    "wa_retrieval": "Answered from non-gold context that the grader accepted; retrieval missed the gold chunk.",
    "wa_generation": "Gold context was available but the answer missed or contradicted key facts (generation error).",
    "grader_fp": "Grader passed a context with no gold chunk (possibly incomplete gold labels).",
    "loop": "Hit MAX_REWRITES on an answerable question; rewrites did not converge.",
}


def diagnose(rec: dict[str, Any]) -> tuple[str, str] | None:
    """(category, one-line diagnosis) if the record is a failure, else None."""
    item, t, j = rec["item"], rec["trace"], rec.get("judge") or {}
    gold = set(item.get("relevant_chunk_ids") or [])
    retrieved = [set(st["data"]["retrieved_ids"]) for st in t["steps"] if st["node"] == "retrieve"]
    grades = [st["data"] for st in t["steps"] if st["node"] == "grade_context"]
    ever_gold = any(gold & r for r in retrieved)
    if not item["should_refuse"] and t["refused"]:
        return "false refusal", DX["fr_grader"] if ever_gold else DX["fr_retrieval"]
    if item["should_refuse"] and not t["refused"]:
        why = "grader accepted topically similar context" if any(g["relevant"] for g in grades) else \
            "no grader in this config; the generator answered anyway"
        return "answered unanswerable", f"Should have refused: {why}."
    if not item["should_refuse"] and not t["refused"] and j.get("correctness") is not None and j["correctness"] < 0.5:
        return "wrong answer", DX["wa_generation"] if ever_gold else DX["wa_retrieval"]
    if any(g["relevant"] and not (gold & set(g["graded_ids"])) for g in grades) and not item["should_refuse"]:
        return "grader false positive", DX["grader_fp"]
    if t["rewrites"] >= rec["trace"]["config"]["max_rewrites"] > 0 and not item["should_refuse"]:
        return "loop exhausted", DX["loop"]
    return None


def failure_examples(res: dict[str, Any], k: int = 5) -> list[tuple[str, str, dict[str, Any]]]:
    """Up to k failures, preferring one per category."""
    found: list[tuple[str, str, dict[str, Any]]] = []
    for rec in res["records"]:
        dx = diagnose(rec)
        if dx:
            found.append((dx[0], dx[1], rec))
    picked, seen = [], set()
    for f in found:  # first pass: one per category
        if f[0] not in seen:
            picked.append(f)
            seen.add(f[0])
    for f in found:  # fill up
        if len(picked) >= k:
            break
        if f not in picked:
            picked.append(f)
    return picked[:k]


def main() -> None:
    """Write ABLATIONS.md."""
    configs = yaml.safe_load((EVAL_DIR / "ablations.yaml").read_text())["configs"]
    results = {n: r for n in configs if (r := latest_complete(n))}
    out = ["# Ablation results", ""]
    if not results:
        out += ["No complete verified eval runs yet: all metrics **TBD**.", ""]
    else:
        missing = [n for n in configs if n not in results]
        out += ["Generated by `eval/make_report.py` from:", ""]
        out += [f"- `{n}`: `{r['_file']}` ({r['n_evaluated']} verified items, commit `{r['git_commit']}`)"
                for n, r in results.items()]
        if missing:
            out += ["", f"Not yet run: {', '.join(f'`{m}`' for m in missing)} (TBD)."]
        any_r = next(iter(results.values()))
        out += ["", f"Generator `{any_r['config']['gen_model']}`, grader/rewriter `{any_r['config']['fast_model']}`, "
                f"judge `{any_r['judge_model']}`. Ragas metrics are computed on a fixed subset of answerable items "
                "for `baseline` and `full` only (free-tier budget); other cells show —.", "",
                metrics_table(results), ""]
    cal = RESULTS / "judge_calibration.md"
    out += ["## Judge calibration", "", cal.read_text() if cal.exists() else "TBD (no human labels yet).", ""]
    if "full" in results:
        out += ["## Failure examples (`full` config)", ""]
        fails = failure_examples(results["full"])
        if not fails:
            out += ["No failures found by the rule-based detector."]
        for i, (cat, dx, rec) in enumerate(fails, 1):
            j = rec.get("judge") or {}
            answer = rec["trace"]["answer"]
            out += [f"### {i}. {cat}: `{rec['item']['id']}`", "",
                    f"**Q:** {rec['item']['question']}  ",
                    f"**Gold chunks:** {rec['item']['relevant_chunk_ids'] or '—'}  ",
                    f"**Answer:** {answer[:400]}{'…' if len(answer) > 400 else ''}"]
            if j.get("correctness") is not None:
                out[-1] += f"  \n**Judge correctness:** {j['correctness']:.2f}"
            out += ["", "```text", compact_trace(rec["trace"]), "```", "", f"**Diagnosis:** {dx}", ""]
    (RESULTS / "ABLATIONS.md").write_text("\n".join(out) + "\n")
    print(f"Wrote {RESULTS / 'ABLATIONS.md'} ({len(results)} configs)")


if __name__ == "__main__":
    main()
