"""Fail (exit 1) if the latest result for a config drops below eval/thresholds.json.

Usage:
    python eval/check_thresholds.py --config full [--result eval/results/full_<ts>.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent


def lookup(metrics: dict[str, Any], dotted: str) -> Any:
    """Follow a dotted path like 'grader.set_level.precision' (keys may contain '@')."""
    cur: Any = metrics
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def check(metrics: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    """Return human-readable failures (empty list = pass)."""
    failures = []
    for kind, op in (("min", lambda v, t: v >= t), ("max", lambda v, t: v <= t)):
        for name, limit in thresholds.get(kind, {}).items():
            if limit is None:
                failures.append(f"{name}: threshold not set (run a full eval and set eval/thresholds.json)")
                continue
            value = lookup(metrics, name)
            if value is None:
                failures.append(f"{name}: metric missing from result")
            elif not op(value, limit):
                failures.append(f"{name}: {value:.3f} {'<' if kind == 'min' else '>'} {kind} {limit:.3f}")
    return failures


def latest_result(config: str) -> Path:
    """Most recent non-dry-run result file for `config`."""
    files = sorted(p for p in (EVAL_DIR / "results").glob(f"{config}_*.json") if "__dryrun" not in p.name)
    if not files:
        sys.exit(f"No results for config {config!r}")
    return files[-1]


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="full")
    ap.add_argument("--result", type=Path, default=None)
    args = ap.parse_args()
    path = args.result or latest_result(args.config)
    result = json.loads(path.read_text())
    if result.get("dry_run"):
        sys.exit(f"{path.name} is a dry run on unverified items; refusing to gate on it")
    thresholds = json.loads((EVAL_DIR / "thresholds.json").read_text())
    failures = check(result["metrics"], thresholds)
    print(f"Checked {path.name} ({result['n_evaluated']} items) against thresholds.json")
    for f in failures:
        print(f"  FAIL {f}")
    if failures:
        sys.exit(1)
    print("  PASS: all metrics within thresholds")


if __name__ == "__main__":
    main()
