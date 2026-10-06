"""Set eval/thresholds.json from the latest complete `full` run, restricted to the CI subset.

Usage:
    python eval/set_thresholds.py [--tolerance 0.10]

For each gated metric the threshold is the value the full run achieved on the CI-subset items,
minus (for "min" metrics) or plus (for "max" metrics) an absolute tolerance, clipped to [0, 1].
The tolerance absorbs run-to-run noise on a 20-item subset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "src"))

from check_thresholds import lookup  # noqa: E402
from make_report import latest_complete  # noqa: E402

from rag_agent.eval_metrics import aggregate, aggregate_judges  # noqa: E402


def main() -> None:
    """Compute and write thresholds."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tolerance", type=float, default=0.10)
    args = ap.parse_args()
    res = latest_complete("full")
    if res is None:
        sys.exit("No complete `full` run yet")
    ci_ids = set(json.loads((EVAL_DIR / "ci_subset.json").read_text())["ids"])
    records = [r for r in res["records"] if r["item"]["id"] in ci_ids]
    if len(records) != len(ci_ids):
        sys.exit(f"full run covers {len(records)}/{len(ci_ids)} CI items; re-run eval/make_ci_subset.py?")
    cfg = res["config"]
    metrics = aggregate(records, cfg["max_rewrites"], cfg["top_k"])
    metrics["answer"] = aggregate_judges(records)
    path = EVAL_DIR / "thresholds.json"
    th = json.loads(path.read_text())
    for kind, sign in (("min", -1), ("max", 1)):
        for name in th[kind]:
            v = lookup(metrics, name)
            th[kind][name] = None if v is None else round(min(1.0, max(0.0, v + sign * args.tolerance)), 3)
    th["source_result"] = res["_file"]
    th["tolerance"] = args.tolerance
    th["observed_on_ci_subset"] = {n: lookup(metrics, n) for k in ("min", "max") for n in th[k]}
    path.write_text(json.dumps(th, indent=2) + "\n")
    print(json.dumps(th, indent=2))


if __name__ == "__main__":
    main()
