"""Write eval/ci_subset.json: a deterministic, stratified 20-item subset for the CI eval gate.

Usage:
    python eval/make_ci_subset.py [--golden-source ai_reviewed]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_eval import load_golden

EVAL_DIR = Path(__file__).resolve().parent
QUOTA = {"answerable": 10, "vague": 3, "unanswerable": 7}


def main() -> None:
    """Pick items by hash order within each type."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden-source", choices=["verified", "ai_reviewed"], default="ai_reviewed")
    args = ap.parse_args()
    items = load_golden(False, args.golden_source)
    ids: list[str] = []
    for kind, n in QUOTA.items():
        pool = sorted((it["id"] for it in items if it["type"] == kind),
                      key=lambda i: hashlib.sha256(i.encode()).hexdigest())
        ids += pool[:n]
    out = {"golden_source": args.golden_source, "quota": QUOTA, "ids": ids}
    (EVAL_DIR / "ci_subset.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"Wrote {len(ids)} ids to eval/ci_subset.json")


if __name__ == "__main__":
    main()
