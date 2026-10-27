"""Write the web demo's static data into public/data/.

    python scripts/export_demo_data.py            # stats.json from the latest eval results
    python scripts/export_demo_data.py --record   # also examples.json: real agent runs (uses Groq quota)

stats.json is copied from eval/results (never hand-entered). examples.json holds real traces of the
example questions; the page plays them back only when the live API is unavailable, and says so.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]
OUT = ROOT / "public" / "data"

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / "config" / "demo.env", override=False)
os.environ["USE_LLM_CACHE"] = "false"  # record fresh runs, as a visitor would see them

from make_report import latest_complete  # noqa: E402


def stats() -> dict:
    """Headline metrics for plain RAG (baseline) vs the full agent, straight from the result files."""
    out: dict = {"configs": {}}
    for name in ("baseline", "full"):
        res = latest_complete(name)
        if res is None:
            continue
        m = res["metrics"]
        out["configs"][name] = {
            "file": res["_file"],
            "items": m["n_items"],
            "answerable": m["n_answerable"],
            "unanswerable": m["n_unanswerable"],
            "end_to_end_accuracy": m["answer"]["end_to_end_accuracy"],
            "correct_refusal_rate": m["refusal"]["correct_refusal_rate"],
            "false_refusal_rate": m["refusal"]["false_refusal_rate"],
            "faithfulness": m["answer"]["faithfulness"],
            "recall_at_k": m["retrieval"]["recall@4"],
            "grader_recall": (m["grader"]["set_level"] or {}).get("recall"),
            "grader_precision": (m["grader"]["set_level"] or {}).get("precision"),
            "latency_p50_s": m["latency_s"]["p50"],
            "cost_per_query_usd": m["cost_per_query_usd"]["mean"],
        }
        out["golden_source"] = res.get("golden_source")
    return out


def record() -> list[dict]:
    """Run the agent on every example question and return the payloads."""
    from rag_agent.config import get_settings
    from rag_agent.demo import EXAMPLES, trace_payload
    from rag_agent.graph import Agent
    from rag_agent.retrieval import Retriever

    s = get_settings()
    agent = Agent(Retriever.from_index(s), s)
    out = []
    for n, ex in enumerate(EXAMPLES):
        if n:
            time.sleep(25)  # stay clear of the per-minute token limit, so latencies are typical
        payload = trace_payload(agent.run(ex["question"]), agent.retriever)
        payload["recorded_at"] = time.strftime("%Y-%m-%d", time.gmtime())
        out.append(payload)
        print(f"{'REFUSED' if payload['refused'] else 'answered'}  {ex['question']}")
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="store_true", help="also record example traces (calls Groq)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(stats(), indent=2) + "\n")
    from rag_agent.demo import EXAMPLES

    examples: dict = {"examples": EXAMPLES, "recorded": []}
    path = OUT / "examples.json"
    if path.exists():
        examples["recorded"] = json.loads(path.read_text()).get("recorded", [])
    if args.record:
        examples["recorded"] = record()
    path.write_text(json.dumps(examples, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
