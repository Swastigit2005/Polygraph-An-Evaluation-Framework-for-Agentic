"""Run every ablation to completion across Groq free-tier daily limits, then build the report.

Usage:
    nohup .venv/bin/python eval/run_all.py > eval/results/run_all.log 2>&1 &

Phase A runs the agent for every config without the judge (uses only the generator/grader models);
phase B adds judge scores (Ragas + G-Eval) to the saved traces, so agent runs are never repeated.
Splitting the phases keeps the judge model's daily token limit from stalling the agent runs.
When run_eval.py stops because a daily quota is exhausted (exit code 2), this sleeps and resumes.
After everything completes, thresholds are set and ABLATIONS.md / the README table are regenerated.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ORDER = ["full", "baseline", "hybrid", "hybrid_rerank"]
SLEEP_ON_QUOTA_S = 30 * 60
MAX_ATTEMPTS = 300  # ~6 days of 30-minute retries


def log(msg: str) -> None:
    """Timestamped log line."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def run(args: list[str]) -> int:
    """Run a command from the repo root, streaming output."""
    return subprocess.call([PY, *args], cwd=ROOT)


def run_until_done(cfg: str, extra: list[str]) -> None:
    """Run one config/phase, sleeping through quota pauses."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"{cfg} {' '.join(extra) or '(with judge)'}: attempt {attempt}")
        code = run(["eval/run_eval.py", "--config", cfg, "--golden-source", "ai_reviewed", *extra])
        if code == 0:
            log(f"{cfg}: complete")
            return
        wait = SLEEP_ON_QUOTA_S if code == 2 else 300
        log(f"{cfg}: {'paused (quota)' if code == 2 else f'exit {code}'}; sleeping {wait // 60} min")
        time.sleep(wait)
    log(f"{cfg}: giving up")
    sys.exit(1)


def main() -> None:
    """Drive all configs to completion: agent runs first, then judge scores."""
    for cfg in ORDER:
        run_until_done(cfg, ["--no-judge"])
        run(["eval/make_report.py"])
    for cfg in ORDER:
        run_until_done(cfg, [])
        run(["eval/make_report.py"])
    # Final replay: every LLM call (agent + judge) is a cache hit, so this costs no quota and
    # yields latency free of the rate-limit queueing that the multi-day run accumulated.
    for cfg in ORDER:
        run_until_done(cfg, ["--fresh"])
    run(["eval/set_thresholds.py"])
    run(["eval/make_report.py"])
    log("all configs complete")


if __name__ == "__main__":
    main()
