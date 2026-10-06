"""Run every ablation to completion across Groq free-tier daily limits, then build the report.

Usage:
    nohup .venv/bin/python eval/run_all.py > eval/results/run_all.log 2>&1 &

Runs configs in order (full first). When run_eval.py stops because the daily quota is exhausted
(exit code 2), sleeps and resumes; completed items are never recomputed. After `full` completes,
thresholds are set; after all configs complete, ABLATIONS.md is regenerated.
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


def main() -> None:
    """Drive all configs to completion."""
    for cfg in ORDER:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            log(f"{cfg}: attempt {attempt}")
            code = run(["eval/run_eval.py", "--config", cfg, "--golden-source", "ai_reviewed"])
            if code == 0:
                log(f"{cfg}: complete")
                break
            if code == 2:
                log(f"{cfg}: paused (quota); sleeping {SLEEP_ON_QUOTA_S // 60} min")
                time.sleep(SLEEP_ON_QUOTA_S)
                continue
            log(f"{cfg}: failed with exit code {code}; retrying in 5 min")
            time.sleep(300)
        else:
            log(f"{cfg}: giving up")
            sys.exit(1)
        if cfg == "full":
            run(["eval/set_thresholds.py"])
        run(["eval/make_report.py"])
    log("all configs complete")


if __name__ == "__main__":
    main()
