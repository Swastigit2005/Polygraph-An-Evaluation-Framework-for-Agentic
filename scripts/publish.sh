#!/usr/bin/env bash
# Regenerate results, commit them, push to GitHub and redeploy the Space.
# Usage: scripts/publish.sh        (after `hf auth login` and adding the GitHub remote once)
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python eval/make_report.py
git add README.md eval/results/*.json eval/results/*.md eval/thresholds.json eval/ci_subset.json 2>/dev/null || true
git commit -m "Update evaluation results" || echo "(no result changes to commit)"
git push origin main
.venv/bin/python scripts/deploy_space.py
