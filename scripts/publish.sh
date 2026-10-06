#!/usr/bin/env bash
# Regenerate results, commit them and push to GitHub (Streamlit Cloud redeploys on push).
# Usage: scripts/publish.sh
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python eval/make_report.py
git add README.md eval/results eval/thresholds.json eval/ci_subset.json
git commit -m "Update evaluation results" || echo "(no result changes to commit)"
git push origin main
