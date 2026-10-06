"""Fetch the demo corpus: a pinned subset of the Docker documentation (Apache-2.0).

Usage:
    python scripts/fetch_corpus.py [--commit <sha>]

Copies selected Markdown pages from https://github.com/docker/docs into
data/corpus/docker-docs/, along with the repo LICENSE, the shared include
fragments, and the scalar version parameters from hugo.yaml that the pages
reference via {{% param %}} shortcodes. Writes data/corpus/manifest.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

REPO_URL = "https://github.com/docker/docs.git"
DEFAULT_COMMIT = "6cf1b1c167f032e8a6629da211602300b623b20e"

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"
OUT_DIR = CORPUS_DIR / "docker-docs"

# Sections of content/manuals/ to include; IT-support / DevOps focused.
INCLUDE_PREFIXES = (
    "engine/",
    "compose/",
    "build/",
    "desktop/setup/",
    "desktop/troubleshoot-and-support/",
    "desktop/settings-and-maintenance/",
    "desktop/use-desktop/",
)
# Excluded: noisy changelogs, legacy orchestration, CI recipes, enterprise policy.
EXCLUDE_PREFIXES = (
    "engine/release-notes/",
    "engine/swarm/",
    "build/ci/",
    "build/policies/",
    "compose/releases/",
)


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _flatten_scalars(params: dict, prefix: str = "") -> dict[str, str]:
    """Keep only top-level scalar params (version strings etc.), dropping analytics config."""
    return {k: str(v) for k, v in params.items() if isinstance(v, (str, int, float))}


def select_pages(manuals: Path) -> list[Path]:
    """Return the sorted list of Markdown pages that make up the corpus."""
    pages = []
    for p in sorted(manuals.rglob("*.md")):
        rel = p.relative_to(manuals).as_posix()
        if rel.startswith(INCLUDE_PREFIXES) and not rel.startswith(EXCLUDE_PREFIXES):
            pages.append(p)
    return pages


def main() -> None:
    """Clone docker/docs at a pinned commit and copy the corpus subset."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "docs"
        _run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", REPO_URL, str(repo)])
        _run(["git", "sparse-checkout", "set", "--no-cone", "/content/manuals/", "/content/includes/",
              "/hugo.yaml", "/LICENSE"], cwd=repo)
        _run(["git", "checkout", "-q", args.commit], cwd=repo)
        commit_date = _run(["git", "log", "-1", "--format=%cI"], cwd=repo)

        manuals = repo / "content" / "manuals"
        pages = select_pages(manuals)

        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        for p in pages:
            dest = OUT_DIR / "manuals" / p.relative_to(manuals)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        shutil.copytree(repo / "content" / "includes", OUT_DIR / "includes")
        shutil.copy2(repo / "LICENSE", OUT_DIR / "LICENSE")

        hugo = yaml.safe_load((repo / "hugo.yaml").read_text())
        params = _flatten_scalars(hugo.get("params", {}))
        (OUT_DIR / "hugo_params.json").write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")

    manifest = {
        "source": REPO_URL,
        "commit": args.commit,
        "commit_date": commit_date,
        "license": "Apache-2.0",
        "include_prefixes": list(INCLUDE_PREFIXES),
        "exclude_prefixes": list(EXCLUDE_PREFIXES),
        "num_pages": len(pages),
    }
    (CORPUS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Copied {len(pages)} pages from {REPO_URL}@{args.commit[:12]} into {OUT_DIR}")


if __name__ == "__main__":
    main()
