# Corpus sources

| Field | Value |
|---|---|
| Source | Docker documentation — https://github.com/docker/docs |
| Commit | `6cf1b1c167f032e8a6629da211602300b623b20e` (2026-10-06) |
| License | Apache License 2.0 — full text in [`docker-docs/LICENSE`](docker-docs/LICENSE) |
| Pages | 242 Markdown pages from `content/manuals/` (see `manifest.json`) |
| Included | `engine/`, `compose/`, `build/`, `desktop/{setup,troubleshoot-and-support,settings-and-maintenance,use-desktop}/` |
| Excluded | `engine/release-notes/`, `engine/swarm/`, `build/ci/`, `build/policies/`, `compose/releases/` |
| Also copied | `content/includes/` (fragments expanded by `{{% include %}}`), scalar version params from `hugo.yaml` → `docker-docs/hugo_params.json` |

## Modifications

Files under `docker-docs/` are unmodified copies. At index time, `src/rag_agent/ingest.py`
strips front matter, expands `{{% include %}}` and `{{% param %}}` shortcodes, turns
`{{< tab >}}` blocks into bold labels, removes other Hugo shortcodes, HTML comments and
link targets (keeping link text), then chunks the text. Docker and the Docker logo are
trademarks of Docker, Inc.; this project is not affiliated with or endorsed by Docker.

Reproduce with: `python scripts/fetch_corpus.py --commit 6cf1b1c167f032e8a6629da211602300b623b20e`
