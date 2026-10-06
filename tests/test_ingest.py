"""Chunking and Markdown cleaning."""

from __future__ import annotations

from pathlib import Path

from rag_agent.ingest import chunk_page, clean_markdown, doc_id_for, split_front_matter, split_sections


def test_front_matter_is_split() -> None:
    meta, body = split_front_matter("---\ntitle: Hello\n---\nBody text\n")
    assert meta == {"title": "Hello"}
    assert body == "Body text\n"


def test_clean_markdown_expands_and_strips_shortcodes(tmp_path: Path) -> None:
    (tmp_path / "frag.md").write_text("---\ntitle: x\n---\nIncluded text.")
    body = (
        'Version {{% param "docker_ce_version" %}}.\n'
        '{{% include "frag.md" %}}\n'
        '{{< tabs >}}\n{{< tab name="Linux" >}}\nRun it.\n{{< /tab >}}\n{{< /tabs >}}\n'
        "See [the guide](../guide.md) and <!-- hidden -->done."
    )
    out = clean_markdown(body, {"docker_ce_version": "29.8.2"}, tmp_path)
    assert "Version 29.8.2." in out
    assert "Included text." in out
    assert "**Linux:**" in out and "Run it." in out
    assert "{{" not in out and "hidden" not in out
    assert "See the guide and done." in out


def test_clean_markdown_keeps_go_templates_in_code() -> None:
    # Go templates in examples look like shortcodes but have no < or % delimiter.
    out = clean_markdown("docker inspect --format '{{json .Mounts}}' c1", {})
    assert "{{json .Mounts}}" in out


def test_doc_id_for() -> None:
    assert doc_id_for(Path("engine/daemon/_index.md")) == "engine/daemon"
    assert doc_id_for(Path("engine/daemon/logs.md")) == "engine/daemon/logs"


def test_split_sections_tracks_heading_path_and_ignores_fenced_hashes() -> None:
    md = "Intro\n## A\ntext a\n```bash\n# not a heading\n```\n### B\ntext b\n## C\ntext c"
    sections = split_sections(md)
    assert [s for s, _ in sections] == ["", "A", "A > B", "C"]
    assert "# not a heading" in sections[1][1]


def test_chunk_page_ids_are_stable_and_sequential() -> None:
    md = "## Install\n" + ("Install docker with the convenience script. " * 10) + "\n## Remove\nUninstall docker " * 3
    chunks = chunk_page("engine/install", "Install", "u", md, chunk_size=200, chunk_overlap=20)
    assert [c.id for c in chunks] == [f"engine/install#{i:03d}" for i in range(len(chunks))]
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].section == "Install"
    assert chunks[0].header() == "Install > Install"


def test_chunk_page_drops_tiny_fragments() -> None:
    assert chunk_page("d", "T", "u", "## H\nshort", 200, 20) == []
