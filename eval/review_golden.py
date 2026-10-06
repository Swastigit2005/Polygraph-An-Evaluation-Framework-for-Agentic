"""Interactive review of drafted golden-set items. Only items you accept become "verified": true.

Usage:
    python eval/review_golden.py            # review unreviewed items
    python eval/review_golden.py --all      # revisit every item (including accepted/rejected)
    python eval/review_golden.py --stats    # print counts and exit
    python eval/review_golden.py --proposals eval/ai_review_proposals.json
                                            # show an AI pre-review per item; Enter = take it

Keys (per item):
    a  accept (verified=true)         x  reject (excluded from metrics)
    q  edit question                  r  edit reference answer
    t  toggle a candidate chunk as relevant / not relevant (by number)
    v  view a candidate chunk in full (by number)
    n  add/replace a note             s  skip for now
    Q  save and quit
    Enter  (with --proposals) apply the shown proposal: accept / accept with edit / reject
Changes are saved to eval/golden_set.jsonl after every action.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_agent.config import get_settings  # noqa: E402
try:
    from rag_agent.ingest import read_chunks  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    sys.exit(f"{exc}. Run with the project venv: .venv/bin/python {Path(__file__).relative_to(Path.cwd())} ...")

GOLDEN = Path(__file__).resolve().parent / "golden_set.jsonl"
BOLD, DIM, GREEN, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"

try:  # pre-fill edits with the current text where readline supports it
    import readline
except ImportError:  # pragma: no cover
    readline = None  # type: ignore[assignment]


def load(path: Path = GOLDEN) -> list[dict]:
    """Read golden items from JSONL."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(items: list[dict], path: Path = GOLDEN) -> None:
    """Atomically write golden items to JSONL."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    tmp.replace(path)


def stats(items: list[dict]) -> str:
    """One-line summary of review progress by type."""
    c: Counter[str] = Counter()
    for it in items:
        state = "verified" if it.get("verified") else "rejected" if it.get("rejected") else "pending"
        c[f"{it['type']}:{state}"] += 1
    by_state = Counter("verified" if it.get("verified") else "rejected" if it.get("rejected") else "pending"
                       for it in items)
    return f"{len(items)} items — {dict(by_state)}\n" + "  ".join(f"{k}={v}" for k, v in sorted(c.items()))


def _edit(prompt: str, current: str) -> str:
    if readline is not None:
        readline.set_startup_hook(lambda: readline.insert_text(current))
    try:
        new = input(prompt).strip()
    finally:
        if readline is not None:
            readline.set_startup_hook()
    return new or current


def _wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(text, width=100, initial_indent=indent, subsequent_indent=indent)


def show(item: dict, idx: int, total: int, chunks: dict) -> list[str]:
    """Print an item; return the numbered candidate chunk ids."""
    print("\n" + "=" * 100)
    status = f"{GREEN}verified{RESET}" if item.get("verified") else f"{RED}rejected{RESET}" \
        if item.get("rejected") else "pending"
    print(f"{BOLD}[{idx + 1}/{total}] {item['id']}{RESET}  type={item['type']}  {status}"
          f"  should_refuse={item['should_refuse']}")
    print(f"{BOLD}Question:{RESET}\n{_wrap(item['question'])}")
    if item["reference_answer"]:
        print(f"{BOLD}Reference answer:{RESET}\n{_wrap(item['reference_answer'])}")
    if item.get("notes"):
        print(f"{BOLD}Notes:{RESET} {item['notes']}")
    cands = list(dict.fromkeys(item["relevant_chunk_ids"] + item.get("pool_ids", [])))
    rel = set(item["relevant_chunk_ids"])
    label = "Candidate chunks (✓ = gold relevant)" if not item["should_refuse"] else \
        "Retrieved pool (none should answer the question)"
    print(f"{BOLD}{label}:{RESET}")
    for n, cid in enumerate(cands, 1):
        c = chunks.get(cid)
        mark = f"{GREEN}✓{RESET}" if cid in rel else " "
        snippet = (c.text[:220].replace("\n", " ") + "…") if c else "(missing chunk)"
        head = c.header() if c else ""
        print(f" {mark} {n:2}. {cid}  {DIM}{head}{RESET}\n{DIM}{_wrap(snippet, '        ')}{RESET}")
    return cands


def review(items: list[dict], revisit_all: bool, proposals: dict[str, dict] | None = None) -> None:
    """Main interactive loop. Proposals are only suggestions; nothing is applied without a key press."""
    chunks = {c.id: c for c in read_chunks(get_settings().index_dir / "chunks.jsonl")}
    queue = [i for i, it in enumerate(items) if revisit_all or not (it.get("verified") or it.get("rejected"))]
    print(stats(items))
    if not queue:
        print("Nothing to review. Use --all to revisit items.")
        return
    print(__doc__.split("Keys (per item):")[1].split("Changes are")[0])
    for qpos, i in enumerate(queue):
        item = items[i]
        while True:
            cands = show(item, qpos, len(queue), chunks)
            prop = (proposals or {}).get(item["id"])
            if prop:
                print(f"{BOLD}Proposal:{RESET} {prop['decision']} — {prop['reason']}")
                for field, value in prop.get("edit", {}).items():
                    print(f"  {field} → {value}")
            hint = "[Enter]=proposal " if prop else ""
            keys = "[a]ccept [x]reject [q]uestion [r]ef [t]oggle [v]iew [n]ote [s]kip [Q]uit"
            cmd = input(f"{BOLD}{hint}{keys} > {RESET}")
            cmd = cmd.strip()
            if cmd == "" and prop:
                item.update(prop.get("edit", {}))
                cmd = "x" if prop["decision"] == "reject" else "a"
                if cmd == "x":
                    item["notes"] = f"rejected: {prop['reason']}"
            elif cmd == "":
                continue
            if cmd == "a":
                if not item["should_refuse"] and not item["relevant_chunk_ids"]:
                    print(f"{RED}An answerable item needs at least one relevant chunk.{RESET}")
                    continue
                item.update(verified=True, rejected=False, reviewer="human",
                            reviewed_at=datetime.now(UTC).isoformat(timespec="seconds"))
            elif cmd == "x":
                item.update(verified=False, rejected=True, reviewer="human",
                            reviewed_at=datetime.now(UTC).isoformat(timespec="seconds"))
            elif cmd == "q":
                item["question"] = _edit("Question: ", item["question"])
                save(items)
                continue
            elif cmd == "r":
                item["reference_answer"] = _edit("Reference answer: ", item["reference_answer"])
                save(items)
                continue
            elif cmd in ("t", "v"):
                raw = input("Chunk number: ").strip()
                if not raw.isdigit() or not 1 <= int(raw) <= len(cands):
                    print("Invalid number.")
                    continue
                cid = cands[int(raw) - 1]
                if cmd == "v":
                    c = chunks.get(cid)
                    print(f"\n{BOLD}{cid}{RESET} — {c.header() if c else ''}\n{c.text if c else '(missing)'}\n")
                    input(f"{DIM}(enter to continue){RESET}")
                elif item["should_refuse"]:
                    print("Unanswerable items have no relevant chunks. If this chunk answers the question, "
                          "reject the item with [x].")
                else:
                    rel = item["relevant_chunk_ids"]
                    item["relevant_chunk_ids"] = [r for r in rel if r != cid] if cid in rel else rel + [cid]
                    save(items)
                continue
            elif cmd == "n":
                item["notes"] = _edit("Note: ", item.get("notes", ""))
                save(items)
                continue
            elif cmd == "s":
                pass
            elif cmd == "Q":
                save(items)
                print(stats(items))
                return
            else:
                continue
            save(items)
            break
    print("\nReview queue finished.\n" + stats(items))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="revisit accepted/rejected items too")
    ap.add_argument("--stats", action="store_true", help="print review progress and exit")
    ap.add_argument("--proposals", type=Path, default=None, help="JSON of AI pre-review proposals to display")
    args = ap.parse_args()
    items = load()
    if args.stats:
        print(stats(items))
        return
    try:
        proposals = json.loads(args.proposals.read_text())["proposals"] if args.proposals else None
        review(items, args.all, proposals)
    except (KeyboardInterrupt, EOFError):
        save(items)
        print("\nSaved.\n" + stats(items))


if __name__ == "__main__":
    main()
