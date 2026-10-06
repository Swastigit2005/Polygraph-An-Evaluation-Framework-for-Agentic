"""Show Groq token/request usage recorded by the LLM wrapper, per UTC day and model.

Free tier (Oct 2026): 1,000 requests and 200,000 tokens per model per day.
Only calls made with the disk cache enabled are recorded (cache hits cost nothing and are not counted).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_agent.config import get_settings  # noqa: E402
from rag_agent.llm import DiskCache  # noqa: E402

TPD, RPD = 200_000, 1_000


def main() -> None:
    """Print the usage ledger."""
    rows = DiskCache(get_settings().cache_dir / "llm_cache.sqlite").usage_by_day()
    if not rows:
        print("No usage recorded yet.")
    for day, model, req, tok in rows:
        print(f"{day}  {model:24} requests={req:5} ({100 * req / RPD:4.0f}% of RPD)  "
              f"tokens={tok:7,} ({100 * tok / TPD:4.0f}% of TPD)")


if __name__ == "__main__":
    main()
