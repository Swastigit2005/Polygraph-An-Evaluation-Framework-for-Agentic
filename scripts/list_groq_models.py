"""List the models currently available on your Groq account, plus your current rate-limit headers.

Usage:
    python scripts/list_groq_models.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    """Print active Groq model ids with context window and owner."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    key = os.getenv("GROQ_API_KEY")
    if not key:
        sys.exit("GROQ_API_KEY is not set (put it in .env)")
    import groq

    client = groq.Groq(api_key=key)
    models = sorted(client.models.list().data, key=lambda m: m.id)
    for m in models:
        if getattr(m, "active", True):
            ctx = getattr(m, "context_window", "?")
            print(f"{m.id:55} ctx={ctx!s:>7}  owner={getattr(m, 'owned_by', '?')}")

    probe = os.getenv("FAST_MODEL") or (models[0].id if models else None)
    if probe:
        raw = client.chat.completions.with_raw_response.create(
            model=probe, messages=[{"role": "user", "content": "ping"}], max_completion_tokens=1)
        print(f"\nRate-limit headers from a 1-token call to {probe}:")
        for k, v in raw.headers.items():
            if k.startswith("x-ratelimit"):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
