"""Write .streamlit/secrets.toml from your local .env (git-ignored) for Streamlit Community Cloud.

Usage:
    .venv/bin/python scripts/make_streamlit_secrets.py

Then open the generated file, copy its contents, and paste them into your app's
Settings → Secrets box on share.streamlit.io. The file never leaves your machine otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["GROQ_API_KEY", "GEN_MODEL", "FAST_MODEL", "JUDGE_MODEL", "GROQ_RPM_LIMIT", "GROQ_TPM_LIMIT",
        "MODEL_PRICES", "JSON_MODE_MODELS", "MODEL_MAX_OUTPUT", "RETRIEVAL_MODE", "USE_RERANKER",
        "USE_REFLECTION", "MAX_REWRITES", "TOP_K", "CANDIDATE_K", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST"]


def main() -> None:
    """Generate the secrets file."""
    env = dotenv_values(ROOT / ".env")
    lines = []
    for key in KEYS:
        value = (env.get(key) or "").split("#")[0].strip()
        if value:
            lines.append(f"{key} = {json.dumps(value)}")
    out = ROOT / ".streamlit" / "secrets.toml"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out} ({len(lines)} keys). Paste its contents into Settings → Secrets on share.streamlit.io.")


if __name__ == "__main__":
    main()
