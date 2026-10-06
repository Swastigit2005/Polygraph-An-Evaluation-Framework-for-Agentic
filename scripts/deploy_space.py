"""Create/update the Hugging Face Space for the live demo and verify it answers a question.

Prerequisite (once):  hf auth login     # paste a token with "write" permission

Usage:
    .venv/bin/python scripts/deploy_space.py [--space agentic-rag-evals] [--no-verify]

What it does:
  1. Creates (or reuses) a public Gradio Space <your-username>/<space>.
  2. Stores GROQ_API_KEY from your local .env as a Space *secret*, and the model/limit settings
     as Space *variables*. Optional Langfuse keys are stored as secrets if present.
  3. Uploads only git-tracked files (so .env, caches and the venv are never uploaded).
  4. Waits for the build and asks the live app one question to confirm it works.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
VARIABLES = ["GEN_MODEL", "FAST_MODEL", "JUDGE_MODEL", "GROQ_RPM_LIMIT", "GROQ_TPM_LIMIT", "MODEL_PRICES",
             "JSON_MODE_MODELS", "MODEL_MAX_OUTPUT", "RETRIEVAL_MODE", "USE_RERANKER", "USE_REFLECTION",
             "MAX_REWRITES", "TOP_K", "CANDIDATE_K", "EMBED_MODEL", "RERANK_MODEL", "LANGFUSE_HOST"]
SECRETS = ["GROQ_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]


def tracked_files() -> list[str]:
    """Files tracked by git (the only files ever uploaded)."""
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f and not f.startswith((".github/", "tests/"))]


def main() -> None:
    """Deploy and verify."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", default="agentic-rag-evals")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:  # noqa: BLE001
        sys.exit("Not logged in to Hugging Face. Run:  hf auth login   (token with write access)")
    repo_id = f"{user}/{args.space}"
    env = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v}
    if not env.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY missing from .env")

    print(f"Creating/reusing Space {repo_id} ...")
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
    for key in SECRETS:
        if env.get(key):
            api.add_space_secret(repo_id, key, env[key])
    for key in VARIABLES:
        if env.get(key):
            api.add_space_variable(repo_id, key, env[key].split("#")[0].strip())
    # The demo shares the free-tier quota: fail fast and show the "demo busy" message.
    api.add_space_variable(repo_id, "LLM_MAX_RETRIES", "2")

    url = f"https://huggingface.co/spaces/{repo_id}"
    readme = ROOT / "README.md"
    if "HF_SPACE_URL" in readme.read_text():
        readme.write_text(readme.read_text().replace("HF_SPACE_URL", url))
        print("Filled the Live Demo link in README.md (commit and push it to GitHub too).")

    files = tracked_files()
    print(f"Uploading {len(files)} tracked files ...")
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(ROOT), allow_patterns=files,
                      commit_message="Deploy from agentic-rag-evals")
    print(f"Uploaded. Space page: {url}")
    if args.no_verify:
        return

    print("Waiting for the Space to build (first build takes several minutes) ...")
    for _ in range(90):
        stage = api.get_space_runtime(repo_id).stage
        print(f"  stage: {stage}")
        if stage == "RUNNING":
            break
        if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"}:
            sys.exit(f"Space failed ({stage}); open {url}?logs=build to see why")
        time.sleep(20)
    else:
        sys.exit("Timed out waiting for the Space; check the logs on the Space page")

    from gradio_client import Client

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    client = Client(repo_id, verbose=False)
    answer, trace = client.predict("How do I limit how much memory a container can use?", api_name="/ask_kb")
    print("\nLive answer (first 300 chars):\n" + answer[:300])
    print(f"\n✅ Live demo: {url}")


if __name__ == "__main__":
    main()
