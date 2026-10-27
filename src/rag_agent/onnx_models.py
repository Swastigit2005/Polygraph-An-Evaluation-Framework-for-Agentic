"""ONNX Runtime versions of the embedder and cross-encoder (no torch).

Used by the serverless demo, where a PyTorch install would not fit the function size limit.
Both classes mirror the sentence-transformers methods the retriever calls (`encode`, `predict`),
so `Retriever` works unchanged with either backend.

Usage (download the model files once):
    python -m rag_agent.onnx_models --download
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

# Exported ONNX copies of the same checkpoints the torch backend loads.
ONNX_REPOS = {
    "BAAI/bge-small-en-v1.5": "Xenova/bge-small-en-v1.5",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "Xenova/ms-marco-MiniLM-L-6-v2",
}
ONNX_FILE = "onnx/model_quantized.onnx"  # int8: ~4x smaller than fp32, same rankings in our checks
MAX_TOKENS = 512


def model_dir(root: Path, model_name: str) -> Path:
    """Local folder holding `model.onnx` and `tokenizer.json` for `model_name`."""
    return root / model_name.split("/")[-1]


class _OnnxModel:
    def __init__(self, path: Path) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tok.enable_truncation(MAX_TOKENS)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(path / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def _run(self, encodings: list[Any]) -> np.ndarray:
        feed = {"input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64)}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        return self.session.run(None, feed)[0]


class OnnxEmbedder(_OnnxModel):
    """BGE embedder: CLS pooling, optional L2 normalisation (as sentence-transformers does for BGE)."""

    def encode(self, texts: list[str], normalize_embeddings: bool = True, batch_size: int = 32) -> np.ndarray:
        """Embed `texts` and return a (n, dim) float32 array."""
        out = []
        for i in range(0, len(texts), batch_size):
            hidden = self._run(self.tok.encode_batch(texts[i:i + batch_size]))
            out.append(hidden[:, 0])
        vecs = np.concatenate(out).astype(np.float32)
        if normalize_embeddings:
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs


class OnnxCrossEncoder(_OnnxModel):
    """Cross-encoder returning one relevance logit per (query, passage) pair."""

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> np.ndarray:
        """Score each (query, passage) pair; higher means more relevant."""
        out = []
        for i in range(0, len(pairs), batch_size):
            out.append(self._run(self.tok.encode_batch(pairs[i:i + batch_size]))[:, 0])
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def download(root: Path) -> None:
    """Fetch the ONNX model and tokenizer for every model in ONNX_REPOS into `root`."""
    import shutil

    from huggingface_hub import hf_hub_download

    for name, repo in ONNX_REPOS.items():
        dest = model_dir(root, name)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(repo, ONNX_FILE), dest / "model.onnx")
        shutil.copy(hf_hub_download(repo, "tokenizer.json"), dest / "tokenizer.json")
        print(f"{name} -> {dest}")


if __name__ == "__main__":
    from rag_agent.config import get_settings

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="download the ONNX models into ONNX_DIR")
    if ap.parse_args().download:
        download(get_settings().onnx_dir)
