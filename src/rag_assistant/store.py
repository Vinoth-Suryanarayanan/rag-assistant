"""
store.py — a small on-disk vector index.

The index is two files in one folder:
  vectors.npy   float32 matrix, one normalised row per chunk
  chunks.json   chunk text + metadata, plus the embedding model name

Brute-force cosine search over a NumPy matrix is exact and fast up to
roughly 100k chunks.  Swap in a vector database beyond that.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rag_assistant.chunking import Chunk


class VectorStore:
    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        vectors: np.ndarray | None = None,
        embedding_model: str = "",
    ) -> None:
        self.chunks = list(chunks or [])
        self.vectors = vectors if vectors is not None else np.zeros((0, 0), dtype=np.float32)
        self.embedding_model = embedding_model
        if len(self.chunks) != len(self.vectors):
            raise ValueError("chunks and vectors must have the same length")

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks by cosine similarity, best first."""
        if not self.chunks or k <= 0:
            return []
        scores = self.vectors @ query_vector
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        meta = {
            "embedding_model": self.embedding_model,
            "chunks": [asdict(c) for c in self.chunks],
        }
        (directory / "chunks.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> VectorStore:
        directory = Path(directory)
        meta_path = directory / "chunks.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No index found in {directory}/ — run `make ingest` first.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vectors = np.load(directory / "vectors.npy")
        chunks = [Chunk(**c) for c in meta["chunks"]]
        return cls(chunks, vectors, meta.get("embedding_model", ""))
