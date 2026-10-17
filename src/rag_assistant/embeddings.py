"""
embeddings.py — turn text into vectors.

FastEmbedEmbedder  local ONNX model (default BAAI/bge-small-en-v1.5).
                   No API key; the model downloads once on first use.
HashEmbedder       deterministic bag-of-words hashing.  No download, not
                   semantic — used by the tests so they run offline.

All vectors are L2-normalised, so a dot product is cosine similarity.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed passages → array of shape (len(texts), dim)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query → array of shape (dim,)."""
        ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding  # lazy: keeps imports fast

            self._model = TextEmbedding(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = np.array(list(self._load().embed(texts)), dtype=np.float32)
        return normalise(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        # query_embed adds the model's query instruction (matters for bge models).
        vector = np.array(list(self._load().query_embed(text)), dtype=np.float32)
        return normalise(vector)[0]


class HashEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                out[row, bucket] += 1.0
        return normalise(out)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
