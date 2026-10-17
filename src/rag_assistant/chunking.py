"""
chunking.py — split documents into overlapping passages.

Chunks are sized in characters.  Each cut prefers a natural boundary
(paragraph → line → sentence → word) in the second half of the window,
and neighbouring chunks overlap so an answer that straddles a boundary
is still retrievable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from rag_assistant.documents import Document

_BOUNDARIES = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Chunk:
    id: str        # "<source>#<index>"
    source: str
    index: int
    text: str


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split `text` into pieces of at most `size` characters."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than the chunk size")

    text = re.sub(r"[ \t]+", " ", text).strip()
    pieces: list[str] = []
    start, n = 0, len(text)

    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            for sep in _BOUNDARIES:
                cut = window.rfind(sep)
                if cut > size // 2:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return pieces


def chunk_documents(docs: Iterable[Document], size: int, overlap: int) -> list[Chunk]:
    """Split every document and give each piece a stable id."""
    return [
        Chunk(id=f"{doc.source}#{i}", source=doc.source, index=i, text=piece)
        for doc in docs
        for i, piece in enumerate(split_text(doc.text, size, overlap))
    ]
