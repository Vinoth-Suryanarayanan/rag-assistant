"""
pipeline.py — the two RAG flows, wired from the pieces above.

  build_index:  load files → chunk → embed → save index
  ask:          embed question → top-k search → Claude answers with citations
"""

from __future__ import annotations

from pathlib import Path

from rag_assistant.chunking import Chunk, chunk_documents
from rag_assistant.documents import load_documents
from rag_assistant.embeddings import Embedder
from rag_assistant.generate import Answer, generate_answer
from rag_assistant.store import VectorStore


def build_index(
    docs_dir: Path,
    index_dir: Path,
    embedder: Embedder,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str = "",
) -> VectorStore:
    docs = load_documents(docs_dir)
    if not docs:
        raise ValueError(f"No .md, .txt or .pdf files with text found in {docs_dir}")

    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    vectors = embedder.embed([c.text for c in chunks])
    store = VectorStore(chunks, vectors, embedding_model)
    store.save(index_dir)
    return store


def retrieve(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    top_k: int,
    min_score: float = 0.0,
) -> list[Chunk]:
    """Top-k passages for `question`, dropping any with no similarity at all."""
    hits = store.search(embedder.embed_query(question), top_k)
    return [chunk for chunk, score in hits if score > min_score]


def ask(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    client,
    model: str,
    top_k: int,
    max_tokens: int = 1024,
) -> Answer:
    passages = retrieve(question, store, embedder, top_k)
    return generate_answer(question, passages, client, model, max_tokens)
