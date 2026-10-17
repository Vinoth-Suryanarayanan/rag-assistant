"""
cli.py — command-line entry point.

    python -m rag_assistant.cli ingest [--docs PATH]
    python -m rag_assistant.cli ask "question" [-k N]

(`make ingest` and `make ask Q="..."` wrap these.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_assistant.config import settings
from rag_assistant.embeddings import Embedder, FastEmbedEmbedder
from rag_assistant.pipeline import ask, build_index
from rag_assistant.store import VectorStore


def main(argv: list[str] | None = None, embedder: Embedder | None = None, client=None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-assistant", description="Ask questions about your own documents."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index a folder of .md/.txt/.pdf files")
    p_ingest.add_argument("--docs", type=Path, default=settings.docs_dir)

    p_ask = sub.add_parser("ask", help="answer a question from the index")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", "--top-k", type=int, default=settings.top_k)

    args = parser.parse_args(argv)
    embedder = embedder or FastEmbedEmbedder(settings.embedding_model)

    try:
        if args.command == "ingest":
            return _ingest(args.docs.expanduser(), embedder)
        return _ask(args.question, args.top_k, embedder, client)
    except (FileNotFoundError, ValueError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1


def _ingest(docs_dir: Path, embedder: Embedder) -> int:
    print(f"📚  Indexing {docs_dir} …")
    store = build_index(
        docs_dir,
        settings.index_dir,
        embedder,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.embedding_model,
    )
    n_files = len({c.source for c in store.chunks})
    print(f"✓ Indexed {len(store)} chunks from {n_files} file(s) → {settings.index_dir}/")
    return 0


def _ask(question: str, top_k: int, embedder: Embedder, client) -> int:
    store = VectorStore.load(settings.index_dir)
    if store.embedding_model and store.embedding_model != settings.embedding_model:
        print(
            f"⚠️  Index was built with {store.embedding_model}, but EMBEDDING_MODEL is "
            f"{settings.embedding_model}. Run `make ingest` again.",
            file=sys.stderr,
        )
        return 1

    if client is None:
        if not settings.anthropic_api_key:
            print("✗ ANTHROPIC_API_KEY is not set — add it to .env", file=sys.stderr)
            return 1
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    answer = ask(
        question,
        store,
        embedder,
        client,
        settings.answer_model,
        top_k,
        settings.max_answer_tokens,
    )

    print(answer.text)
    if answer.sources:
        print("\nSources:")
        for i, chunk in enumerate(answer.sources, start=1):
            print(f"  [{i}] {chunk.source} (chunk {chunk.index})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
