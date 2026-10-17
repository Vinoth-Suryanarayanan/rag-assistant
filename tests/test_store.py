import pytest

from rag_assistant.chunking import Chunk
from rag_assistant.store import VectorStore


def _store(embedder):
    chunks = [
        Chunk("a#0", "a", 0, "vacation leave days per year"),
        Chunk("b#0", "b", 0, "meal expenses reimbursed with receipts"),
        Chunk("c#0", "c", 0, "office wifi password reset"),
    ]
    return VectorStore(chunks, embedder.embed([c.text for c in chunks]), "hash")


def test_search_ranks_most_similar_first(embedder):
    store = _store(embedder)
    hits = store.search(embedder.embed_query("how many vacation days"), k=2)
    assert hits[0][0].id == "a#0"
    assert len(hits) == 2
    assert hits[0][1] >= hits[1][1]


def test_empty_store_returns_nothing(embedder):
    assert VectorStore().search(embedder.embed_query("anything"), k=3) == []


def test_save_and_load_round_trip(tmp_path, embedder):
    store = _store(embedder)
    store.save(tmp_path / "idx")
    loaded = VectorStore.load(tmp_path / "idx")
    assert loaded.chunks == store.chunks
    assert loaded.embedding_model == "hash"
    assert (loaded.vectors == store.vectors).all()


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="make ingest"):
        VectorStore.load(tmp_path / "missing")


def test_mismatched_lengths_raise(embedder):
    with pytest.raises(ValueError):
        VectorStore([Chunk("a#0", "a", 0, "x")], embedder.embed(["x", "y"]))
