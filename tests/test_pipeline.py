import pytest

from rag_assistant.pipeline import ask, build_index, retrieve
from rag_assistant.store import VectorStore


def test_build_index_saves_all_chunks(docs_dir, tmp_path, embedder):
    store = build_index(docs_dir, tmp_path / "idx", embedder, 1000, 100, "hash")
    assert {c.source for c in store.chunks} == {"policies/expenses.md", "policies/leave.txt"}
    assert len(VectorStore.load(tmp_path / "idx")) == len(store)


def test_retrieve_finds_the_relevant_file(docs_dir, tmp_path, embedder):
    store = build_index(docs_dir, tmp_path / "idx", embedder, 1000, 100)
    hits = retrieve("How many vacation days do employees get?", store, embedder, top_k=1)
    assert hits[0].source == "policies/leave.txt"


def test_ask_end_to_end(docs_dir, tmp_path, embedder, client):
    store = build_index(docs_dir, tmp_path / "idx", embedder, 1000, 100)
    answer = ask("meal reimbursement per day", store, embedder, client, "m", top_k=2)
    assert answer.sources[0].source == "policies/expenses.md"
    assert "[1]" in answer.text


def test_empty_folder_raises(tmp_path, embedder):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No .md"):
        build_index(tmp_path / "empty", tmp_path / "idx", embedder, 1000, 100)
