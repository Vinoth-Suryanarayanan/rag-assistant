import pytest

from rag_assistant.documents import load_documents


def test_loads_supported_files_with_relative_sources(docs_dir):
    docs = load_documents(docs_dir)
    assert [d.source for d in docs] == ["policies/expenses.md", "policies/leave.txt"]
    assert "60 dollars" in docs[0].text


def test_skips_hidden_unsupported_and_empty(docs_dir):
    (docs_dir / "empty.md").write_text("   \n")
    sources = {d.source for d in load_documents(docs_dir)}
    assert "notes.csv" not in sources
    assert ".hidden/secret.md" not in sources
    assert "empty.md" not in sources


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")
