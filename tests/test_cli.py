import pytest

from rag_assistant import cli
from rag_assistant.config import settings


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "index_dir", tmp_path / "idx")
    monkeypatch.setattr(settings, "embedding_model", "hash")
    monkeypatch.setattr(settings, "anthropic_api_key", "")


def test_ingest_then_ask(docs_dir, embedder, client, capsys):
    assert cli.main(["ingest", "--docs", str(docs_dir)], embedder=embedder) == 0
    assert "Indexed" in capsys.readouterr().out

    assert cli.main(["ask", "meal limit per day?"], embedder=embedder, client=client) == 0
    out = capsys.readouterr().out
    assert "60 dollars" in out
    assert "Sources:" in out and "policies/expenses.md" in out


def test_ask_without_index_fails_cleanly(embedder, client, capsys):
    assert cli.main(["ask", "anything"], embedder=embedder, client=client) == 1
    assert "make ingest" in capsys.readouterr().err


def test_ask_without_api_key_fails_cleanly(docs_dir, embedder, capsys):
    cli.main(["ingest", "--docs", str(docs_dir)], embedder=embedder)
    assert cli.main(["ask", "anything"], embedder=embedder) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_ask_detects_embedding_model_change(docs_dir, embedder, client, monkeypatch, capsys):
    cli.main(["ingest", "--docs", str(docs_dir)], embedder=embedder)
    monkeypatch.setattr(settings, "embedding_model", "other-model")
    assert cli.main(["ask", "anything"], embedder=embedder, client=client) == 1
    assert "make ingest" in capsys.readouterr().err
