"""
Shared fixtures.

Everything runs offline: HashEmbedder stands in for the real embedding
model and Claude is a MagicMock, so no API key or model download is needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rag_assistant.embeddings import HashEmbedder


@pytest.fixture
def docs_dir(tmp_path):
    root = tmp_path / "docs"
    (root / "policies").mkdir(parents=True)
    (root / "policies" / "expenses.md").write_text(
        "# Expenses\n\nMeals while travelling are reimbursed up to 60 dollars per day. "
        "Submit receipts within 30 days.\n"
    )
    (root / "policies" / "leave.txt").write_text(
        "Employees get 20 days of paid vacation leave per year. "
        "Unused vacation days roll over up to a maximum of 5 days.\n"
    )
    (root / "notes.csv").write_text("not,indexed\n")          # unsupported type
    (root / ".hidden").mkdir()
    (root / ".hidden" / "secret.md").write_text("should be skipped")
    return root


@pytest.fixture
def embedder():
    return HashEmbedder()


def make_client(reply: str = "Meals are reimbursed up to 60 dollars per day [1].") -> MagicMock:
    """A stand-in for anthropic.Anthropic that returns `reply`."""
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=reply)]
    )
    return client


@pytest.fixture
def client():
    return make_client()
