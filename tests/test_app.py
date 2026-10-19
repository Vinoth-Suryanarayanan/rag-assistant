"""
Headless tests of app.py (Streamlit AppTest).

Claude and OpenAI are mocks and HashEmbedder replaces the real embedding
model, so these run offline.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from rag_assistant.config import settings
from rag_assistant.embeddings import HashEmbedder
from tests.conftest import make_client

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def speech():
    client = MagicMock()
    client.audio.transcriptions.create.return_value = SimpleNamespace(
        text="How many vacation days roll over?", language="english"
    )
    client.audio.speech.create.return_value = SimpleNamespace(content=b"ID3-mp3")
    return client


@pytest.fixture
def env(docs_dir, tmp_path, monkeypatch, speech):
    claude = make_client("Up to 5 unused vacation days roll over [1].")
    monkeypatch.setattr(settings, "docs_dir", docs_dir)
    monkeypatch.setattr(settings, "index_dir", tmp_path / "idx")
    monkeypatch.setattr(settings, "embedding_model", "hash")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    st.cache_resource.clear()
    with patch("rag_assistant.clients.embedder", return_value=HashEmbedder()), \
         patch("rag_assistant.clients.anthropic_client", return_value=claude), \
         patch("rag_assistant.clients.openai_client", return_value=speech):
        yield SimpleNamespace(claude=claude, speech=speech)
    st.cache_resource.clear()


def _run() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def test_builds_the_index_when_none_exists(env):
    at = _run()
    assert not at.exception, at.exception
    assert "from 2 file(s)" in at.sidebar.caption[0].value
    assert (settings.index_dir / "chunks.json").exists()


def test_typed_question_answered_with_sources_and_audio(env):
    at = _run()
    at.chat_input[0].set_value("How many vacation days roll over?").run()
    assert not at.exception, at.exception
    turn = at.session_state.turns[0]
    assert "[1]" in turn["answer"]
    assert turn["sources"][0]["source"] == "policies/leave.txt"
    assert turn["audio"] == b"ID3-mp3"
    assert env.speech.audio.speech.create.call_args.kwargs["input"].endswith("roll over.")


def test_voice_question_transcribed_once(env):
    at = _run()
    at.audio_input[0].set_value(("q.wav", b"RIFF-fake", "audio/wav")).run()
    assert not at.exception, at.exception
    assert at.session_state.turns[0]["voice"] is True
    at.run()
    assert len(at.session_state.turns) == 1
    assert env.speech.audio.transcriptions.create.call_count == 1


def test_without_openai_key_text_still_works(env, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")
    at = _run()
    assert not at.audio_input
    assert any("OPENAI_API_KEY" in i.value for i in at.sidebar.info)
    at.chat_input[0].set_value("vacation days?").run()
    assert at.session_state.turns[0]["audio"] == b""


def test_without_anthropic_key_explains(env, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    at = _run()
    assert "ANTHROPIC_API_KEY" in at.error[0].value
