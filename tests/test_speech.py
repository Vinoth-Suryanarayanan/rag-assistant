from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_assistant.speech import strip_citations, synthesize, transcribe


def _openai():
    client = MagicMock()
    client.audio.transcriptions.create.return_value = SimpleNamespace(
        text=" How many vacation days? ", language="english"
    )
    client.audio.speech.create.return_value = SimpleNamespace(content=b"ID3-mp3")
    return client


def test_transcribe_asks_for_language_and_trims():
    client = _openai()
    result = transcribe(client, b"RIFF")
    assert result.text == "How many vacation days?" and result.language == "English"
    assert client.audio.transcriptions.create.call_args.kwargs["response_format"] == "verbose_json"


def test_synthesize_strips_citations_before_speaking():
    client = _openai()
    assert synthesize(client, "Twenty days [1][2].", voice="alloy") == b"ID3-mp3"
    kwargs = client.audio.speech.create.call_args.kwargs
    assert kwargs["input"] == "Twenty days." and kwargs["voice"] == "alloy"


def test_synthesize_skips_empty_text():
    client = _openai()
    assert synthesize(client, " [1] ") == b""
    client.audio.speech.create.assert_not_called()


def test_strip_citations():
    assert strip_citations("A [1]. B [2][3].") == "A. B."
