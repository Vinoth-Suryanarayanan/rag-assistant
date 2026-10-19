"""
speech.py — voice in and out for the Streamlit app, using OpenAI.

    transcribe   Whisper speech-to-text (verbose_json also returns the language)
    synthesize   OpenAI text-to-speech → MP3 bytes

Answers still come from Claude; OpenAI is only the ears and the voice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CITATION = re.compile(r"\s*\[\d+\](\[\d+\])*")


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str


def transcribe(client, audio: bytes, model: str = "whisper-1") -> Transcript:
    response = client.audio.transcriptions.create(
        model=model, file=("question.wav", audio), response_format="verbose_json"
    )
    language = (getattr(response, "language", None) or "english").strip().title()
    return Transcript((response.text or "").strip(), language)


def synthesize(client, text: str, model: str = "tts-1", voice: str = "nova") -> bytes:
    text = strip_citations(text)
    if not text:
        return b""
    response = client.audio.speech.create(
        model=model, voice=voice, input=text[:4096], response_format="mp3"
    )
    return response.content


def strip_citations(text: str) -> str:
    """Remove [1]-style markers so they aren't read aloud."""
    return _CITATION.sub("", text).strip()
