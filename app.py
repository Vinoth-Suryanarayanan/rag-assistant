"""
app.py — voice front end for rag-assistant (Streamlit).

    🎙️ speak / ⌨️ type ─▶ Whisper ─▶ local embeddings search ─▶ Claude ─▶ 🔊 OpenAI TTS

Same index and answer pipeline as `make ingest` / `make ask`; this file only
adds a browser UI with speech in and out.

    make app        (or: uv run streamlit run app.py)
"""

from __future__ import annotations

import hashlib

import streamlit as st

from rag_assistant import clients
from rag_assistant.config import settings
from rag_assistant.pipeline import ask, build_index
from rag_assistant.speech import synthesize, transcribe
from rag_assistant.store import VectorStore

VOICES = ["nova", "alloy", "shimmer", "echo", "onyx", "fable"]

st.set_page_config(page_title="RAG Assistant", page_icon="🎙️")
st.title("🎙️ RAG Assistant")
st.caption("Ask about your documents by voice or text. Answers cite the passages they use.")


def secret(name: str, fallback: str) -> str:
    """st.secrets (Streamlit Cloud / secrets.toml) first, then .env via settings."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return fallback


anthropic_key = secret("ANTHROPIC_API_KEY", settings.anthropic_api_key)
openai_key = secret("OPENAI_API_KEY", settings.openai_api_key)
if not anthropic_key:
    st.error("Set `ANTHROPIC_API_KEY` in `.env` (or Streamlit secrets) to get answers.")
    st.stop()


# ── Shared resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading the embedding model…")
def get_embedder():
    return clients.embedder(settings.embedding_model)


@st.cache_resource(show_spinner="Loading the document index…")
def get_store(_embedder, version: int) -> VectorStore:
    """Reuse the `make ingest` index when it matches; otherwise build one.

    `version` changes when the user presses "Re-index", which busts the cache.
    """
    if version == 0:
        try:
            store = VectorStore.load(settings.index_dir)
            if store.embedding_model in ("", settings.embedding_model):
                return store
        except FileNotFoundError:
            pass
    return build_index(
        settings.docs_dir,
        settings.index_dir,
        _embedder,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.embedding_model,
    )


@st.cache_resource(show_spinner=False)
def get_claude(key: str):
    return clients.anthropic_client(key)


@st.cache_resource(show_spinner=False)
def get_openai(key: str):
    return clients.openai_client(key)


state = st.session_state
state.setdefault("turns", [])
state.setdefault("index_version", 0)
state.setdefault("last_audio_hash", None)

embedder = get_embedder()
try:
    store = get_store(embedder, state.index_version)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Couldn't load the documents: {exc}")
    st.stop()
claude = get_claude(anthropic_key)
speech_client = get_openai(openai_key) if openai_key else None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    files = sorted({c.source for c in store.chunks})
    st.header("📚 Documents")
    st.caption(f"{len(store)} passages from {len(files)} file(s) in `{settings.docs_dir}`")
    for name in files:
        st.caption(f"• {name}")
    if st.button("Re-index documents", help="Run after adding or editing files."):
        state.index_version += 1
        st.rerun()

    st.header("⚙️ Answers")
    top_k = st.slider("Passages per answer", 1, 10, settings.top_k)

    st.header("🗣️ Voice")
    if speech_client is None:
        st.info("Set `OPENAI_API_KEY` to enable the microphone and spoken answers.")
    voice = st.selectbox("Voice", VOICES, index=VOICES.index(settings.tts_voice)
                         if settings.tts_voice in VOICES else 0, disabled=speech_client is None)
    speak = st.toggle("Read answers aloud", value=True, disabled=speech_client is None)

    if st.button("Clear chat"):
        state.turns = []
        st.rerun()


# ── Conversation ──────────────────────────────────────────────────────────────

for n, turn in enumerate(state.turns):
    with st.chat_message("user"):
        st.markdown(("🎙️ " if turn["voice"] else "") + turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if turn.get("audio"):
            latest = n == len(state.turns) - 1
            st.audio(turn["audio"], format="audio/mp3", autoplay=latest and turn["autoplay"])
            turn["autoplay"] = False
        if turn["sources"]:
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for i, s in enumerate(turn["sources"], start=1):
                    st.markdown(f"**[{i}] {s['source']}** — chunk {s['index']}")
                    st.caption(s["text"])

audio = st.audio_input("Ask with your voice") if speech_client else None
typed = st.chat_input("Ask a question about your documents")

question, via_voice, answered = None, False, False
try:
    if typed:
        question = typed.strip()
    elif audio is not None:
        recording = audio.getvalue()
        digest = hashlib.sha256(recording).hexdigest()
        if digest != state.last_audio_hash:  # reruns re-send the same recording
            state.last_audio_hash = digest
            with st.spinner("👂 Transcribing…"):
                question = transcribe(speech_client, recording, settings.transcribe_model).text
            via_voice = True

    if question:
        with st.spinner("🔍 Searching and asking Claude…"):
            answer = ask(question, store, embedder, claude, settings.answer_model, top_k,
                         settings.max_answer_tokens)
        audio_out = b""
        if speak and speech_client is not None:
            with st.spinner("🗣️ Speaking…"):
                audio_out = synthesize(speech_client, answer.text, settings.tts_model, voice)
        for old in state.turns:
            old["audio"] = None
        state.turns.append({
            "question": question,
            "answer": answer.text,
            "voice": via_voice,
            "audio": audio_out,
            "autoplay": True,
            "sources": [{"source": c.source, "index": c.index, "text": c.text}
                        for c in answer.sources],
        })
        answered = True
except Exception as exc:
    st.error(f"Something went wrong: {exc}")

if answered:
    st.rerun()
