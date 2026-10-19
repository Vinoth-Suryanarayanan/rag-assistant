# rag-assistant

Ask questions about your own documents and get answers with cited sources.

It indexes a folder of **Markdown, text and PDF** files using a **local embedding model**
(free, runs on your machine, no extra API key), then uses **Anthropic Claude** to write
an answer grounded in the most relevant passages.

---

## What it does

```
You:  make ask Q="How many vacation days roll over?"

Assistant:
  Up to 5 unused vacation days roll over into the next year; anything above
  that is forfeited on 31 December [1].

  Sources:
    [1] example-handbook.md (chunk 0)
```

---

## Architecture

```
src/rag_assistant/
├── documents.py    ← load .md / .txt / .pdf files from a folder
├── chunking.py     ← split into overlapping ~1,000-character passages
├── embeddings.py   ← local embeddings (fastembed) + offline test embedder
├── store.py        ← on-disk vector index (NumPy + JSON), cosine search
├── generate.py     ← Claude prompt: answer only from numbered passages, cite them
├── pipeline.py     ← build_index() and ask(), wiring the pieces together
├── cli.py          ← `ingest` and `ask` commands
└── config.py       ← settings from .env
```

**Ingest:** load → chunk → embed → save to `.index/`
**Ask:** embed question → top-k search → Claude answers with citations

Each piece is a small module with no hidden state, so it's easy to swap one out:
a vector database instead of `store.py`, a different embedding model, or a reranker
between retrieval and generation.

---

## Quick start

```bash
git clone https://github.com/Vinoth-Suryanarayanan/rag-assistant
cd rag-assistant

make setup      # installs deps with uv, creates .env
# edit .env → set ANTHROPIC_API_KEY

make ingest     # indexes data/docs (downloads the embedding model once, ~70 MB)
make ask Q="What is the daily meal allowance?"
```

To use your own documents:

```bash
make ingest DOCS=~/Documents/my-notes
```

or set `DOCS_DIR` in `.env`. Run `make ingest` again whenever the documents change.

---

## Voice app

A browser front end with speech in and out, on the same index and pipeline:

```bash
make app        # opens http://localhost:8501
```

```
🎙️ speak / ⌨️ type ─▶ Whisper ─▶ local embedding search ─▶ Claude ─▶ 🔊 OpenAI TTS
```

- Needs `ANTHROPIC_API_KEY` for answers. Add `OPENAI_API_KEY` to `.env` to turn on the
  microphone and spoken answers; without it the app works with typed questions.
- Uses the `make ingest` index if there is one, and builds it otherwise.
  **Re-index documents** in the sidebar picks up new or edited files.
- Citation markers like [1] are shown on screen but not read aloud.

For a deployable, OpenAI-only version with hybrid search and document upload, see
[voice-rag-assistant](https://github.com/Vinoth-Suryanarayanan/voice-rag-assistant).

---

## Configuration (`.env`)

| Variable            | Default                  | What it does                              |
|---------------------|--------------------------|-------------------------------------------|
| `ANTHROPIC_API_KEY` | —                        | Needed for `make ask` only                |
| `ANSWER_MODEL`      | `claude-haiku-5-5`       | Claude model that writes the answer       |
| `EMBEDDING_MODEL`   | `BAAI/bge-small-en-v1.5` | Local embedding model (re-ingest if changed) |
| `DOCS_DIR`          | `data/docs`              | Folder to index                           |
| `TOP_K`             | `5`                      | Passages sent to Claude per question      |
| `OPENAI_API_KEY`    | —                        | Voice app only: speech-to-text + speech   |
| `CHUNK_SIZE`        | `1000`                   | Characters per passage                    |
| `CHUNK_OVERLAP`     | `150`                    | Characters shared by neighbouring passages |

---

## Development

```bash
make test       # fully offline (fake embedder + mocked Claude)
make lint       # ruff
make doctor     # check prerequisites
```

CI runs lint and tests on every push and pull request (`.github/workflows/ci.yml`).

---

## Limitations

- PDFs are read as plain text; scanned PDFs need OCR first.
- The index is brute-force NumPy search: exact and fast up to ~100k passages.
- Each question is answered on its own; there's no chat memory between questions.

## License

MIT
