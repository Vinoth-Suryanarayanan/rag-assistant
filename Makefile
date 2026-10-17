.PHONY: setup install ingest ask test lint doctor help

# ── Setup ─────────────────────────────────────────────────────────────────────

setup: install
	@if [ ! -f .env ]; then cp .env.example .env; echo "✓ .env created — fill in ANTHROPIC_API_KEY"; fi

install:
	uv sync

# ── Run ───────────────────────────────────────────────────────────────────────

# Usage: make ingest                 (indexes DOCS_DIR from .env, default data/docs)
#        make ingest DOCS=~/notes    (index another folder)
ingest:
	uv run python -m rag_assistant.cli ingest $(if $(DOCS),--docs "$(DOCS)")

# Usage: make ask Q="What is the expense limit?"
ask:
	@test -n "$(Q)" || (echo 'Usage: make ask Q="your question"' && exit 1)
	uv run python -m rag_assistant.cli ask "$(Q)"

# ── Quality ───────────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -v --tb=short

lint:
	uv run ruff check src tests

# ── Health check ──────────────────────────────────────────────────────────────

doctor:
	@echo "=== rag-assistant doctor ==="
	@command -v uv  >/dev/null 2>&1 && echo "✓ uv found"   || echo "✗ uv missing  →  brew install uv"
	@test -f .env               && echo "✓ .env found"     || echo "✗ .env missing  →  make setup"
	@grep -q "sk-ant-your_key" .env 2>/dev/null && echo "✗ ANTHROPIC_API_KEY not set" || echo "✓ ANTHROPIC_API_KEY set"
	@test -f .index/chunks.json && echo "✓ index built"    || echo "✗ no index yet  →  make ingest"
	@echo "============================"

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  make setup            Install deps + create .env"
	@echo "  make ingest           Index the docs folder (DOCS=path to override)"
	@echo "  make ask Q=\"...\"      Ask a question about your docs"
	@echo "  make test             Run all tests (no API keys or downloads needed)"
	@echo "  make lint             Ruff linter"
	@echo "  make doctor           Check prerequisites"
	@echo ""
