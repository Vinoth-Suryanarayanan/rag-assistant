"""
documents.py — load .md, .txt and .pdf files from a folder.

Each file becomes one Document whose `source` is its path relative to the
folder, so citations stay readable ("handbook/expenses.md").
Hidden files and folders (starting with ".") are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


@dataclass(frozen=True)
class Document:
    source: str
    text: str


def load_documents(root: Path) -> list[Document]:
    """Recursively load every supported, non-empty file under `root`."""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Docs folder not found: {root}")

    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        text = read_file(path)
        if text.strip():
            docs.append(Document(source=rel.as_posix(), text=text))
    return docs


def read_file(path: Path) -> str:
    """Return the plain text of one file."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # imported lazily: only needed for PDFs

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")
