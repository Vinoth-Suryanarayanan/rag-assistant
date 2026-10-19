import pytest

from rag_assistant.chunking import chunk_documents, split_text
from rag_assistant.documents import Document


def test_short_text_is_one_chunk():
    assert split_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert split_text("   ", size=100, overlap=10) == []


def test_long_text_respects_size_and_overlaps():
    text = " ".join(f"word{i}" for i in range(500))
    pieces = split_text(text, size=200, overlap=40)
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)
    # neighbouring chunks share some words
    for a, b in zip(pieces, pieces[1:], strict=False):
        assert set(a.split()[-3:]) & set(b.split()[:10])


def test_overlap_never_starts_mid_word():
    text = " ".join(["supercalifragilistic"] * 60)
    for piece in split_text(text, size=100, overlap=20):
        assert set(piece.split()) == {"supercalifragilistic"}


def test_prefers_paragraph_boundaries():
    para = "A" * 70
    text = f"{para}\n\n{para}\n\n{para}"
    pieces = split_text(text, size=100, overlap=0)
    assert pieces == [para, para, para]


@pytest.mark.parametrize("size, overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_settings_raise(size, overlap):
    with pytest.raises(ValueError):
        split_text("text", size=size, overlap=overlap)


def test_chunk_ids_are_stable():
    docs = [Document("a.md", "one. " * 100), Document("b.md", "two")]
    chunks = chunk_documents(docs, size=120, overlap=20)
    assert chunks[0].id == "a.md#0"
    assert chunks[-1].id == "b.md#0"
    assert [c.index for c in chunks if c.source == "a.md"] == list(
        range(sum(c.source == "a.md" for c in chunks))
    )
