"""
generate.py — write a grounded, cited answer with Claude.

The retrieved passages are numbered [1], [2], … and Claude is told to
answer only from them and cite the numbers it used.  If retrieval found
nothing, we answer without calling the API at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_assistant.chunking import Chunk

SYSTEM_PROMPT = """\
You answer questions using only the numbered context passages you are given.

Rules:
- Cite the passages you rely on inline, like [1] or [2][3].
- If the passages don't contain the answer, say you couldn't find it in the \
documents. Don't guess and don't use outside knowledge.
- Be concise and direct."""

NO_CONTEXT_ANSWER = "I couldn't find anything relevant to that in the indexed documents."


@dataclass
class Answer:
    text: str
    sources: list[Chunk] = field(default_factory=list)


def build_context(passages: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {chunk.source})\n{chunk.text}"
        for i, chunk in enumerate(passages, start=1)
    )


def generate_answer(
    question: str,
    passages: list[Chunk],
    client,
    model: str,
    max_tokens: int = 1024,
) -> Answer:
    """Ask Claude to answer `question` from `passages`.  `client` is an anthropic.Anthropic."""
    if not passages:
        return Answer(NO_CONTEXT_ANSWER, [])

    prompt = f"Context passages:\n\n{build_context(passages)}\n\nQuestion: {question}"
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    return Answer(text.strip(), passages)
