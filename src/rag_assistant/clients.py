"""clients.py — where API clients are created (the app tests patch these)."""

from __future__ import annotations


def anthropic_client(api_key: str):
    import anthropic

    return anthropic.Anthropic(api_key=api_key)


def openai_client(api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def embedder(model_name: str):
    from rag_assistant.embeddings import FastEmbedEmbedder

    return FastEmbedEmbedder(model_name)
