"""Reusable embedding runtime adapter.

Embedding implementation is deployment infrastructure. The model and Ollama
endpoint are configurable through environment variables and are not tied to
a particular institution.
"""

from __future__ import annotations

import os

from langchain_ollama import OllamaEmbeddings


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).strip()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text:latest",
).strip()


def get_embeddings() -> OllamaEmbeddings:
    """Create the configured embedding model."""

    if not OLLAMA_BASE_URL:
        raise ValueError("OLLAMA_BASE_URL cannot be empty")

    if not EMBEDDING_MODEL:
        raise ValueError("EMBEDDING_MODEL cannot be empty")

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


# Runtime singleton for the current application.
embeddings = get_embeddings()


__all__ = [
    "OLLAMA_BASE_URL",
    "EMBEDDING_MODEL",
    "get_embeddings",
    "embeddings",
]
