"""Backward-compatible embedding bridge.

The active embedding configuration now lives in backend.runtime.embedding.
This module preserves the old import path used by the legacy retrieval and
ingestion modules during the migration.
"""

from backend.runtime.embedding import (
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    embeddings,
    get_embeddings,
)


__all__ = [
    "OLLAMA_BASE_URL",
    "EMBEDDING_MODEL",
    "get_embeddings",
    "embeddings",
]
