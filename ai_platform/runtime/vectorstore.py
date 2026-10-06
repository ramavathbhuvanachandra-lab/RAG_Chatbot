"""Standalone vector-store runtime for the reusable AI platform."""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma

from ai_platform.runtime.config import RUNTIME
from ai_platform.runtime.embedding import embeddings


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Return the active institution's Chroma collection."""
    profile = RUNTIME.institution
    profile.validate()

    collection_name = (
        profile.vectorstore_collection
        or profile.institution_id
    )

    return Chroma(
        collection_name=collection_name,
        persist_directory=str(profile.vectorstore_path),
        embedding_function=embeddings,
    )


# Lazily created through get_vectorstore(); kept as a platform-only convenience
# for callers that explicitly want the active store object.
vectorstore = get_vectorstore()


__all__ = ["get_vectorstore", "vectorstore"]
