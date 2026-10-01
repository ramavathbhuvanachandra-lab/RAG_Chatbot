"""Reusable vector-store runtime adapter.

The vector-store implementation is generic. Institution-specific storage
location and collection name come from the active InstitutionProfile.
Embedding configuration comes from the runtime embedding adapter.
"""

from __future__ import annotations

from langchain_chroma import Chroma

from backend.runtime.config import RUNTIME
from backend.runtime.embedding import embeddings


def get_vectorstore() -> Chroma:
    """Create the vector store for the active institution deployment."""

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


# Backward-compatible runtime object for the migration period.
vectorstore = get_vectorstore()


__all__ = [
    "get_vectorstore",
    "vectorstore",
]