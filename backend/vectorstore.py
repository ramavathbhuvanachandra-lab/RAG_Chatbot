"""
Reusable vector store access layer.

This module belongs to the reusable RAG runtime.

Deployment-specific values are supplied by the active
InstitutionProfile through backend.config.

No institution name, institution-specific path, or collection name is
defined directly in this module.
"""

from __future__ import annotations

from langchain_chroma import Chroma

from backend.config import (
    CHROMA_DB_PATH,
    VECTORSTORE_COLLECTION,
)
from backend.embedding import embeddings


# ---------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------
#
# These values come from the active institution deployment.
#
# The vectorstore implementation itself does not know which institution
# is currently active.
# ---------------------------------------------------------------------

PERSIST_DIRECTORY = str(
    CHROMA_DB_PATH
)

COLLECTION_NAME = (
    VECTORSTORE_COLLECTION
)


# ---------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
)


# ---------------------------------------------------------------------
# Compatibility export
# ---------------------------------------------------------------------
#
# Existing ingestion code currently imports VECTORSTORE_COLLECTION
# from this module.
#
# Keep the export temporarily while the backend is being migrated.
#
# IMPORTANT:
# This is not a hard-coded institution value.
# It is simply the value supplied by backend.config.
# ---------------------------------------------------------------------

__all__ = [
    "vectorstore",
    "VECTORSTORE_COLLECTION",
]