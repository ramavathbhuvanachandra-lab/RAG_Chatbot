"""Standalone retrieval adapters for the reusable AI platform.

Responsibilities
----------------
- load the active institution's canonical chunk corpus;
- provide deterministic BM25 retrieval over those chunks;
- provide dense retrieval over the active institution vector store;
- expose corpus diagnostics used by the reusable core.

Non-responsibilities
--------------------
- RRF fusion
- semantic alignment
- reranking policy
- evidence selection
- answer generation

Those concerns belong to ``ai_platform.core``.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import math
import re
from pathlib import Path
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from ai_platform.runtime.config import RUNTIME
from ai_platform.runtime.corpus import load_and_split
from ai_platform.runtime.vectorstore import get_vectorstore


RETRIEVER_K = 20


def _tokenize(text: str) -> set[str]:
    normalized = re.sub(
        r"[^a-z0-9\s]",
        " ",
        str(text or "").casefold(),
    )
    return {
        token
        for token in normalized.split()
        if len(token) > 2
    }


@lru_cache(maxsize=1)
def get_canonical_chunks() -> tuple[Document, ...]:
    """Load and deterministically chunk the active institution corpus once."""
    data_path = Path(RUNTIME.institution.data_path)
    return tuple(load_and_split(data_path))


@lru_cache(maxsize=1)
def get_bm25_retriever() -> BM25Retriever:
    """Build BM25 over the same canonical chunks used by the runtime."""
    documents = list(get_canonical_chunks())
    if not documents:
        raise RuntimeError(
            "The active institution corpus contains no retrievable documents."
        )

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = RETRIEVER_K
    return retriever


@lru_cache(maxsize=1)
def get_dense_retriever() -> Any:
    """Return the dense retriever for the active institution vector store."""
    return get_vectorstore().as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )


def dense_retrieve(query: str) -> list[Document]:
    """Retrieve documents using dense vector similarity."""
    question = str(query or "").strip()
    if not question:
        return []
    return list(get_dense_retriever().invoke(question))


def keyword_retrieve(query: str) -> list[Document]:
    """Retrieve documents using BM25 keyword scoring."""
    question = str(query or "").strip()
    if not question:
        return []
    return list(get_bm25_retriever().invoke(question))


def corpus_document_count() -> int:
    """Return the number of canonical chunks available to retrieval."""
    return len(get_canonical_chunks())


def corpus_token_document_frequency() -> dict[str, int]:
    """Return corpus-derived document frequencies for diagnostics."""
    frequency: defaultdict[str, int] = defaultdict(int)
    for document in get_canonical_chunks():
        for token in _tokenize(document.page_content):
            frequency[token] += 1
    return dict(frequency)


def corpus_term_idf(term: str) -> float:
    """Return a smoothed IDF estimate from the active canonical corpus."""
    token = str(term or "").casefold().strip()
    if not token:
        return 0.0

    document_count = max(corpus_document_count(), 1)
    document_frequency = corpus_token_document_frequency().get(token, 0)
    return math.log(
        (document_count + 1.0)
        / (document_frequency + 1.0)
    ) + 1.0


# ``CoreNodes`` uses this as the source for conservative local-context
# expansion. It is a data-only export; retrieval policy remains in core.
chunks = get_canonical_chunks()


__all__ = [
    "RETRIEVER_K",
    "chunks",
    "get_canonical_chunks",
    "get_bm25_retriever",
    "get_dense_retriever",
    "dense_retrieve",
    "keyword_retrieve",
    "corpus_document_count",
    "corpus_token_document_frequency",
    "corpus_term_idf",
]
