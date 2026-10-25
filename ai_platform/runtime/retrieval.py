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
from ai_platform.runtime.embedding import embeddings
from ai_platform.runtime.vectorstore import get_vectorstore


RETRIEVER_K = 25


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


@lru_cache(maxsize=8)
def _cached_canonical_chunks(target_id: str) -> tuple[Document, ...]:
    from ai_platform.institutions.loader import load_institution_profile

    profile = load_institution_profile(target_id)
    data_path = Path(profile.data_path)
    return tuple(load_and_split(data_path))


def get_canonical_chunks(institution_id: str | None = None) -> tuple[Document, ...]:
    """Load and deterministically chunk the active or requested institution corpus."""
    target_id = (
        institution_id
        or RUNTIME.institution.institution_id
    ).strip().lower()
    return _cached_canonical_chunks(target_id)


@lru_cache(maxsize=8)
def _cached_bm25_retriever(target_id: str) -> BM25Retriever:
    documents = list(_cached_canonical_chunks(target_id))
    if not documents:
        raise RuntimeError(
            f"The institution corpus '{target_id}' contains no retrievable documents."
        )

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = RETRIEVER_K
    return retriever


def get_bm25_retriever(institution_id: str | None = None) -> BM25Retriever:
    """Build BM25 over the same canonical chunks used by the runtime."""
    target_id = (
        institution_id
        or RUNTIME.institution.institution_id
    ).strip().lower()
    return _cached_bm25_retriever(target_id)


@lru_cache(maxsize=8)
def _cached_dense_retriever(target_id: str) -> Any:
    return get_vectorstore(target_id).as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )


def get_dense_retriever(institution_id: str | None = None) -> Any:
    """Return the dense retriever for the active or requested institution vector store."""
    target_id = (
        institution_id
        or RUNTIME.institution.institution_id
    ).strip().lower()
    return _cached_dense_retriever(target_id)


def dense_retrieve(query: str, institution_id: str | None = None) -> list[Document]:
    """Retrieve documents using dense vector similarity, excluding quarantined fallback buffers."""
    question = str(query or "").strip()
    if not question:
        return []
    raw_hits = list(get_dense_retriever(institution_id).invoke(question))
    from ai_platform.runtime.corpus import is_quarantined_fallback
    return [
        doc for doc in raw_hits
        if not is_quarantined_fallback(getattr(doc, "metadata", {}).get("source", ""))
    ]


def dense_retrieve_with_scores(
    query: str,
    institution_id: str | None = None,
    k: int = RETRIEVER_K,
) -> list[tuple[Document, float]]:
    """Retrieve documents using dense vector similarity along with similarity distance."""
    question = str(query or "").strip()
    if not question:
        return []
    store = get_vectorstore(institution_id)
    raw_hits = list(store.similarity_search_with_score(question, k=k))
    from ai_platform.runtime.corpus import is_quarantined_fallback
    return [
        (doc, float(score))
        for doc, score in raw_hits
        if not is_quarantined_fallback(getattr(doc, "metadata", {}).get("source", ""))
    ]


import threading

_QUERY_EMBEDDING_CACHE: dict[tuple[str, str], list[float]] = {}
_EMBEDDING_CACHE_LOCK = threading.Lock()
_MAX_EMBEDDING_CACHE_SIZE = 1024


def clear_query_embedding_cache() -> None:
    """Clear query embedding cache."""
    with _EMBEDDING_CACHE_LOCK:
        _QUERY_EMBEDDING_CACHE.clear()


def dense_retrieve_batch_with_scores(
    queries: Sequence[str],
    institution_id: str | None = None,
    k: int = RETRIEVER_K,
) -> tuple[dict[str, list[tuple[Document, float]]], float, float]:
    """Retrieve documents for multiple queries using batched embedding and vector search.

    Parameters:
        queries: Sequence of query strings to retrieve candidates for.
        institution_id: Optional institution tenant identifier.
        k: Maximum documents per query.

    Returns:
        tuple of (results_by_query, embedding_duration_seconds, chroma_search_duration_seconds)
    """
    import time
    clean_queries = [str(q or "").strip() for q in queries if str(q or "").strip()]
    if not clean_queries:
        return {}, 0.0, 0.0

    distinct_queries = list(dict.fromkeys(clean_queries))
    model_id = str(getattr(embeddings, "model_name", getattr(embeddings, "model", "default")))

    # 1. Batched query embedding generation with LRU-style bounded caching
    t_e0 = time.perf_counter()
    with _EMBEDDING_CACHE_LOCK:
        missing_queries = [q for q in distinct_queries if (model_id, q) not in _QUERY_EMBEDDING_CACHE]

    if missing_queries:
        new_vecs = embeddings.embed_documents(missing_queries)
        with _EMBEDDING_CACHE_LOCK:
            for q, v in zip(missing_queries, new_vecs):
                if len(_QUERY_EMBEDDING_CACHE) >= _MAX_EMBEDDING_CACHE_SIZE:
                    try:
                        del _QUERY_EMBEDDING_CACHE[next(iter(_QUERY_EMBEDDING_CACHE))]
                    except (KeyError, StopIteration):
                        pass
                _QUERY_EMBEDDING_CACHE[(model_id, q)] = v

    with _EMBEDDING_CACHE_LOCK:
        vecs = [_QUERY_EMBEDDING_CACHE[(model_id, q)] for q in distinct_queries]
    embedding_duration = time.perf_counter() - t_e0

    # 2. Batched Chroma vector search
    t_c0 = time.perf_counter()
    store = get_vectorstore(institution_id)
    from ai_platform.runtime.corpus import is_quarantined_fallback

    collection = getattr(store, "_collection", None)
    results_by_query: dict[str, list[tuple[Document, float]]] = {}

    if collection is not None and hasattr(collection, "query"):
        raw_res = collection.query(
            query_embeddings=vecs,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        chroma_duration = time.perf_counter() - t_c0

        docs_per_q = raw_res.get("documents", []) or []
        metas_per_q = raw_res.get("metadatas", []) or []
        dists_per_q = raw_res.get("distances", []) or []

        for idx, q_text in enumerate(distinct_queries):
            q_hits: list[tuple[Document, float]] = []
            q_docs = docs_per_q[idx] if idx < len(docs_per_q) else []
            q_metas = metas_per_q[idx] if idx < len(metas_per_q) else []
            q_dists = dists_per_q[idx] if idx < len(dists_per_q) else []

            for d_idx, doc_content in enumerate(q_docs):
                meta = q_metas[d_idx] if d_idx < len(q_metas) and isinstance(q_metas[d_idx], dict) else {}
                dist = float(q_dists[d_idx]) if d_idx < len(q_dists) else 0.0
                if not is_quarantined_fallback(meta.get("source", "")):
                    doc = Document(page_content=str(doc_content or ""), metadata=meta)
                    q_hits.append((doc, dist))
            results_by_query[q_text] = q_hits
    else:
        for idx, q_text in enumerate(distinct_queries):
            vec = vecs[idx]
            raw_hits = list(store.similarity_search_by_vector_with_relevance_scores(vec, k=k))
            results_by_query[q_text] = [
                (doc, float(score))
                for doc, score in raw_hits
                if not is_quarantined_fallback(getattr(doc, "metadata", {}).get("source", ""))
            ]
        chroma_duration = time.perf_counter() - t_c0

    return results_by_query, embedding_duration, chroma_duration



def keyword_retrieve(query: str, institution_id: str | None = None) -> list[Document]:
    """Retrieve documents using BM25 keyword scoring."""
    question = str(query or "").strip()
    if not question:
        return []
    return list(get_bm25_retriever(institution_id).invoke(question))


def keyword_retrieve_with_scores(
    query: str,
    institution_id: str | None = None,
    k: int = RETRIEVER_K,
) -> list[tuple[Document, float]]:
    """Retrieve documents using BM25 keyword scoring along with exact scores."""
    question = str(query or "").strip()
    if not question:
        return []
    retriever = get_bm25_retriever(institution_id)
    tokens = retriever.preprocess_func(question)
    scores = retriever.vectorizer.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        (retriever.docs[i], float(scores[i]))
        for i in top_indices
        if scores[i] > 0
    ]


def corpus_document_count(institution_id: str | None = None) -> int:
    """Return the number of canonical chunks available to retrieval."""
    return len(get_canonical_chunks(institution_id))


def corpus_token_document_frequency(institution_id: str | None = None) -> dict[str, int]:
    """Return corpus-derived document frequencies for diagnostics."""
    frequency: defaultdict[str, int] = defaultdict(int)
    for document in get_canonical_chunks(institution_id):
        for token in _tokenize(document.page_content):
            frequency[token] += 1
    return dict(frequency)


def corpus_term_idf(term: str, institution_id: str | None = None) -> float:
    """Return a smoothed IDF estimate from the active canonical corpus."""
    token = str(term or "").casefold().strip()
    if not token:
        return 0.0

    document_count = max(corpus_document_count(institution_id), 1)
    document_frequency = corpus_token_document_frequency(institution_id).get(token, 0)
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
    "dense_retrieve_with_scores",
    "dense_retrieve_batch_with_scores",
    "keyword_retrieve",
    "keyword_retrieve_with_scores",
    "corpus_document_count",
    "corpus_token_document_frequency",
    "corpus_term_idf",
]
