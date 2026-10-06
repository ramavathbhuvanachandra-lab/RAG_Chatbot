"""
Retrieval V2 — Retrieval Backends

The retrieval core works through a small backend interface.

This keeps the reusable retrieval architecture independent from
Chroma/Supabase/Pinecone/etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .models import QuerySpec, RetrievedDocument


class DenseRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        k: int,
        filters: dict[str, object] | None = None,
    ) -> Sequence[RetrievedDocument]:
        ...


class LexicalRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        k: int,
        filters: dict[str, object] | None = None,
    ) -> Sequence[RetrievedDocument]:
        ...


@dataclass(frozen=True)
class RetrievalBatch:
    """
    Raw outputs from the independent retrieval channels.
    """

    dense: tuple[RetrievedDocument, ...] = ()
    lexical: tuple[RetrievedDocument, ...] = ()


def build_scope_filter(
    query: QuerySpec,
) -> dict[str, object]:
    """
    Build a generic metadata filter.

    The key is deliberately generic:
    `institution_id`.

    No college name is embedded in this function.
    """

    if not query.institution_id:
        return {}

    return {
        "institution_id": query.institution_id,
    }


def retrieve(
    query: QuerySpec,
    *,
    dense: DenseRetriever,
    lexical: LexicalRetriever,
    dense_k: int = 20,
    lexical_k: int = 20,
) -> RetrievalBatch:
    """
    Run independent dense and lexical retrieval.

    No verification happens here.
    No candidate is declared relevant here.
    """

    scope_filter = build_scope_filter(query)

    dense_results = dense.search(
        query.raw_query,
        k=dense_k,
        filters=scope_filter or None,
    )

    lexical_results = lexical.search(
        query.raw_query,
        k=lexical_k,
        filters=scope_filter or None,
    )

    return RetrievalBatch(
        dense=tuple(dense_results),
        lexical=tuple(lexical_results),
    )