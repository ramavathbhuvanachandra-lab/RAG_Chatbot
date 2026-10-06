"""
Retrieval V2 — Ranking

Combines independent retrieval channels using Reciprocal Rank Fusion.

Ranking is ranking.
It is not verification.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .models import RankedCandidate, RetrievedDocument
from .retrieve import RetrievalBatch


DEFAULT_RRF_K = 60


def reciprocal_rank(rank: int, *, k: int = DEFAULT_RRF_K) -> float:
    if rank < 1:
        raise ValueError("rank must be >= 1")

    return 1.0 / (k + rank)


def fuse(
    batch: RetrievalBatch,
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> tuple[RankedCandidate, ...]:
    """
    Fuse dense and lexical rankings.

    Documents appearing in both channels naturally receive
    stronger scores.
    """

    scores: defaultdict[str, float] = defaultdict(float)
    documents: dict[str, RetrievedDocument] = {}

    dense_rank: dict[str, int] = {}
    lexical_rank: dict[str, int] = {}

    for rank, document in enumerate(batch.dense, start=1):
        documents[document.document_id] = document
        dense_rank[document.document_id] = rank
        scores[document.document_id] += reciprocal_rank(
            rank,
            k=rrf_k,
        )

    for rank, document in enumerate(batch.lexical, start=1):
        documents[document.document_id] = document
        lexical_rank[document.document_id] = rank
        scores[document.document_id] += reciprocal_rank(
            rank,
            k=rrf_k,
        )

    ranked = []

    for document_id, score in scores.items():
        document = documents[document_id]

        ranked.append(
            RankedCandidate(
                document=document,
                rrf_score=score,
                semantic_score=document.dense_score,
                lexical_score=document.lexical_score,
            )
        )

    ranked.sort(
        key=lambda candidate: candidate.rrf_score,
        reverse=True,
    )

    return tuple(ranked)


def deduplicate(
    candidates: Iterable[RankedCandidate],
) -> tuple[RankedCandidate, ...]:
    """
    Remove exact duplicate document IDs.

    More sophisticated near-duplicate handling belongs here later,
    not inside verification.
    """

    seen: set[str] = set()
    result: list[RankedCandidate] = []

    for candidate in candidates:
        document_id = candidate.document.document_id

        if document_id in seen:
            continue

        seen.add(document_id)
        result.append(candidate)

    return tuple(result)