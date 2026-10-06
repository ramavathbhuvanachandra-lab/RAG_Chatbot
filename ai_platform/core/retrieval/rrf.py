"""
Generic weighted Reciprocal Rank Fusion (RRF).

Purpose
-------
Fuse ranked retrieval outputs into immutable RetrievalCandidate objects while
preserving retrieval provenance for downstream auditing and reranking.

Core invariants
---------------
- RRF is the fusion layer only; it does not perform semantic reranking.
- Default two-channel fusion uses Dense=0.7 and BM25=0.3, matching the
  established retrieval behavior.
- For every contribution, score = weight / (rrf_k + rank).
- The first occurrence of a document within a single ranked list contributes;
  repeated occurrences in that same list are ignored to prevent accidental
  duplicate-score inflation.
- Existing RetrievalCandidate state is retained where possible, but fused RRF
  provenance becomes the authoritative retrieval provenance for this stage.
- No institution-specific vocabulary, paths, models, or runtime services are
  imported here.
"""

from __future__ import annotations

from dataclasses import replace
import math
from collections.abc import Iterable, Sequence
from typing import Any

from ai_platform.core.retrieval.contracts import (
    RetrievalCandidate,
    RetrievalProvenance,
    RetrievalSignal,
)


RRF_K = 60
DEFAULT_TWO_CHANNEL_WEIGHTS = (0.70, 0.30)
DEFAULT_CHANNEL_NAMES = ("dense", "bm25")


class RRFInputError(ValueError):
    """Raised when ranked-list inputs violate the RRF contract."""


def _validate_rrf_k(value: int | float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RRFInputError("rrf_k must be a positive finite number.")

    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise RRFInputError("rrf_k must be a positive finite number.")

    return numeric


def _validate_weights(
    ranked_lists: Sequence[Sequence[Any]],
    weights: Sequence[float] | None,
) -> tuple[float, ...]:
    if weights is None:
        if len(ranked_lists) == 2:
            return DEFAULT_TWO_CHANNEL_WEIGHTS
        return tuple(1.0 for _ in ranked_lists)

    values = tuple(weights)
    if len(values) != len(ranked_lists):
        raise RRFInputError(
            "Number of RRF weights must match number of ranked lists."
        )

    normalized: list[float] = []
    for index, weight in enumerate(values):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise RRFInputError(
                f"RRF weight at index {index} must be a non-negative finite number."
            )
        numeric = float(weight)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise RRFInputError(
                f"RRF weight at index {index} must be a non-negative finite number."
            )
        normalized.append(numeric)

    return tuple(normalized)


def _channel_names(
    count: int,
    channels: Sequence[str] | None,
) -> tuple[str, ...]:
    if channels is not None:
        values = tuple(str(value).strip() for value in channels)
        if len(values) != count:
            raise RRFInputError(
                "Number of channel names must match number of ranked lists."
            )
        if any(not value for value in values):
            raise RRFInputError("RRF channel names cannot be empty.")
        return values

    if count == 2:
        return DEFAULT_CHANNEL_NAMES

    return tuple(f"channel_{index + 1}" for index in range(count))


def _coerce_candidate(item: Any) -> RetrievalCandidate:
    if isinstance(item, RetrievalCandidate):
        return item

    return RetrievalCandidate.from_document(item)


def _candidate_signal(
    candidate: RetrievalCandidate,
    *,
    channel: str,
    rank: int,
    weight: float,
    query: str | None,
) -> RetrievalSignal:
    """Create a canonical signal without assuming a document type."""
    score: float | None = None

    # Preserve a score already attached to the candidate for the matching
    # conventional channel where available. We do not infer scores from
    # arbitrary document metadata.
    provenance = candidate.provenance
    if channel.casefold() == "dense":
        score = provenance.dense_score
    elif channel.casefold() == "bm25":
        score = provenance.bm25_score

    return RetrievalSignal(
        channel=channel,
        rank=rank,
        score=score,
        weight=weight,
        query=query,
    )


def _merge_provenance(
    candidate: RetrievalCandidate,
    *,
    rrf_score: float,
    signals: tuple[RetrievalSignal, ...],
    retrieval_queries: tuple[str, ...],
    primary_query: str,
    alternate_score_contribution: float = 0.0,
) -> RetrievalProvenance:
    """Build the stage-authoritative provenance snapshot."""

    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None

    for signal in signals:
        if signal.channel.casefold() == "dense":
            if dense_rank is None:
                dense_rank = signal.rank
                dense_score = signal.score
        elif signal.channel.casefold() == "bm25":
            if bm25_rank is None:
                bm25_rank = signal.rank
                bm25_score = signal.score

    prior = candidate.provenance
    prior_signals = tuple(prior.signals)
    combined_signals = prior_signals + signals

    effective_primary_query = (
        primary_query or prior.primary_query
    )
    effective_retrieval_queries = (
        retrieval_queries or prior.retrieval_queries
    )
    effective_alternate_contribution = (
        max(0.0, prior.alternate_score_contribution)
        + max(0.0, alternate_score_contribution)
    )

    # Prefer the newly fused dense/BM25 observations, but preserve prior
    # channel provenance when this candidate already had it.
    if dense_rank is None:
        dense_rank = prior.dense_rank
        dense_score = prior.dense_score
    if bm25_rank is None:
        bm25_rank = prior.bm25_rank
        bm25_score = prior.bm25_score

    return RetrievalProvenance(
        dense_rank=dense_rank,
        dense_score=dense_score,
        bm25_rank=bm25_rank,
        bm25_score=bm25_score,
        rrf_score=rrf_score,
        alternate_score_contribution=effective_alternate_contribution,
        primary_query=effective_primary_query,
        retrieval_queries=effective_retrieval_queries,
        signals=combined_signals,
    )


def fuse_ranked_lists(
    ranked_lists: Sequence[Sequence[Any]],
    *,
    weights: Sequence[float] | None = None,
    rrf_k: int | float = RRF_K,
    channels: Sequence[str] | None = None,
    primary_query: str = "",
    retrieval_queries: Sequence[str] | None = None,
) -> tuple[RetrievalCandidate, ...]:
    """
    Fuse ranked retrieval lists into RetrievalCandidate objects.

    Parameters
    ----------
    ranked_lists:
        Ranked lists in descending retrieval order. Items may be plain
        LangChain-style documents or existing RetrievalCandidate objects.
    weights:
        Per-list RRF weights. With exactly two lists the default is
        (0.70, 0.30). With any other list count the default is 1.0 per list.
    rrf_k:
        Positive finite RRF constant; the historical default is 60.
    channels:
        Optional human-readable channel names. Two-list fusion defaults to
        ("dense", "bm25"). Other counts receive generic channel names.
    primary_query:
        Optional authoritative user query retained for provenance.
    retrieval_queries:
        Optional bounded query set retained for provenance.

    Returns
    -------
    tuple[RetrievalCandidate, ...]
        Candidates sorted by descending RRF score, with first-seen order as
        the deterministic tie-breaker.
    """

    # Convert each iterable once so generators are supported and the shape is
    # validated deterministically.
    lists = tuple(tuple(items) for items in ranked_lists)
    k = _validate_rrf_k(rrf_k)
    normalized_weights = _validate_weights(lists, weights)
    normalized_channels = _channel_names(len(lists), channels)

    primary = str(primary_query or "").strip()
    queries = tuple(
        str(query).strip()
        for query in (retrieval_queries or ())
        if str(query).strip()
    )

    scores: dict[str, float] = {}
    candidates: dict[str, RetrievalCandidate] = {}
    signals: dict[str, list[RetrievalSignal]] = {}
    first_seen: dict[str, int] = {}
    ordinal = 0

    for list_index, ranked_items in enumerate(lists):
        weight = normalized_weights[list_index]
        channel = normalized_channels[list_index]
        seen_in_list: set[str] = set()

        for rank, item in enumerate(ranked_items, start=1):
            candidate = _coerce_candidate(item)
            document_id = candidate.document_id

            if document_id in seen_in_list:
                continue
            seen_in_list.add(document_id)

            if document_id not in candidates:
                candidates[document_id] = candidate
                signals[document_id] = []
                first_seen[document_id] = ordinal
                ordinal += 1

            contribution = weight / (k + rank)
            scores[document_id] = scores.get(document_id, 0.0) + contribution
            signals[document_id].append(
                _candidate_signal(
                    candidate,
                    channel=channel,
                    rank=rank,
                    weight=weight,
                    query=(queries[list_index] if list_index < len(queries) else None),
                )
            )

    ordered_ids = sorted(
        scores,
        key=lambda document_id: (
            -scores[document_id],
            first_seen[document_id],
        ),
    )

    output: list[RetrievalCandidate] = []
    for document_id in ordered_ids:
        candidate = candidates[document_id]
        provenance = _merge_provenance(
            candidate,
            rrf_score=scores[document_id],
            signals=tuple(signals[document_id]),
            retrieval_queries=queries,
            primary_query=primary,
        )
        output.append(replace(candidate, provenance=provenance, final_score=0.0))

    return tuple(output)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Any]],
    *,
    weights: Sequence[float] | None = None,
    k: int | float = RRF_K,
    channels: Sequence[str] | None = None,
    primary_query: str = "",
    retrieval_queries: Sequence[str] | None = None,
) -> list[Any]:
    """
    Backward-compatible wrapper returning plain documents.

    New retrieval code should prefer `fuse_ranked_lists()` so provenance is
    retained for reranking, evidence selection, and diagnostics.
    """

    candidates = fuse_ranked_lists(
        ranked_lists,
        weights=weights,
        rrf_k=k,
        channels=channels,
        primary_query=primary_query,
        retrieval_queries=retrieval_queries,
    )
    return [candidate.document for candidate in candidates]


__all__ = [
    "RRF_K",
    "DEFAULT_TWO_CHANNEL_WEIGHTS",
    "DEFAULT_CHANNEL_NAMES",
    "RRFInputError",
    "fuse_ranked_lists",
    "reciprocal_rank_fusion",
]
