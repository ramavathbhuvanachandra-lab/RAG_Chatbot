"""
Reusable hybrid retrieval executor.

Purpose
-------
Execute a RetrievalQueryPlan through injected Dense and BM25 retrieval
functions.

Responsibilities
----------------
- preserve the original user question as retrieval query #1
- execute bounded retrieval variants
- apply controlled query decay
- preserve query/channel provenance
- return ranked result lists suitable for weighted RRF

Non-responsibilities
--------------------
- semantic interpretation
- institution-specific vocabulary
- reranking
- evidence selection
- answer generation
- duplicate removal

The retrievers are injected as callables so this core module does not depend
on a specific vector database, LangChain retriever, institution, or deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ai_platform.core.retrieval.query import (
    MAX_RETRIEVAL_QUERIES,
    RetrievalQueryPlan,
)


# ============================================================
# Configuration
# ============================================================


DEFAULT_QUERY_DECAYS = (
    1.00,
    0.72,
    0.50,
)


DENSE_WEIGHT = 0.70
BM25_WEIGHT = 0.30


# ============================================================
# Result contract
# ============================================================


@dataclass(frozen=True, slots=True)
class HybridRetrievalResult:
    """
    Output of bounded Dense + BM25 retrieval.

    `ranked_lists` contains alternating retrieval channels:

        query 1 dense
        query 1 bm25
        query 2 dense
        query 2 bm25
        ...

    `weights` contains the corresponding weighted-RRF contribution
    multipliers in exactly the same order.

    `queries[0]` is always the original user query.
    """

    original_query: str

    queries: tuple[str, ...]

    ranked_lists: tuple[tuple[Any, ...], ...]

    weights: tuple[float, ...]

    decays: tuple[float, ...]

    channel_names: tuple[str, ...]

    def __post_init__(self) -> None:

        if not self.original_query.strip():
            raise ValueError(
                "original_query cannot be empty."
            )

        if not self.queries:
            raise ValueError(
                "queries cannot be empty."
            )

        if (
            self.queries[0].casefold()
            != self.original_query.casefold()
        ):
            raise ValueError(
                "queries[0] must be the original query."
            )

        if len(
            self.ranked_lists
        ) != len(
            self.weights
        ):
            raise ValueError(
                "ranked_lists and weights must have equal length."
            )

        if len(
            self.ranked_lists
        ) != len(
            self.channel_names
        ):
            raise ValueError(
                "ranked_lists and channel_names must have equal length."
            )

        if len(
            self.queries
        ) != len(
            self.decays
        ):
            raise ValueError(
                "queries and decays must have equal length."
            )

        for weight in self.weights:

            if weight < 0.0:
                raise ValueError(
                    "retrieval weights cannot be negative."
                )

        for decay in self.decays:

            if decay < 0.0:
                raise ValueError(
                    "query decay cannot be negative."
                )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "queries": list(
                self.queries
            ),
            "weights": list(
                self.weights
            ),
            "decays": list(
                self.decays
            ),
            "channel_names": list(
                self.channel_names
            ),
            "ranked_list_sizes": [
                len(
                    ranked_list
                )
                for ranked_list in self.ranked_lists
            ],
        }


# ============================================================
# Validation
# ============================================================


def _validate_retriever(
    retriever: Callable[[str], Any],
    name: str,
) -> None:

    if not callable(
        retriever
    ):
        raise TypeError(
            f"{name} must be callable."
        )


def _validate_decays(
    query_count: int,
    decays: tuple[float, ...] | None,
) -> tuple[float, ...]:
    """
    Return bounded per-query decay values.

    When fewer queries are used than the default schedule, only the required
    prefix is used.

    When a caller supplies custom decays, the number must match the actual
    query count.
    """

    if decays is None:

        return tuple(
            DEFAULT_QUERY_DECAYS[
                index
            ]
            if index
            < len(
                DEFAULT_QUERY_DECAYS
            )
            else (
                DEFAULT_QUERY_DECAYS[-1]
                ** (
                    index
                    - len(
                        DEFAULT_QUERY_DECAYS
                    )
                    + 1
                )
            )
            for index in range(
                query_count
            )
        )

    if len(
        decays
    ) != query_count:
        raise ValueError(
            "decays must contain exactly one value per query."
        )

    normalized: list[float] = []

    for decay in decays:

        numeric = float(
            decay
        )

        if numeric < 0.0:
            raise ValueError(
                "query decay cannot be negative."
            )

        normalized.append(
            numeric
        )

    return tuple(
        normalized
    )


# ============================================================
# Execution
# ============================================================


def execute_hybrid_retrieval(
    plan: RetrievalQueryPlan,
    *,
    dense_retrieve: Callable[[str], Any],
    bm25_retrieve: Callable[[str], Any],
    decays: tuple[float, ...] | None = None,
    dense_weight: float = DENSE_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
) -> HybridRetrievalResult:
    """
    Execute the bounded retrieval plan.

    Query #1:
        original user question

    Query #2+:
        controlled recovery queries from the plan

    Each query is executed through:
        Dense
        BM25

    The base Dense/BM25 weights are then multiplied by the query decay.

    Example:

        query 1:
            Dense = 0.70
            BM25  = 0.30

        query 2:
            Dense = 0.504
            BM25  = 0.216

        query 3:
            Dense = 0.35
            BM25  = 0.15

    The executor does not perform RRF itself.
    """

    if not isinstance(
        plan,
        RetrievalQueryPlan,
    ):
        raise TypeError(
            "plan must be a RetrievalQueryPlan."
        )

    _validate_retriever(
        dense_retrieve,
        "dense_retrieve",
    )

    _validate_retriever(
        bm25_retrieve,
        "bm25_retrieve",
    )

    dense_weight = float(
        dense_weight
    )

    bm25_weight = float(
        bm25_weight
    )

    if dense_weight < 0.0:
        raise ValueError(
            "dense_weight cannot be negative."
        )

    if bm25_weight < 0.0:
        raise ValueError(
            "bm25_weight cannot be negative."
        )

    queries = tuple(
        plan.queries[
            :MAX_RETRIEVAL_QUERIES
        ]
    )

    if not queries:
        raise ValueError(
            "Retrieval plan contains no queries."
        )

    if queries[0].casefold() != (
        plan.original_query.casefold()
    ):
        raise ValueError(
            "The first retrieval query must be the original question."
        )

    resolved_decays = _validate_decays(
        len(
            queries
        ),
        decays,
    )

    ranked_lists: list[
        tuple[Any, ...]
    ] = []

    weights: list[float] = []

    channel_names: list[str] = []

    for index, query in enumerate(
        queries
    ):

        decay = resolved_decays[
            index
        ]

        dense_result = dense_retrieve(
            query
        )

        bm25_result = bm25_retrieve(
            query
        )

        if dense_result is None:
            dense_result = ()

        if bm25_result is None:
            bm25_result = ()

        dense_list = tuple(
            dense_result
        )

        bm25_list = tuple(
            bm25_result
        )

        ranked_lists.extend(
            (
                dense_list,
                bm25_list,
            )
        )

        weights.extend(
            (
                dense_weight
                * decay,
                bm25_weight
                * decay,
            )
        )

        channel_names.extend(
            (
                f"dense_query_{index + 1}",
                f"bm25_query_{index + 1}",
            )
        )

    return HybridRetrievalResult(
        original_query=plan.original_query,
        queries=queries,
        ranked_lists=tuple(
            ranked_lists
        ),
        weights=tuple(
            weights
        ),
        decays=resolved_decays,
        channel_names=tuple(
            channel_names
        ),
    )


# ============================================================
# Convenience helpers
# ============================================================


def retrieval_pairs(
    result: HybridRetrievalResult,
) -> tuple[
    tuple[Any, ...],
    ...,
]:
    """
    Return the alternating retrieval lists.

    Kept as a tiny compatibility/helper API for callers that only need
    the ranked lists.
    """

    return result.ranked_lists


def retrieval_weights(
    result: HybridRetrievalResult,
) -> tuple[float, ...]:
    """
    Return the exact weighted-RRF multipliers.
    """

    return result.weights


# ============================================================
# Public API
# ============================================================


__all__ = [
    "DEFAULT_QUERY_DECAYS",
    "DENSE_WEIGHT",
    "BM25_WEIGHT",
    "HybridRetrievalResult",
    "execute_hybrid_retrieval",
    "retrieval_pairs",
    "retrieval_weights",
]