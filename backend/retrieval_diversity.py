"""
IIT Jodhpur V1 — Phase 6 Retrieval Diversity

Purpose
-------
Select a better final evidence candidate set after the existing relevance
scoring stage.

This module does NOT perform retrieval and does NOT call an LLM.

Pipeline position
-----------------

    Dense + BM25
          ↓
         RRF
          ↓
      deduplication
          ↓
    existing relevance scoring
          ↓
    Phase-6 diversity selection
          ↓
       final Top-K

Design goals
------------
- Relevance remains the primary signal.
- Redundant candidates should not consume the entire evidence budget.
- Legitimately different same-source documents must remain available.
- Diversity is a tie-break / soft preference, not a hard source round-robin.
- No dependency on backend.retriever.
- Fully deterministic.
"""

from __future__ import annotations

import re
from typing import Any, Sequence


# =========================================================
# Configuration
# =========================================================

# Keep these values aligned with the current Phase-6 retrieval contract.
# They deliberately live here to keep this module dependency-free.
DEFAULT_MAX_DOCUMENTS_PER_SOURCE = 2

# The existing retriever uses 0.92 for near-duplicate removal.
# Phase 6 uses a lower threshold to detect "high redundancy" without
# declaring the documents actual duplicates.
CONTENT_REDUNDANCY_THRESHOLD = 0.70

# A lower-scoring candidate may be preferred only when it remains
# reasonably competitive with the current candidate.
RELEVANCE_TOLERANCE = 0.85


# =========================================================
# Text normalization
# =========================================================

def _normalize_text(
    text: str,
) -> str:
    """
    Normalize text for deterministic redundancy comparison.

    This intentionally mirrors the retrieval normalization contract
    without importing backend.retriever, avoiding circular imports.
    """
    text = str(
        text or ""
    ).lower()

    replacements = {
        "b.tech.": "btech",
        "b.tech": "btech",
        "m.tech.": "mtech",
        "m.tech": "mtech",
        "m.sc.": "msc",
        "m.sc": "msc",
        "ph.d.": "phd",
        "ph.d": "phd",
        "_": " ",
        "-": " ",
        "/": " ",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    text = re.sub(
        r"[^a-z0-9\s/&+]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _tokens(
    text: str,
) -> set[str]:
    """
    Return normalized content tokens.
    """
    return {
        token
        for token in _normalize_text(
            text
        ).split()
        if len(token) > 2
    }


def _jaccard_similarity(
    first_text: str,
    second_text: str,
) -> float:
    """
    Compute token-set Jaccard similarity.
    """
    first_tokens = _tokens(
        first_text
    )

    second_tokens = _tokens(
        second_text
    )

    if not first_tokens or not second_tokens:
        return 0.0

    intersection = (
        first_tokens
        & second_tokens
    )

    union = (
        first_tokens
        | second_tokens
    )

    if not union:
        return 0.0

    return (
        len(intersection)
        / len(union)
    )


# =========================================================
# Candidate helpers
# =========================================================

def _document(
    candidate: dict[str, Any],
):
    """
    Return the LangChain document carried by a scored candidate.
    """
    return candidate.get(
        "document"
    )


def _document_text(
    candidate: dict[str, Any],
) -> str:
    """
    Return candidate document text.
    """
    document = _document(
        candidate
    )

    if document is None:
        return ""

    return str(
        getattr(
            document,
            "page_content",
            "",
        )
        or ""
    )


def _source(
    candidate: dict[str, Any],
) -> str:
    """
    Return the candidate source metadata already computed by the
    existing reranker.
    """
    return str(
        candidate.get(
            "source",
            "",
        )
        or ""
    )


def _score(
    candidate: dict[str, Any],
) -> float:
    """
    Read the existing relevance score.
    """
    try:
        return float(
            candidate.get(
                "score",
                0.0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def _original_rank(
    candidate: dict[str, Any],
) -> int:
    """
    Read the original ranking position.
    """
    try:
        return int(
            candidate.get(
                "original_rank",
                10**9,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return 10**9


# =========================================================
# Similarity / source analysis
# =========================================================

def _maximum_selected_similarity(
    candidate: dict[str, Any],
    selected: Sequence[
        dict[str, Any]
    ],
) -> float:
    """
    Return the candidate's strongest similarity to selected evidence.
    """
    if not selected:
        return 0.0

    candidate_text = _document_text(
        candidate
    )

    return max(
        (
            _jaccard_similarity(
                candidate_text,
                _document_text(
                    other
                ),
            )
            for other in selected
        ),
        default=0.0,
    )


def _selected_source_count(
    candidate: dict[str, Any],
    selected: Sequence[
        dict[str, Any]
    ],
) -> int:
    """
    Count already-selected candidates from this source.
    """
    source = _source(
        candidate
    )

    if not source:
        return 0

    return sum(
        1
        for other in selected
        if _source(
            other
        ) == source
    )


def _has_competitive_nonredundant_alternative(
    candidate: dict[str, Any],
    remaining: Sequence[
        dict[str, Any]
    ],
    selected: Sequence[
        dict[str, Any]
    ],
) -> bool:
    """
    Determine whether another candidate can safely provide more diversity.

    An alternative must:
        - be different from the current candidate
        - have a reasonably competitive relevance score
        - not already be redundant with selected evidence
    """
    candidate_score = _score(
        candidate
    )

    for alternative in remaining:

        if alternative is candidate:
            continue

        alternative_score = _score(
            alternative
        )

        if alternative_score <= 0:
            continue

        if alternative_score < (
            candidate_score
            * RELEVANCE_TOLERANCE
        ):
            continue

        similarity = (
            _maximum_selected_similarity(
                alternative,
                selected,
            )
        )

        if similarity >= (
            CONTENT_REDUNDANCY_THRESHOLD
        ):
            continue

        return True

    return False


# =========================================================
# Candidate selection
# =========================================================

def select_diverse_documents(
    scored_documents: Sequence[
        dict[str, Any]
    ],
    top_k: int,
    *,
    max_documents_per_source: int = (
        DEFAULT_MAX_DOCUMENTS_PER_SOURCE
    ),
) -> list[Any]:
    """
    Select a final evidence set from already-scored candidates.

    Important policy
    ----------------
    Relevance is never sacrificed merely to create diversity.

    Diversity can influence selection only when:
        - candidates are reasonably competitive, and
        - the current candidate is already redundant/saturated.

    Same-source documents therefore remain completely valid when they
    contain distinct information and no sufficiently competitive
    alternative exists.
    """
    if top_k <= 0:
        return []

    candidates = [
        candidate
        for candidate in scored_documents
        if _document(
            candidate
        ) is not None
    ]

    if not candidates:
        return []

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -_score(
                candidate
            ),
            _original_rank(
                candidate
            ),
        ),
    )

    selected: list[
        dict[str, Any]
    ] = []

    remaining = list(
        ordered
    )

    while (
        remaining
        and
        len(selected) < top_k
    ):

        # -----------------------------------------------------
        # First candidate is always the strongest candidate.
        # -----------------------------------------------------
        if not selected:
            selected.append(
                remaining.pop(0)
            )
            continue

        best_index = 0
        best_key = None

        for index, candidate in enumerate(
            remaining
        ):

            score = _score(
                candidate
            )

            source_count = (
                _selected_source_count(
                    candidate,
                    selected,
                )
            )

            similarity = (
                _maximum_selected_similarity(
                    candidate,
                    selected,
                )
            )

            source_saturated = (
                bool(
                    _source(
                        candidate
                    )
                )
                and
                source_count
                >= max_documents_per_source
            )

            has_alternative = (
                _has_competitive_nonredundant_alternative(
                    candidate,
                    remaining,
                    selected,
                )
            )

            # -------------------------------------------------
            # Diversity flags are preferences, NOT hard filters.
            # -------------------------------------------------

            source_penalty = (
                1
                if (
                    source_saturated
                    and has_alternative
                )
                else 0
            )

            redundancy_penalty = (
                1
                if (
                    similarity
                    >= CONTENT_REDUNDANCY_THRESHOLD
                    and has_alternative
                )
                else 0
            )

            # -------------------------------------------------
            # Relevance bucket.
            #
            # Candidates within the tolerance window compete on
            # diversity. A clearly stronger candidate stays ahead.
            # -------------------------------------------------

            strongest_score = max(
                (
                    _score(
                        item
                    )
                    for item in remaining
                ),
                default=score,
            )

            score_gap = (
                strongest_score
                - score
            )

            candidate_is_competitive = (
                score >= (
                    strongest_score
                    * RELEVANCE_TOLERANCE
                )
            )

            # -------------------------------------------------
            # Diversity is only activated for competitive candidates.
            # -------------------------------------------------

            diversity_rank = (
                0
                if (
                    candidate_is_competitive
                    and
                    source_penalty == 0
                    and
                    redundancy_penalty == 0
                )
                else
                (
                    source_penalty
                    + redundancy_penalty
                )
            )

            key = (
                diversity_rank,
                -score,
                -(
                    1.0
                    - similarity
                ),
                _original_rank(
                    candidate
                ),
                index,
            )

            # A candidate that is clearly stronger must never lose to
            # diversity merely because another candidate comes from a
            # different source.
            if not candidate_is_competitive:
                key = (
                    0,
                    -score,
                    -(
                        1.0
                        - similarity
                    ),
                    _original_rank(
                        candidate
                    ),
                    index,
                )

            if (
                best_key is None
                or
                key < best_key
            ):
                best_key = key
                best_index = index

        selected.append(
            remaining.pop(
                best_index
            )
        )

    return [
        candidate["document"]
        for candidate in selected
    ]