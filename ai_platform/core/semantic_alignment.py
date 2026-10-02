"""
Generic semantic alignment for retrieval candidates.

Responsibilities
----------------
Compare a user query with a candidate document using the active semantic
registry as a supporting signal.

The semantic registry is NOT the primary source of understanding.

The broader retrieval system still relies on:
    - dense retrieval
    - lexical retrieval
    - query understanding
    - RRF
    - evidence quality
    - final reranking

This module only produces bounded semantic-alignment signals that other
stages may use as supporting evidence.

No institution-specific vocabulary is defined here.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from ai_platform.core.retrieval_contracts import (
    CandidateAlignment,
    DocumentMeaning,
)
from ai_platform.core.semantic_registry import (
    SemanticRegistry,
)


# ============================================================
# Helpers
# ============================================================


def _safe_ratio(
    matched: int,
    total: int,
) -> float:
    """
    Return a bounded overlap ratio.

    A category with no query-side signals contributes 0.0 and is ignored
    when calculating the aggregate semantic match.
    """

    if total <= 0:
        return 0.0

    return min(
        1.0,
        max(
            0.0,
            matched / total,
        ),
    )


def _normalize_values(values: Any) -> tuple[str, ...]:
    """Normalize optional registry output into deterministic string labels."""
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    try:
        items = tuple(values)
    except TypeError:
        return ()

    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = " ".join(item.strip().split())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return tuple(sorted(result, key=str.casefold))


def _set_overlap(
    query_values: set[str],
    document_values: set[str],
) -> float:
    """Measure directional query-side semantic coverage."""
    if not query_values:
        return 0.0

    query_normalized = {value.casefold() for value in query_values}
    document_normalized = {value.casefold() for value in document_values}

    return _safe_ratio(
        len(query_normalized & document_normalized),
        len(query_normalized),
    )


def _registry_detect(
    registry: SemanticRegistry,
    method_name: str,
    text: str,
) -> tuple[str, ...]:
    """Run an optional registry detector without breaking retrieval on failure."""
    try:
        detector = getattr(registry, method_name)
        return _normalize_values(detector(text))
    except Exception:
        return ()


def _combined_coverage(
    pairs: tuple[
        tuple[set[str], set[str]],
        ...,
    ],
) -> float:
    """
    Calculate combined query-side semantic coverage.
    """

    total_query_signals = sum(
        len(query_values)
        for query_values, _ in pairs
    )

    if total_query_signals <= 0:
        return 0.0

    total_matches = sum(
        len(
            query_values
            & document_values
        )
        for query_values, document_values in pairs
    )

    return _safe_ratio(
        total_matches,
        total_query_signals,
    )


# ============================================================
# Meaning extraction
# ============================================================


def build_meaning(
    text: str,
    registry: SemanticRegistry,
) -> DocumentMeaning:
    """
    Build a semantic representation using the supplied registry.

    Registry detection is optional supporting information.

    An empty detection result is valid and must never prevent the broader
    retrieval pipeline from continuing.
    """

    if not isinstance(
        registry,
        SemanticRegistry,
    ):
        raise TypeError(
            "registry must be a SemanticRegistry."
        )

    programs = _registry_detect(
        registry,
        "detect_programs",
        text,
    )

    entities = _registry_detect(
        registry,
        "detect_entities",
        text,
    )

    topics = _registry_detect(
        registry,
        "detect_topics",
        text,
    )

    return DocumentMeaning(
        programs=programs,
        entities=entities,
        topics=topics,
        attributes=(),
        intent=None,
        scope=(),
        qualifiers=(),
        constraints=(),
    )


def build_query_meaning(
    query: str,
    registry: SemanticRegistry,
) -> DocumentMeaning:
    """
    Build semantic meaning for the user query.
    """

    return build_meaning(
        query,
        registry,
    )


def build_document_meaning(
    document: Any,
    registry: SemanticRegistry,
) -> DocumentMeaning:
    """
    Build semantic meaning from a LangChain-style document.

    Source metadata is included as additional context because institutions
    may encode useful semantic labels in source metadata. This remains a
    supporting signal only.
    """

    page_content = getattr(
        document,
        "page_content",
        "",
    )

    metadata = getattr(
        document,
        "metadata",
        {},
    )

    source = ""

    if isinstance(
        metadata,
        Mapping,
    ):
        source = str(
            metadata.get(
                "source",
                "",
            )
            or ""
        )

    combined_text = (
        f"{source}\n"
        f"{page_content}"
    ).strip()

    return build_meaning(
        combined_text,
        registry,
    )


# ============================================================
# Meaning alignment
# ============================================================


def align_meanings(
    query_meaning: DocumentMeaning,
    document_meaning: DocumentMeaning,
) -> CandidateAlignment:
    """
    Compare query meaning against document meaning.

    All generated alignment signals are normalized to [0, 1].

    Additional document concepts are NOT contradictions.

    For example:

        Query:
            M.Tech registration

        Document:
            M.Tech registration + hostel + fees

    The hostel/fees concepts do not automatically make the candidate wrong.
    """

    if not isinstance(
        query_meaning,
        DocumentMeaning,
    ):
        raise TypeError(
            "query_meaning must be a DocumentMeaning."
        )

    if not isinstance(
        document_meaning,
        DocumentMeaning,
    ):
        raise TypeError(
            "document_meaning must be a DocumentMeaning."
        )

    # --------------------------------------------------------
    # Programs
    # --------------------------------------------------------

    query_programs = set(
        query_meaning.programs
    )

    document_programs = set(
        document_meaning.programs
    )

    program_match = _set_overlap(
        query_programs,
        document_programs,
    )

    # --------------------------------------------------------
    # Entities
    # --------------------------------------------------------

    query_entities = set(
        query_meaning.entities
    )

    document_entities = set(
        document_meaning.entities
    )

    entity_match = _set_overlap(
        query_entities,
        document_entities,
    )

    # --------------------------------------------------------
    # Topics
    # --------------------------------------------------------

    query_topics = set(
        query_meaning.topics
    )

    document_topics = set(
        document_meaning.topics
    )

    topic_match = _set_overlap(
        query_topics,
        document_topics,
    )

    # --------------------------------------------------------
    # Attributes
    # --------------------------------------------------------

    query_attributes = set(
        query_meaning.attributes
    )

    document_attributes = set(
        document_meaning.attributes
    )

    attribute_match = _set_overlap(
        query_attributes,
        document_attributes,
    )

    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    intent_match = 0.0

    if (
        query_meaning.intent
        and document_meaning.intent
    ):

        if (
            query_meaning.intent.casefold()
            == document_meaning.intent.casefold()
        ):
            intent_match = 1.0

    # --------------------------------------------------------
    # Scope
    # --------------------------------------------------------

    query_scope = set(
        query_meaning.scope
    )

    document_scope = set(
        document_meaning.scope
    )

    scope_match = _set_overlap(
        query_scope,
        document_scope,
    )

    # --------------------------------------------------------
    # Combined coverage
    # --------------------------------------------------------

    coverage_pairs = (
        (
            query_programs,
            document_programs,
        ),
        (
            query_entities,
            document_entities,
        ),
        (
            query_topics,
            document_topics,
        ),
        (
            query_attributes,
            document_attributes,
        ),
        (
            query_scope,
            document_scope,
        ),
    )

    coverage = _combined_coverage(
        coverage_pairs
    )

    # --------------------------------------------------------
    # Aggregate semantic match
    # --------------------------------------------------------
    #
    # Only categories that actually exist on the query side participate.
    #
    # This is important:
    #
    # If the registry does not recognize a term, semantic_match does not
    # become a negative score. It simply contributes no registry signal.
    #
    # Dense/BM25/query-understanding signals remain responsible for the
    # broader understanding.
    # --------------------------------------------------------

    weighted_scores: list[
        tuple[float, float]
    ] = []

    if query_programs:
        weighted_scores.append(
            (
                program_match,
                0.30,
            )
        )

    if query_entities:
        weighted_scores.append(
            (
                entity_match,
                0.20,
            )
        )

    if query_topics:
        weighted_scores.append(
            (
                topic_match,
                0.20,
            )
        )

    if query_attributes:
        weighted_scores.append(
            (
                attribute_match,
                0.10,
            )
        )

    if query_meaning.intent:
        weighted_scores.append(
            (
                intent_match,
                0.10,
            )
        )

    if query_scope:
        weighted_scores.append(
            (
                scope_match,
                0.10,
            )
        )

    if weighted_scores:

        total_weight = sum(
            weight
            for _, weight in weighted_scores
        )

        semantic_match = (
            sum(
                score * weight
                for score, weight
                in weighted_scores
            )
            / total_weight
        )

    else:
        semantic_match = 0.0

    return CandidateAlignment(
        program_match=program_match,
        entity_match=entity_match,
        topic_match=topic_match,
        attribute_match=attribute_match,
        intent_match=intent_match,
        scope_match=scope_match,
        coverage=coverage,
        semantic_match=semantic_match,
        conflicts=(),
    )


# ============================================================
# Query-to-document alignment
# ============================================================


def align_query_to_document(
    query: str,
    document: Any,
    registry: SemanticRegistry,
) -> tuple[
    DocumentMeaning,
    DocumentMeaning,
    CandidateAlignment,
]:
    """
    Build query meaning, document meaning, and alignment.
    """

    query_meaning = build_query_meaning(
        query,
        registry,
    )

    document_meaning = build_document_meaning(
        document,
        registry,
    )

    alignment = align_meanings(
        query_meaning,
        document_meaning,
    )

    return (
        query_meaning,
        document_meaning,
        alignment,
    )


# ============================================================
# Soft ranking contribution
# ============================================================


def semantic_alignment_bonus(
    alignment: CandidateAlignment,
    *,
    maximum: float = 8.0,
) -> float:
    """
    Convert semantic alignment into a bounded soft ranking bonus.

    The registry can improve ranking precision but cannot independently
    determine the final relevance of a document.
    """

    if not isinstance(
        alignment,
        CandidateAlignment,
    ):
        raise TypeError(
            "alignment must be a CandidateAlignment."
        )

    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
        raise TypeError(
            "maximum must be a finite non-negative number."
        )

    maximum_value = float(maximum)

    if not math.isfinite(maximum_value) or maximum_value < 0.0:
        raise ValueError(
            "maximum must be a finite non-negative number."
        )

    # Weighted combination of independent semantic dimensions.
    #
    # Program/entity/topic signals are stronger than optional future
    # attribute/intent/scope signals, but everything remains bounded.

    base = (
        0.30
        * alignment.program_match
        + 0.20
        * alignment.entity_match
        + 0.20
        * alignment.topic_match
        + 0.10
        * alignment.attribute_match
        + 0.10
        * alignment.intent_match
        + 0.10
        * alignment.scope_match
    )

    score = (
        0.75 * base
        + 0.25 * alignment.coverage
    )

    return min(
        maximum_value,
        max(
            0.0,
            float(score) * maximum_value,
        ),
    )


# ============================================================
# Public API
# ============================================================


__all__ = [
    "build_meaning",
    "build_query_meaning",
    "build_document_meaning",
    "align_meanings",
    "align_query_to_document",
    "semantic_alignment_bonus",
]