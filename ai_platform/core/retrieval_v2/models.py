"""
Retrieval V2 — Core Data Models

This module contains only the shared data structures used by the
institution-agnostic retrieval V2 pipeline.

Design principles
-----------------
- No institution-specific logic.
- No LLM calls.
- No retrieval policy.
- No verification state machine.
- Immutable models where practical.
- Metadata is preserved throughout the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# Query models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalIntent:
    """
    A single semantic intent extracted from a user query.

    Example:
        target="mtech"
        facets=("eligibility",)
        entities=()
        keywords=("degree", "qualification")
    """

    target: str | None = None
    facets: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "facets", _clean_many(self.facets))
        object.__setattr__(self, "entities", _clean_many(self.entities))
        object.__setattr__(self, "keywords", _clean_many(self.keywords))

        if self.target is not None:
            object.__setattr__(
                self,
                "target",
                _clean(self.target),
            )


@dataclass(frozen=True)
class QuerySpec:
    """
    Normalized representation of a user query.

    The retrieval core should operate on QuerySpec rather than repeatedly
    interpreting the raw query string.
    """

    raw_query: str
    intents: tuple[RetrievalIntent, ...] = ()
    institution_id: str | None = None
    language: str | None = None
    request_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_query",
            _clean(self.raw_query),
        )

        object.__setattr__(
            self,
            "intents",
            tuple(self.intents),
        )

        if self.institution_id is not None:
            object.__setattr__(
                self,
                "institution_id",
                _clean(self.institution_id),
            )

        if self.language is not None:
            object.__setattr__(
                self,
                "language",
                _clean(self.language),
            )

        if self.request_type is not None:
            object.__setattr__(
                self,
                "request_type",
                _clean(self.request_type),
            )


# ---------------------------------------------------------------------------
# Retrieval models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedDocument:
    """
    A document returned by a retrieval backend.

    Dense and lexical scores are preserved independently so later ranking
    layers can combine them without losing provenance.
    """

    document_id: str
    text: str
    source: str | None = None
    title: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    dense_score: float = 0.0
    lexical_score: float = 0.0

    dense_rank: int | None = None
    lexical_rank: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_id",
            _clean(self.document_id),
        )

        object.__setattr__(
            self,
            "text",
            _clean(self.text),
        )

        if self.source is not None:
            object.__setattr__(
                self,
                "source",
                _clean(self.source),
            )

        if self.title is not None:
            object.__setattr__(
                self,
                "title",
                _clean(self.title),
            )


@dataclass(frozen=True)
class RankedCandidate:
    """
    Candidate after retrieval fusion/ranking.

    The underlying RetrievedDocument is preserved intact.
    """

    document: RetrievedDocument

    rrf_score: float = 0.0
    semantic_score: float = 0.0
    lexical_score: float = 0.0

    matched_intents: tuple[int, ...] = ()
    duplicate_of: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "matched_intents",
            tuple(self.matched_intents),
        )


# ---------------------------------------------------------------------------
# Evidence-unit models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceUnitCandidate:
    """
    A local piece of a retrieved document.

    A document is not automatically treated as one piece of evidence.

    Retrieval V2 first converts a document into local evidence units.
    Relevance is then evaluated against these local units.

    This is the key protection against cases such as:

        "M.Tech programs are listed here. ... [many unrelated lines] ...
         Ph.D. eligibility requirements are listed separately."

    The distant Ph.D. statement should not become M.Tech evidence merely
    because both terms occur somewhere in the same document.
    """

    document_id: str
    text: str

    source: str | None = None
    title: str | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    unit_index: int = 0
    start_char: int = 0
    end_char: int = 0

    semantic_score: float = 0.0
    lexical_score: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_id",
            _clean(self.document_id),
        )

        object.__setattr__(
            self,
            "text",
            _clean(self.text),
        )

        if self.source is not None:
            object.__setattr__(
                self,
                "source",
                _clean(self.source),
            )

        if self.title is not None:
            object.__setattr__(
                self,
                "title",
                _clean(self.title),
            )


@dataclass(frozen=True)
class EvidenceScore:
    """
    Component-level evidence relevance score.

    Each component is normalized to approximately [0, 1].
    """

    target: float = 0.0
    facet: float = 0.0
    entity: float = 0.0
    semantic: float = 0.0
    lexical: float = 0.0
    scope: float = 0.0

    final: float = 0.0


@dataclass(frozen=True)
class EvidenceDecision:
    """
    Decision produced when evaluating one evidence unit.

    The decision explains why a unit was kept or rejected without introducing
    a large verification state machine.
    """

    keep: bool
    score: EvidenceScore

    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reasons",
            tuple(self.reasons),
        )


@dataclass(frozen=True)
class EvidenceUnit:
    """
    Clean evidence unit passed to the context-building stage.

    This represents evidence that has survived relevance filtering.
    """

    document_id: str
    text: str

    source: str | None = None
    title: str | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    unit_index: int = 0
    start_char: int = 0
    end_char: int = 0

    score: float = 0.0
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_id",
            _clean(self.document_id),
        )

        object.__setattr__(
            self,
            "text",
            _clean(self.text),
        )

        if self.source is not None:
            object.__setattr__(
                self,
                "source",
                _clean(self.source),
            )

        if self.title is not None:
            object.__setattr__(
                self,
                "title",
                _clean(self.title),
            )

        object.__setattr__(
            self,
            "reasons",
            tuple(self.reasons),
        )


@dataclass(frozen=True)
class EvidenceSet:
    """
    Final collection of selected evidence units.
    """

    units: tuple[EvidenceUnit, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "units",
            tuple(self.units),
        )

    def __iter__(self):
        return iter(self.units)

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int) -> EvidenceUnit:
        return self.units[index]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _clean(value: str) -> str:
    """
    Normalize a string without changing its semantic content.
    """
    return " ".join(str(value).strip().split())


def _clean_many(values: Sequence[str]) -> tuple[str, ...]:
    """
    Normalize a sequence of strings and remove empty values.
    """
    cleaned: list[str] = []

    for value in values:
        normalized = _clean(value)

        if normalized:
            cleaned.append(normalized)

    return tuple(cleaned)