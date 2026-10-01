"""
Reusable retrieval contracts for the institution-agnostic RAG core.

This module defines the stable data objects passed between retrieval,
fusion, semantic alignment, reranking, evidence selection, and answering.

Design rules:
    - No institution-specific vocabulary or facts belong here.
    - Retrieval provenance is preserved instead of being discarded.
    - Retrieval signals are kept separate from semantic alignment.
    - Evidence quality is kept separate from relevance.
    - Semantic dimensions remain explicitly separated.
    - Objects are immutable so downstream stages cannot silently mutate
      upstream retrieval state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import re
import math
from typing import Any, Iterable, Mapping


# ============================================================
# Generic helpers
# ============================================================


def _clean_text(value: object) -> str:
    """Return a normalized single-line text representation."""

    return " ".join(
        str(value or "").strip().split()
    )


def _clean_items(
    values: Iterable[object] | None,
) -> tuple[str, ...]:
    """Normalize, deduplicate, and preserve order for string collections."""

    if values is None:
        return ()

    # A single string is one item, not an iterable of characters.
    if isinstance(values, str):
        values = (values,)

    output: list[str] = []
    seen: set[str] = set()

    for value in values:

        cleaned = _clean_text(
            value
        )

        if not cleaned:
            continue

        key = cleaned.casefold()

        if key in seen:
            continue

        seen.add(key)
        output.append(cleaned)

    return tuple(output)


def _validate_score(
    value: float | None,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    """Validate a numeric score without imposing retrieval semantics."""

    if value is None:
        return

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{name} must be a finite number or None."
        )

    numeric = float(value)

    if not math.isfinite(numeric):
        raise ValueError(
            f"{name} must be a finite number or None."
        )

    if (
        minimum is not None
        and numeric < minimum
    ):
        raise ValueError(
            f"{name} must be >= {minimum}."
        )

    if (
        maximum is not None
        and numeric > maximum
    ):
        raise ValueError(
            f"{name} must be <= {maximum}."
        )


# ============================================================
# Retrieval signal
# ============================================================


@dataclass(frozen=True, slots=True)
class RetrievalSignal:
    """
    One retrieval observation for one candidate.

    Examples of `channel` values include:
        dense
        bm25

    The core does not require a fixed vocabulary so additional generic
    retrieval channels can be added later.
    """

    channel: str
    rank: int
    score: float | None = None
    weight: float = 1.0
    query: str | None = None

    def __post_init__(self) -> None:

        channel = _clean_text(
            self.channel
        )

        if not channel:
            raise ValueError(
                "RetrievalSignal.channel cannot be empty."
            )

        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise ValueError(
                "RetrievalSignal.rank must be a positive integer."
            )

        _validate_score(
            self.score,
            name="RetrievalSignal.score",
        )

        _validate_score(
            self.weight,
            name="RetrievalSignal.weight",
            minimum=0.0,
        )

        object.__setattr__(
            self,
            "channel",
            channel,
        )

        if self.query is not None:

            object.__setattr__(
                self,
                "query",
                _clean_text(
                    self.query
                )
                or None,
            )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "channel": self.channel,
            "rank": self.rank,
            "score": self.score,
            "weight": self.weight,
            "query": self.query,
        }


# ============================================================
# Retrieval provenance
# ============================================================


@dataclass(frozen=True, slots=True)
class RetrievalProvenance:
    """
    Complete retrieval history retained for one candidate.

    Explicit dense/BM25 fields make common operations convenient while
    `signals` preserves the full underlying observations for auditing.
    """

    dense_rank: int | None = None
    dense_score: float | None = None

    bm25_rank: int | None = None
    bm25_score: float | None = None

    rrf_score: float = 0.0

    alternate_score_contribution: float = 0.0

    primary_query: str = ""

    retrieval_queries: tuple[str, ...] = field(
        default_factory=tuple
    )

    signals: tuple[RetrievalSignal, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:

        for name, rank in (
            (
                "dense_rank",
                self.dense_rank,
            ),
            (
                "bm25_rank",
                self.bm25_rank,
            ),
        ):

            if (
                rank is not None
                and (
                    isinstance(rank, bool)
                    or not isinstance(rank, int)
                    or rank < 1
                )
            ):
                raise ValueError(
                    f"{name} must be a positive integer or None."
                )

        _validate_score(
            self.dense_score,
            name="dense_score",
        )

        _validate_score(
            self.bm25_score,
            name="bm25_score",
        )

        _validate_score(
            self.rrf_score,
            name="rrf_score",
            minimum=0.0,
        )

        _validate_score(
            self.alternate_score_contribution,
            name="alternate_score_contribution",
            minimum=0.0,
        )

        object.__setattr__(
            self,
            "primary_query",
            _clean_text(
                self.primary_query
            ),
        )

        object.__setattr__(
            self,
            "retrieval_queries",
            _clean_items(
                self.retrieval_queries
            ),
        )

        normalized_signals = tuple(
            self.signals
        )

        for signal in normalized_signals:

            if not isinstance(
                signal,
                RetrievalSignal,
            ):
                raise TypeError(
                    "RetrievalProvenance.signals must contain "
                    "RetrievalSignal objects."
                )

        object.__setattr__(
            self,
            "signals",
            normalized_signals,
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "dense_rank": self.dense_rank,
            "dense_score": self.dense_score,
            "bm25_rank": self.bm25_rank,
            "bm25_score": self.bm25_score,
            "rrf_score": self.rrf_score,
            "alternate_score_contribution": (
                self.alternate_score_contribution
            ),
            "primary_query": self.primary_query,
            "retrieval_queries": list(
                self.retrieval_queries
            ),
            "signals": [
                signal.to_dict()
                for signal in self.signals
            ],
        }

    @property
    def retrieved_by_dense(self) -> bool:
        return self.dense_rank is not None

    @property
    def retrieved_by_bm25(self) -> bool:
        return self.bm25_rank is not None


# ============================================================
# Document meaning
# ============================================================


@dataclass(frozen=True, slots=True)
class DocumentMeaning:
    """
    Generic semantic representation of a document or query.

    Semantic dimensions are intentionally separated:

        programs
        entities
        topics
        attributes
        intent
        scope
        qualifiers
        constraints

    The values come from the active institution/deployment and generic
    query-understanding layers. This contract does not define any
    institution-specific vocabulary.
    """

    programs: tuple[str, ...] = field(
        default_factory=tuple
    )

    entities: tuple[str, ...] = field(
        default_factory=tuple
    )

    topics: tuple[str, ...] = field(
        default_factory=tuple
    )

    attributes: tuple[str, ...] = field(
        default_factory=tuple
    )

    intent: str | None = None

    scope: tuple[str, ...] = field(
        default_factory=tuple
    )

    qualifiers: tuple[str, ...] = field(
        default_factory=tuple
    )

    constraints: tuple[str, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:

        object.__setattr__(
            self,
            "programs",
            _clean_items(
                self.programs
            ),
        )

        object.__setattr__(
            self,
            "entities",
            _clean_items(
                self.entities
            ),
        )

        object.__setattr__(
            self,
            "topics",
            _clean_items(
                self.topics
            ),
        )

        object.__setattr__(
            self,
            "attributes",
            _clean_items(
                self.attributes
            ),
        )

        object.__setattr__(
            self,
            "scope",
            _clean_items(
                self.scope
            ),
        )

        object.__setattr__(
            self,
            "qualifiers",
            _clean_items(
                self.qualifiers
            ),
        )

        object.__setattr__(
            self,
            "constraints",
            _clean_items(
                self.constraints
            ),
        )

        if self.intent is not None:

            cleaned_intent = _clean_text(
                self.intent
            )

            object.__setattr__(
                self,
                "intent",
                cleaned_intent or None,
            )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "programs": list(
                self.programs
            ),
            "entities": list(
                self.entities
            ),
            "topics": list(
                self.topics
            ),
            "attributes": list(
                self.attributes
            ),
            "intent": self.intent,
            "scope": list(
                self.scope
            ),
            "qualifiers": list(
                self.qualifiers
            ),
            "constraints": list(
                self.constraints
            ),
        }


# ============================================================
# Candidate alignment
# ============================================================


@dataclass(frozen=True, slots=True)
class CandidateAlignment:
    """
    Semantic compatibility between a user request and one candidate.

    All normalized alignment scores are in [0, 1].
    """

    program_match: float = 0.0
    entity_match: float = 0.0
    topic_match: float = 0.0
    attribute_match: float = 0.0
    intent_match: float = 0.0
    scope_match: float = 0.0

    coverage: float = 0.0
    semantic_match: float = 0.0

    conflicts: tuple[str, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:

        for name in (
            "program_match",
            "entity_match",
            "topic_match",
            "attribute_match",
            "intent_match",
            "scope_match",
            "coverage",
            "semantic_match",
        ):

            _validate_score(
                getattr(
                    self,
                    name,
                ),
                name=name,
                minimum=0.0,
                maximum=1.0,
            )

        object.__setattr__(
            self,
            "conflicts",
            _clean_items(
                self.conflicts
            ),
        )

    @property
    def is_compatible(self) -> bool:
        """Return whether no explicit conflict was declared."""

        return not self.conflicts

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "program_match": self.program_match,
            "entity_match": self.entity_match,
            "topic_match": self.topic_match,
            "attribute_match": self.attribute_match,
            "intent_match": self.intent_match,
            "scope_match": self.scope_match,
            "coverage": self.coverage,
            "semantic_match": self.semantic_match,
            "conflicts": list(
                self.conflicts
            ),
            "is_compatible": self.is_compatible,
        }


# ============================================================
# Evidence quality
# ============================================================


@dataclass(frozen=True, slots=True)
class EvidenceQuality:
    """
    Intrinsic quality signals for the candidate's content.

    These describe whether content is clean and usable as evidence;
    they do not claim that content answers the current question.
    """

    content_quality: float = 1.0
    structural_quality: float = 1.0
    noise: float = 0.0

    def __post_init__(self) -> None:

        _validate_score(
            self.content_quality,
            name="content_quality",
            minimum=0.0,
            maximum=1.0,
        )

        _validate_score(
            self.structural_quality,
            name="structural_quality",
            minimum=0.0,
            maximum=1.0,
        )

        _validate_score(
            self.noise,
            name="noise",
            minimum=0.0,
            maximum=1.0,
        )

    @property
    def usable(self) -> bool:

        return (
            self.content_quality > 0.0
            and self.structural_quality > 0.0
            and self.noise < 1.0
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "content_quality": self.content_quality,
            "structural_quality": self.structural_quality,
            "noise": self.noise,
            "usable": self.usable,
        }


# ============================================================
# Stable document identity
# ============================================================


def _canonical_source_key(source: str) -> str:
    """Normalize equivalent local source paths without knowing any institution."""
    value = _clean_text(source).replace("\\", "/")
    value = re.sub(r"/{2,}", "/", value)

    # Collapse absolute local paths at a stable `data/` root when present.
    # URLs and non-local identifiers are preserved as-is.
    lowered = value.casefold()
    marker = "/data/"
    index = lowered.find(marker)
    if index >= 0 and not re.match(r"^[a-z][a-z0-9+.-]*://", value, re.I):
        value = "data/" + value[index + len(marker):]

    value = re.sub(r"^\./+", "", value)
    return value.casefold().strip()


def document_identity(
    document: Any,
) -> str:
    """
    Produce a stable identity from canonical source + chunk content.
    """

    page_content = _clean_text(
        getattr(
            document,
            "page_content",
            "",
        )
    )

    metadata = getattr(
        document,
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        Mapping,
    ):
        metadata = {}

    source = _canonical_source_key(str(metadata.get("source", "") or ""))

    identity = (
        f"{source}\n"
        f"{page_content}"
    )

    return hashlib.sha256(
        identity.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# Retrieval candidate
# ============================================================


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """
    Stable unit passed through the retrieval pipeline.

    A candidate owns the source document and all derived state required by
    later stages. Downstream modules should prefer this object over passing
    loose `(document, score, rank, metadata)` tuples.
    """

    document: Any

    document_id: str

    source: str

    provenance: RetrievalProvenance = field(
        default_factory=RetrievalProvenance
    )

    meaning: DocumentMeaning = field(
        default_factory=DocumentMeaning
    )

    alignment: CandidateAlignment = field(
        default_factory=CandidateAlignment
    )

    quality: EvidenceQuality = field(
        default_factory=EvidenceQuality
    )

    final_score: float = 0.0

    def __post_init__(self) -> None:

        if self.document is None:
            raise ValueError(
                "RetrievalCandidate.document cannot be None."
            )

        cleaned_source = _clean_text(
            self.source
        )

        if not cleaned_source:
            raise ValueError(
                "RetrievalCandidate.source cannot be empty."
            )

        cleaned_document_id = _clean_text(
            self.document_id
        )

        if not cleaned_document_id:
            raise ValueError(
                "RetrievalCandidate.document_id cannot be empty."
            )

        if not isinstance(
            self.provenance,
            RetrievalProvenance,
        ):
            raise TypeError(
                "provenance must be a RetrievalProvenance object."
            )

        if not isinstance(
            self.meaning,
            DocumentMeaning,
        ):
            raise TypeError(
                "meaning must be a DocumentMeaning object."
            )

        if not isinstance(
            self.alignment,
            CandidateAlignment,
        ):
            raise TypeError(
                "alignment must be a CandidateAlignment object."
            )

        if not isinstance(
            self.quality,
            EvidenceQuality,
        ):
            raise TypeError(
                "quality must be an EvidenceQuality object."
            )

        _validate_score(
            self.final_score,
            name="final_score",
        )

        object.__setattr__(
            self,
            "source",
            cleaned_source,
        )

        object.__setattr__(
            self,
            "document_id",
            cleaned_document_id,
        )

    @classmethod
    def from_document(
        cls,
        document: Any,
        *,
        provenance: RetrievalProvenance | None = None,
        meaning: DocumentMeaning | None = None,
        alignment: CandidateAlignment | None = None,
        quality: EvidenceQuality | None = None,
        document_id: str | None = None,
        source: str | None = None,
        final_score: float = 0.0,
    ) -> "RetrievalCandidate":
        """Build a candidate from a LangChain-style document."""

        metadata = getattr(
            document,
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            Mapping,
        ):
            metadata = {}

        resolved_source = (
            _clean_text(source)
            or _clean_text(
                metadata.get(
                    "source",
                    "",
                )
            )
        )

        if not resolved_source:
            resolved_source = "unknown"

        resolved_id = (
            document_id
            or document_identity(
                document
            )
        )

        return cls(
            document=document,
            document_id=resolved_id,
            source=resolved_source,
            provenance=(
                provenance
                or RetrievalProvenance()
            ),
            meaning=(
                meaning
                or DocumentMeaning()
            ),
            alignment=(
                alignment
                or CandidateAlignment()
            ),
            quality=(
                quality
                or EvidenceQuality()
            ),
            final_score=final_score,
        )

    def with_provenance(
        self,
        provenance: RetrievalProvenance,
    ) -> "RetrievalCandidate":

        return replace(
            self,
            provenance=provenance,
        )

    def with_meaning(
        self,
        meaning: DocumentMeaning,
    ) -> "RetrievalCandidate":

        return replace(
            self,
            meaning=meaning,
        )

    def with_alignment(
        self,
        alignment: CandidateAlignment,
    ) -> "RetrievalCandidate":

        return replace(
            self,
            alignment=alignment,
        )

    def with_quality(
        self,
        quality: EvidenceQuality,
    ) -> "RetrievalCandidate":

        return replace(
            self,
            quality=quality,
        )

    def with_final_score(
        self,
        final_score: float,
    ) -> "RetrievalCandidate":

        return replace(
            self,
            final_score=final_score,
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Return a JSON-friendly diagnostic representation."""

        return {
            "document_id": self.document_id,
            "source": self.source,
            "provenance": self.provenance.to_dict(),
            "meaning": self.meaning.to_dict(),
            "alignment": self.alignment.to_dict(),
            "quality": self.quality.to_dict(),
            "final_score": self.final_score,
        }


# ============================================================
# Public API
# ============================================================


__all__ = [
    "RetrievalSignal",
    "RetrievalProvenance",
    "DocumentMeaning",
    "CandidateAlignment",
    "EvidenceQuality",
    "RetrievalCandidate",
    "document_identity",
]