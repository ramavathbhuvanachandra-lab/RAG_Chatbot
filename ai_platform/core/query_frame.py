"""
Reusable semantic query-frame contracts.

The query frame represents what the user is actually asking for.

Core principles
---------------
1. The original user wording is always preserved.
2. Semantic interpretation is supporting structure, not a replacement
   for the original question.
3. Exact factual requests can declare stronger matching requirements.
4. Unknown / unusual wording is valid and does not fail the frame.
5. No institution-specific vocabulary belongs in this module.

Typical flow:

    User question
        ↓
    Query understanding
        ↓
    SemanticQueryFrame
        ↓
    Retrieval planning
        ↓
    Dense + BM25
        ↓
    Evidence alignment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


# ============================================================
# Helpers
# ============================================================


def _clean_text(value: object) -> str:
    """Normalize whitespace without changing the user's meaning."""

    return " ".join(
        str(
            value or ""
        ).strip().split()
    )


def _clean_items(
    values: Iterable[object] | None,
) -> tuple[str, ...]:
    """
    Normalize and deduplicate string collections while preserving order.
    """

    if values is None:
        return ()

    result: list[str] = []
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

        seen.add(
            key
        )

        result.append(
            cleaned
        )

    return tuple(
        result
    )


def _validate_unit_interval(
    value: float,
    name: str,
) -> None:
    """Validate a confidence-style value in [0, 1]."""

    if not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{name} must be numeric."
        )

    numeric = float(
        value
    )

    if not (
        0.0
        <= numeric
        <= 1.0
    ):
        raise ValueError(
            f"{name} must be between 0 and 1."
        )


# ============================================================
# Query facet
# ============================================================


@dataclass(frozen=True, slots=True)
class QueryFacet:
    """
    One semantic facet of the user's request.

    Examples:

        target:
            "item A"

        requested attribute:
            "price"

        condition:
            "for external candidates"

    The core does not define what labels are valid for an institution.
    """

    name: str
    value: str
    required: bool = True
    importance: float = 1.0

    def __post_init__(self) -> None:

        name = _clean_text(
            self.name
        )

        value = _clean_text(
            self.value
        )

        if not name:
            raise ValueError(
                "QueryFacet.name cannot be empty."
            )

        if not value:
            raise ValueError(
                "QueryFacet.value cannot be empty."
            )

        if not isinstance(
            self.required,
            bool,
        ):
            raise TypeError(
                "QueryFacet.required must be bool."
            )

        _validate_unit_interval(
            self.importance,
            "QueryFacet.importance",
        )

        object.__setattr__(
            self,
            "name",
            name,
        )

        object.__setattr__(
            self,
            "value",
            value,
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "name": self.name,
            "value": self.value,
            "required": self.required,
            "importance": self.importance,
        }


# ============================================================
# Query requirement
# ============================================================


@dataclass(frozen=True, slots=True)
class QueryRequirement:
    """
    Retrieval strictness for the current question.

    This is critical for preventing related-but-wrong evidence from being
    treated as equivalent.

    Example:

        "What is the price of item A?"

    should normally become an exact factual requirement with target + requested
    attribute alignment expected.

    A broader question such as:

        "Tell me about accommodation."

    can use a more flexible requirement.
    """

    mode: str = "standard"

    require_target_alignment: bool = False
    require_attribute_alignment: bool = False
    require_scope_alignment: bool = False

    reject_explicit_conflict: bool = True

    allow_partial_evidence: bool = True

    max_unmatched_required_facets: int = 0

    def __post_init__(self) -> None:

        mode = _clean_text(
            self.mode
        ).casefold()

        allowed_modes = {
            "broad",
            "standard",
            "focused",
            "exact",
        }

        if mode not in allowed_modes:
            raise ValueError(
                "QueryRequirement.mode must be one of: "
                + ", ".join(
                    sorted(
                        allowed_modes
                    )
                )
            )

        if not isinstance(
            self.max_unmatched_required_facets,
            int,
        ):
            raise TypeError(
                "max_unmatched_required_facets must be an integer."
            )

        if (
            self.max_unmatched_required_facets
            < 0
        ):
            raise ValueError(
                "max_unmatched_required_facets cannot be negative."
            )

        object.__setattr__(
            self,
            "mode",
            mode,
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "mode": self.mode,
            "require_target_alignment": (
                self.require_target_alignment
            ),
            "require_attribute_alignment": (
                self.require_attribute_alignment
            ),
            "require_scope_alignment": (
                self.require_scope_alignment
            ),
            "reject_explicit_conflict": (
                self.reject_explicit_conflict
            ),
            "allow_partial_evidence": (
                self.allow_partial_evidence
            ),
            "max_unmatched_required_facets": (
                self.max_unmatched_required_facets
            ),
        }


# ============================================================
# Semantic query frame
# ============================================================


@dataclass(frozen=True, slots=True)
class SemanticQueryFrame:
    """
    Structured representation of the user's information need.

    The original question is always authoritative.

    `semantic_query` is an optional retrieval aid. It must never replace
    `original_query` unless a later, explicitly designed retrieval gate
    determines that doing so is safe.
    """

    # --------------------------------------------------------
    # Original / normalized question
    # --------------------------------------------------------

    original_query: str

    normalized_query: str = ""

    semantic_query: str = ""

    # --------------------------------------------------------
    # Main request meaning
    # --------------------------------------------------------

    target: str | None = None

    request_type: str | None = None

    # --------------------------------------------------------
    # Structured facets
    # --------------------------------------------------------

    facets: tuple[QueryFacet, ...] = field(
        default_factory=tuple
    )

    qualifiers: tuple[str, ...] = field(
        default_factory=tuple
    )

    conditions: tuple[str, ...] = field(
        default_factory=tuple
    )

    relations: tuple[str, ...] = field(
        default_factory=tuple
    )

    temporal_context: tuple[str, ...] = field(
        default_factory=tuple
    )

    comparison_targets: tuple[str, ...] = field(
        default_factory=tuple
    )

    preserved_terms: tuple[str, ...] = field(
        default_factory=tuple
    )

    # --------------------------------------------------------
    # Question shape
    # --------------------------------------------------------

    is_list_question: bool = False

    is_comparison_question: bool = False

    is_multi_part: bool = False

    # --------------------------------------------------------
    # Retrieval strictness
    # --------------------------------------------------------

    requirement: QueryRequirement = field(
        default_factory=QueryRequirement
    )

    # --------------------------------------------------------
    # Confidence / safety
    # --------------------------------------------------------

    confidence: float = 0.0

    needs_clarification: bool = False

    clarification_reason: str = ""

    language: str | None = None

    interpretation_status: str = "unresolved"

    def __post_init__(self) -> None:

        original_query = _clean_text(
            self.original_query
        )

        if not original_query:
            raise ValueError(
                "SemanticQueryFrame.original_query "
                "cannot be empty."
            )

        normalized_query = _clean_text(
            self.normalized_query
        )

        semantic_query = _clean_text(
            self.semantic_query
        )

        target = (
            _clean_text(
                self.target
            )
            or None
        )

        request_type = (
            _clean_text(
                self.request_type
            )
            or None
        )

        language = (
            _clean_text(
                self.language
            )
            or None
        )

        interpretation_status = (
            _clean_text(
                self.interpretation_status
            )
            .casefold()
        )

        allowed_statuses = {
            "unresolved",
            "trusted",
            "review",
            "rejected",
        }

        if (
            interpretation_status
            not in allowed_statuses
        ):
            raise ValueError(
                "interpretation_status must be one of: "
                + ", ".join(
                    sorted(
                        allowed_statuses
                    )
                )
            )

        _validate_unit_interval(
            self.confidence,
            "SemanticQueryFrame.confidence",
        )

        if not isinstance(
            self.is_list_question,
            bool,
        ):
            raise TypeError(
                "is_list_question must be bool."
            )

        if not isinstance(
            self.is_comparison_question,
            bool,
        ):
            raise TypeError(
                "is_comparison_question must be bool."
            )

        if not isinstance(
            self.is_multi_part,
            bool,
        ):
            raise TypeError(
                "is_multi_part must be bool."
            )

        if not isinstance(
            self.needs_clarification,
            bool,
        ):
            raise TypeError(
                "needs_clarification must be bool."
            )

        object.__setattr__(
            self,
            "original_query",
            original_query,
        )

        object.__setattr__(
            self,
            "normalized_query",
            normalized_query,
        )

        object.__setattr__(
            self,
            "semantic_query",
            semantic_query,
        )

        object.__setattr__(
            self,
            "target",
            target,
        )

        object.__setattr__(
            self,
            "request_type",
            request_type,
        )

        object.__setattr__(
            self,
            "language",
            language,
        )

        object.__setattr__(
            self,
            "interpretation_status",
            interpretation_status,
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
            "conditions",
            _clean_items(
                self.conditions
            ),
        )

        object.__setattr__(
            self,
            "relations",
            _clean_items(
                self.relations
            ),
        )

        object.__setattr__(
            self,
            "temporal_context",
            _clean_items(
                self.temporal_context
            ),
        )

        object.__setattr__(
            self,
            "comparison_targets",
            _clean_items(
                self.comparison_targets
            ),
        )

        object.__setattr__(
            self,
            "preserved_terms",
            _clean_items(
                self.preserved_terms
            ),
        )

        normalized_facets = tuple(
            self.facets
        )

        for facet in normalized_facets:

            if not isinstance(
                facet,
                QueryFacet,
            ):
                raise TypeError(
                    "facets must contain QueryFacet objects."
                )

        object.__setattr__(
            self,
            "facets",
            normalized_facets,
        )

        if not self.normalized_query:
            object.__setattr__(
                self,
                "normalized_query",
                original_query,
            )

    # ========================================================
    # Derived properties
    # ========================================================

    @property
    def has_semantic_interpretation(
        self,
    ) -> bool:
        """
        Return whether a meaningful interpretation exists.
        """

        return bool(
            self.target
            or self.request_type
            or self.facets
            or self.qualifiers
            or self.conditions
            or self.relations
            or self.temporal_context
        )

    @property
    def is_trusted(
        self,
    ) -> bool:
        return (
            self.interpretation_status
            == "trusted"
            and self.confidence
            > 0.0
        )

    @property
    def exact_request(
        self,
    ) -> bool:
        return (
            self.requirement.mode
            == "exact"
        )

    @property
    def required_facets(
        self,
    ) -> tuple[QueryFacet, ...]:

        return tuple(
            facet
            for facet in self.facets
            if facet.required
        )

    @property
    def retrieval_text(
        self,
    ) -> str:
        """
        Return safe retrieval input.

        The original question is always present.

        A semantic query may supplement the original wording, but cannot
        silently replace it.
        """

        parts = [
            self.original_query
        ]

        if self.semantic_query:
            semantic_normalized = (
                self.semantic_query.casefold()
            )

            original_normalized = (
                self.original_query.casefold()
            )

            if (
                semantic_normalized
                != original_normalized
            ):
                parts.append(
                    self.semantic_query
                )

        return " ".join(
            part
            for part in parts
            if part
        )

    # ========================================================
    # Serialization
    # ========================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "semantic_query": self.semantic_query,
            "target": self.target,
            "request_type": self.request_type,
            "facets": [
                facet.to_dict()
                for facet in self.facets
            ],
            "qualifiers": list(
                self.qualifiers
            ),
            "conditions": list(
                self.conditions
            ),
            "relations": list(
                self.relations
            ),
            "temporal_context": list(
                self.temporal_context
            ),
            "comparison_targets": list(
                self.comparison_targets
            ),
            "preserved_terms": list(
                self.preserved_terms
            ),
            "is_list_question": (
                self.is_list_question
            ),
            "is_comparison_question": (
                self.is_comparison_question
            ),
            "is_multi_part": (
                self.is_multi_part
            ),
            "requirement": (
                self.requirement.to_dict()
            ),
            "confidence": self.confidence,
            "needs_clarification": (
                self.needs_clarification
            ),
            "clarification_reason": (
                self.clarification_reason
            ),
            "language": self.language,
            "interpretation_status": (
                self.interpretation_status
            ),
        }


# ============================================================
# Public API
# ============================================================


__all__ = [
    "QueryFacet",
    "QueryRequirement",
    "SemanticQueryFrame",
]