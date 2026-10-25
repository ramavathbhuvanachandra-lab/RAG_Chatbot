"""Generic query contracts for the reusable institutional AI assistant core.

This module defines contracts only. It does not call models, retrieve data,
rank candidates, or make evidence decisions.

Migration rule:
    old backend = behavioral reference
    this module  = reusable structural contract

``SemanticQueryFrame`` keeps the observed legacy fields so downstream code can
migrate incrementally without replacing working behavior all at once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from math import isfinite
from typing import Any, TypeAlias

ScalarValue: TypeAlias = str | int | float | bool
ResolutionState: TypeAlias = str
VerificationState: TypeAlias = str


class QueryRoute(str, Enum):
    """Principled routing taxonomy for student institutional questions."""

    SIMPLE_SINGLE_INTENT = "simple_single_intent"
    SINGLE_INTENT_WITH_CONSTRAINTS = "single_intent_with_constraints"
    MULTI_SOURCE_SINGLE_INTENT = "multi_source_single_intent"
    COMPARISON = "comparison"
    MULTI_INTENT = "multi_intent"
    CONVERSATIONAL_FOLLOW_UP = "conversational_follow_up"
    AMBIGUOUS_NEEDS_CLARIFICATION = "ambiguous_needs_clarification"
    OUT_OF_SCOPE = "out_of_scope"
    UNCERTAIN_SCOPE = "uncertain_scope"


class GuardDecision(str, Enum):
    """Explicit decision contract for input-level scope and security evaluation."""

    IN_SCOPE = "IN_SCOPE"
    MIXED_SCOPE_WITH_INJECTION = "MIXED_SCOPE_WITH_INJECTION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    SENSITIVE_REDIRECT = "SENSITIVE_REDIRECT"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class GuardResult:
    """Typed result of input-level scope and security evaluation."""

    decision: GuardDecision
    original_query: str
    sanitized_query: str
    reason: str
    flags: tuple[str, ...] = ()
    confidence: float = 1.0
    safe_response: str | None = None



def _text(value: object | None) -> str:
    return " ".join(str(value or "").strip().split())


def _optional_text(value: object | None) -> str | None:
    value = _text(value)
    return value or None


def _texts(values: object | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, (list, tuple, set, frozenset)):
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = _text(value)
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _confidence(value: object | None, default: float = 0.0) -> float:
    try:
        numeric = float(default if value is None else value)
    except (TypeError, ValueError):
        numeric = default
    return min(1.0, max(0.0, numeric)) if isfinite(numeric) else min(1.0, max(0.0, default))


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


class _Contract:
    """Small shared serialization helper; no domain behavior lives here."""

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class Intent(_Contract):
    name: str
    confidence: float = 0.0
    priority: float = 1.0

    def __post_init__(self) -> None:
        name = _text(self.name)
        if not name:
            raise ValueError("Intent.name cannot be empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "priority", _confidence(self.priority, 1.0))


@dataclass(frozen=True, slots=True)
class Target(_Contract):
    text: str
    entity_id: str | None = None
    entity_type: str | None = None
    scope_id: str | None = None
    confidence: float = 0.0
    resolution_state: ResolutionState = "unresolved"

    def __post_init__(self) -> None:
        text = _text(self.text)
        if not text:
            raise ValueError("Target.text cannot be empty.")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "entity_id", _optional_text(self.entity_id))
        object.__setattr__(self, "entity_type", _optional_text(self.entity_type))
        object.__setattr__(self, "scope_id", _optional_text(self.scope_id))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "resolution_state", _text(self.resolution_state) or "unresolved")


@dataclass(frozen=True, slots=True)
class Entity(_Contract):
    name: str
    entity_type: str
    entity_id: str | None = None
    aliases: tuple[str, ...] = ()
    scope_id: str | None = None
    confidence: float = 0.0
    resolution_state: ResolutionState = "unresolved"

    def __post_init__(self) -> None:
        name, entity_type = _text(self.name), _text(self.entity_type)
        if not name or not entity_type:
            raise ValueError("Entity.name and Entity.entity_type are required.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "entity_type", entity_type)
        object.__setattr__(self, "entity_id", _optional_text(self.entity_id))
        object.__setattr__(self, "aliases", _texts(self.aliases))
        object.__setattr__(self, "scope_id", _optional_text(self.scope_id))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "resolution_state", _text(self.resolution_state) or "unresolved")


@dataclass(frozen=True, slots=True)
class EntityMention(_Contract):
    text: str
    start: int | None = None
    end: int | None = None
    normalized_text: str | None = None
    entity_id: str | None = None
    entity_type: str | None = None
    confidence: float = 0.0
    resolution_state: ResolutionState = "unresolved"

    def __post_init__(self) -> None:
        text = _text(self.text)
        if not text:
            raise ValueError("EntityMention.text cannot be empty.")
        if self.start is not None and self.start < 0:
            raise ValueError("EntityMention.start cannot be negative.")
        if self.end is not None and self.end < 0:
            raise ValueError("EntityMention.end cannot be negative.")
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("EntityMention.end cannot be less than start.")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "normalized_text", _optional_text(self.normalized_text))
        object.__setattr__(self, "entity_id", _optional_text(self.entity_id))
        object.__setattr__(self, "entity_type", _optional_text(self.entity_type))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "resolution_state", _text(self.resolution_state) or "unresolved")


@dataclass(frozen=True, slots=True)
class Concept(_Contract):
    name: str
    concept_id: str | None = None
    confidence: float = 0.0

    def __post_init__(self) -> None:
        name = _text(self.name)
        if not name:
            raise ValueError("Concept.name cannot be empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "concept_id", _optional_text(self.concept_id))
        object.__setattr__(self, "confidence", _confidence(self.confidence))


@dataclass(frozen=True, slots=True)
class Relation(_Contract):
    relation_type: str
    subject_ref: str
    object_ref: str
    confidence: float = 0.0
    required: bool = True

    def __post_init__(self) -> None:
        values = tuple(_text(v) for v in (self.relation_type, self.subject_ref, self.object_ref))
        if not all(values):
            raise ValueError("Relation type and references are required.")
        object.__setattr__(self, "relation_type", values[0])
        object.__setattr__(self, "subject_ref", values[1])
        object.__setattr__(self, "object_ref", values[2])
        object.__setattr__(self, "confidence", _confidence(self.confidence))


@dataclass(frozen=True, slots=True)
class Qualifier(_Contract):
    name: str
    value: str
    required: bool = True
    importance: float = 1.0

    def __post_init__(self) -> None:
        name, value = _text(self.name), _text(self.value)
        if not name or not value:
            raise ValueError("Qualifier.name and Qualifier.value are required.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "importance", _confidence(self.importance, 1.0))


@dataclass(frozen=True, slots=True)
class Constraint(_Contract):
    name: str
    value: ScalarValue
    operator: str = "eq"
    required: bool = True
    importance: float = 1.0

    def __post_init__(self) -> None:
        name, operator = _text(self.name), _text(self.operator) or "eq"
        if not name:
            raise ValueError("Constraint.name cannot be empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "importance", _confidence(self.importance, 1.0))


@dataclass(frozen=True, slots=True)
class TemporalConstraint(_Contract):
    kind: str
    value: str | None = None
    start: str | None = None
    end: str | None = None
    granularity: str | None = None
    timezone: str | None = None
    confidence: float = 0.0

    def __post_init__(self) -> None:
        kind = _text(self.kind)
        if not kind:
            raise ValueError("TemporalConstraint.kind cannot be empty.")
        value, start, end = map(_optional_text, (self.value, self.start, self.end))
        if not value and not (start or end):
            raise ValueError("TemporalConstraint needs value or start/end.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "granularity", _optional_text(self.granularity))
        object.__setattr__(self, "timezone", _optional_text(self.timezone))
        object.__setattr__(self, "confidence", _confidence(self.confidence))


@dataclass(frozen=True, slots=True)
class NumericRequirement(_Contract):
    name: str
    value: float | int
    operator: str = "eq"
    unit: str | None = None
    confidence: float = 0.0

    def __post_init__(self) -> None:
        name, operator = _text(self.name), _text(self.operator) or "eq"
        if not name:
            raise ValueError("NumericRequirement.name cannot be empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "unit", _optional_text(self.unit))
        object.__setattr__(self, "confidence", _confidence(self.confidence))


@dataclass(frozen=True, slots=True)
class Scope(_Contract):
    scope_id: str | None = None
    scope_type: str | None = None
    value: str | None = None
    confidence: float = 0.0
    resolution_state: ResolutionState = "unresolved"

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope_id", _optional_text(self.scope_id))
        object.__setattr__(self, "scope_type", _optional_text(self.scope_type))
        object.__setattr__(self, "value", _optional_text(self.value))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "resolution_state", _text(self.resolution_state) or "unresolved")


@dataclass(frozen=True, slots=True)
class ListIntent(_Contract):
    is_list: bool = False
    item_type: str | None = None
    min_items: int | None = None
    max_items: int | None = None
    ordered: bool = False
    sort_by: str | None = None

    def __post_init__(self) -> None:
        min_items = None if self.min_items is None else max(0, int(self.min_items))
        max_items = None if self.max_items is None else max(0, int(self.max_items))
        if min_items is not None and max_items is not None and max_items < min_items:
            raise ValueError("ListIntent.max_items cannot be less than min_items.")
        object.__setattr__(self, "min_items", min_items)
        object.__setattr__(self, "max_items", max_items)
        object.__setattr__(self, "item_type", _optional_text(self.item_type))
        object.__setattr__(self, "sort_by", _optional_text(self.sort_by))


@dataclass(frozen=True, slots=True)
class Ambiguity(_Contract):
    is_ambiguous: bool = False
    terms: tuple[str, ...] = ()
    reason: str = ""
    candidate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", _texts(self.terms))
        object.__setattr__(self, "candidate_ids", _texts(self.candidate_ids))
        object.__setattr__(self, "reason", _text(self.reason))


@dataclass(frozen=True, slots=True)
class RetrievalRequirement(_Contract):
    """Retrieval/evidence strictness derived from query structure."""

    mode: str = "standard"
    require_target_alignment: bool = False
    require_attribute_alignment: bool = False
    require_scope_alignment: bool = False
    reject_explicit_conflict: bool = True
    allow_partial_evidence: bool = True
    max_unmatched_required_facets: int = 1

    def __post_init__(self) -> None:
        mode = _text(self.mode).casefold() or "standard"
        if mode not in {"broad", "standard", "focused", "exact"}:
            raise ValueError(f"Unsupported retrieval requirement mode: {mode!r}")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "max_unmatched_required_facets", max(0, int(self.max_unmatched_required_facets)))

    @property
    def exact(self) -> bool:
        return self.mode == "exact"


# Current legacy code imports this exact name.
QueryRequirement: TypeAlias = RetrievalRequirement


@dataclass(frozen=True, slots=True)
class QueryFacet(_Contract):
    """Named semantic facet used by the current interpreter/verifier."""

    name: str
    value: str
    required: bool = True
    importance: float = 1.0

    def __post_init__(self) -> None:
        name, value = _text(self.name), _text(self.value)
        if not name or not value:
            raise ValueError("QueryFacet.name and QueryFacet.value are required.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "importance", _confidence(self.importance, 1.0))


@dataclass(frozen=True, slots=True)
class Query(_Contract):
    """Canonical query contract consumed by new core modules."""

    original_query: str
    normalized_query: str = ""
    search_query: str = ""
    intents: tuple[Intent, ...] = ()
    target: Target | None = None
    entities: tuple[Entity, ...] = ()
    entity_mentions: tuple[EntityMention, ...] = ()
    concepts: tuple[Concept, ...] = ()
    relations: tuple[Relation, ...] = ()
    qualifiers: tuple[Qualifier, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    temporal_constraints: tuple[TemporalConstraint, ...] = ()
    numeric_requirements: tuple[NumericRequirement, ...] = ()
    scope: Scope | None = None
    request_type: str | None = None
    list_intent: ListIntent = field(default_factory=ListIntent)
    ambiguity: Ambiguity = field(default_factory=Ambiguity)
    unknown_terms: tuple[str, ...] = ()
    resolution_state: ResolutionState = "unresolved"
    verification_status: VerificationState = "not_run"
    confidence: float = 0.0
    language: str | None = None
    is_comparison: bool = False
    is_multi_part: bool = False
    comparison_targets: tuple[Target, ...] = ()
    preserved_terms: tuple[str, ...] = ()
    needs_clarification: bool = False
    clarification_reason: str = ""
    retrieval_requirement: RetrievalRequirement = field(default_factory=RetrievalRequirement)
    semantic_grounded: bool = False

    def __post_init__(self) -> None:
        original = _text(self.original_query)
        if not original:
            raise ValueError("Query.original_query cannot be empty.")
        object.__setattr__(self, "original_query", original)
        object.__setattr__(self, "normalized_query", _text(self.normalized_query) or original)
        object.__setattr__(self, "search_query", _text(self.search_query))
        object.__setattr__(self, "intents", tuple(self.intents))
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "entity_mentions", tuple(self.entity_mentions))
        object.__setattr__(self, "concepts", tuple(self.concepts))
        object.__setattr__(self, "relations", tuple(self.relations))
        object.__setattr__(self, "qualifiers", tuple(self.qualifiers))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "temporal_constraints", tuple(self.temporal_constraints))
        object.__setattr__(self, "numeric_requirements", tuple(self.numeric_requirements))
        object.__setattr__(self, "comparison_targets", tuple(self.comparison_targets))
        object.__setattr__(self, "unknown_terms", _texts(self.unknown_terms))
        object.__setattr__(self, "preserved_terms", _texts(self.preserved_terms))
        object.__setattr__(self, "request_type", _optional_text(self.request_type))
        object.__setattr__(self, "language", _optional_text(self.language))
        object.__setattr__(self, "resolution_state", _text(self.resolution_state) or "unresolved")
        object.__setattr__(self, "verification_status", _text(self.verification_status) or "not_run")
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "clarification_reason", _text(self.clarification_reason))

    @property
    def primary_intent(self) -> Intent | None:
        return self.intents[0] if self.intents else None

    @property
    def retrieval_text(self) -> str:
        if not self.search_query or self.search_query.casefold() == self.original_query.casefold():
            return self.original_query
        return f"{self.original_query} {self.search_query}"


@dataclass(frozen=True, slots=True)
class SemanticQueryFrame(_Contract):
    """Legacy-compatible frame plus optional structured query fields."""

    # Observed legacy constructor contract.
    original_query: str
    normalized_query: str = ""
    semantic_query: str = ""
    target: str | None = None
    request_type: str | None = None
    facets: tuple[QueryFacet, ...] = ()
    qualifiers: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    temporal_context: tuple[str, ...] = ()
    comparison_targets: tuple[str, ...] = ()
    preserved_terms: tuple[str, ...] = ()
    is_list_question: bool = False
    is_comparison_question: bool = False
    is_multi_part: bool = False
    requirement: RetrievalRequirement = field(default_factory=RetrievalRequirement)
    confidence: float = 0.0
    needs_clarification: bool = False
    clarification_reason: str = ""
    language: str | None = None
    interpretation_status: str = "review"

    # Structured fields for the new architecture.
    intents: tuple[Intent, ...] = ()
    entities: tuple[Entity, ...] = ()
    entity_mentions: tuple[EntityMention, ...] = ()
    concepts: tuple[Concept, ...] = ()
    structured_qualifiers: tuple[Qualifier, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    temporal_constraints: tuple[TemporalConstraint, ...] = ()
    numeric_requirements: tuple[NumericRequirement, ...] = ()
    scope: Scope | None = None
    ambiguity: Ambiguity = field(default_factory=Ambiguity)
    unknown_terms: tuple[str, ...] = ()
    resolution_state: ResolutionState = "unresolved"
    verification_status: VerificationState = "not_run"
    semantic_grounded: bool = False

    def __post_init__(self) -> None:
        original = _text(self.original_query)
        if not original:
            raise ValueError("SemanticQueryFrame.original_query cannot be empty.")
        object.__setattr__(self, "original_query", original)
        object.__setattr__(self, "normalized_query", _text(self.normalized_query) or original)
        object.__setattr__(self, "semantic_query", _text(self.semantic_query))
        object.__setattr__(self, "target", _optional_text(self.target))
        object.__setattr__(self, "request_type", _optional_text(self.request_type))
        object.__setattr__(self, "facets", tuple(self.facets))
        object.__setattr__(self, "qualifiers", _texts(self.qualifiers))
        object.__setattr__(self, "conditions", _texts(self.conditions))
        object.__setattr__(self, "relations", _texts(self.relations))
        object.__setattr__(self, "temporal_context", _texts(self.temporal_context))
        object.__setattr__(self, "comparison_targets", _texts(self.comparison_targets))
        object.__setattr__(self, "preserved_terms", _texts(self.preserved_terms))
        object.__setattr__(self, "unknown_terms", _texts(self.unknown_terms))
        object.__setattr__(self, "intents", tuple(self.intents))
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "entity_mentions", tuple(self.entity_mentions))
        object.__setattr__(self, "concepts", tuple(self.concepts))
        object.__setattr__(self, "structured_qualifiers", tuple(self.structured_qualifiers))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "temporal_constraints", tuple(self.temporal_constraints))
        object.__setattr__(self, "numeric_requirements", tuple(self.numeric_requirements))
        object.__setattr__(self, "resolution_state", _text(self.resolution_state) or "unresolved")
        object.__setattr__(self, "verification_status", _text(self.verification_status) or "not_run")
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "clarification_reason", _text(self.clarification_reason))
        object.__setattr__(self, "language", _optional_text(self.language))

    @property
    def retrieval_text(self) -> str:
        """The original user wording always remains part of retrieval."""
        if not self.semantic_query or self.semantic_query.casefold() == self.original_query.casefold():
            return self.original_query
        return f"{self.original_query} {self.semantic_query}"

    def to_query(self) -> Query:
        """Convert the migration frame into the canonical query contract."""
        qualifiers = self.structured_qualifiers or tuple(Qualifier("qualifier", value) for value in self.qualifiers)
        constraints = self.constraints or tuple(Constraint("condition", value) for value in self.conditions)
        temporal = self.temporal_constraints or tuple(TemporalConstraint("context", value=value) for value in self.temporal_context)
        target = Target(self.target, confidence=self.confidence, resolution_state=self.resolution_state) if self.target else None
        comparisons = tuple(Target(value, confidence=self.confidence, resolution_state=self.resolution_state) for value in self.comparison_targets)
        ambiguity = self.ambiguity
        if self.needs_clarification and not ambiguity.is_ambiguous:
            ambiguity = Ambiguity(True, reason=self.clarification_reason)
        return Query(
            original_query=self.original_query,
            normalized_query=self.normalized_query,
            search_query=self.semantic_query,
            intents=self.intents,
            target=target,
            entities=self.entities,
            entity_mentions=self.entity_mentions,
            concepts=self.concepts,
            qualifiers=qualifiers,
            constraints=constraints,
            temporal_constraints=temporal,
            numeric_requirements=self.numeric_requirements,
            scope=self.scope,
            request_type=self.request_type,
            list_intent=ListIntent(is_list=self.is_list_question),
            ambiguity=ambiguity,
            unknown_terms=self.unknown_terms,
            resolution_state=self.resolution_state,
            verification_status=self.verification_status,
            confidence=self.confidence,
            language=self.language,
            is_comparison=self.is_comparison_question,
            is_multi_part=self.is_multi_part,
            comparison_targets=comparisons,
            preserved_terms=self.preserved_terms,
            needs_clarification=self.needs_clarification,
            clarification_reason=self.clarification_reason,
            retrieval_requirement=self.requirement,
            semantic_grounded=self.semantic_grounded,
        )


@dataclass(frozen=True, slots=True)
class SemanticCondition(_Contract):
    """Atomic constraint or condition with logical relationship (AND/OR/NOT)."""

    name: str
    value: str
    operator: str = "eq"
    logical_relation: str = "AND"
    required: bool = True
    unit: str | None = None

    def __post_init__(self) -> None:
        name = _text(self.name)
        value = _text(self.value)
        if not name or not value:
            raise ValueError("SemanticCondition name and value are required.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "operator", _text(self.operator) or "eq")
        object.__setattr__(self, "logical_relation", _text(self.logical_relation).upper() or "AND")
        object.__setattr__(self, "unit", _optional_text(self.unit))


@dataclass(frozen=True, slots=True)
class AtomicIntentBinding(_Contract):
    """Lossless representation of a single atomic query intent."""

    intent_id: str
    question: str
    request_type: str | None = None
    program: str | None = None
    degree: str | None = None
    target: str | None = None
    attributes: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    numeric_constraints: tuple[NumericRequirement, ...] = ()
    qualifications: tuple[str, ...] = ()
    conditions: tuple[SemanticCondition, ...] = ()
    negated: bool = False
    retrieval_query: str = ""
    retrieval_queries: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_id", _text(self.intent_id))
        object.__setattr__(self, "question", _text(self.question))
        object.__setattr__(self, "request_type", _optional_text(self.request_type))
        object.__setattr__(self, "program", _optional_text(self.program))
        object.__setattr__(self, "degree", _optional_text(self.degree))
        object.__setattr__(self, "target", _optional_text(self.target))
        object.__setattr__(self, "attributes", _texts(self.attributes))
        object.__setattr__(self, "categories", _texts(self.categories))
        object.__setattr__(self, "numeric_constraints", tuple(self.numeric_constraints))
        object.__setattr__(self, "qualifications", _texts(self.qualifications))
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(self, "retrieval_query", _text(self.retrieval_query))
        object.__setattr__(self, "retrieval_queries", _texts(self.retrieval_queries))


@dataclass(frozen=True, slots=True)
class LosslessSemanticQuery(_Contract):
    """Phase 2 Lossless Semantic Query Contract (LSQC).

    Ensures zero loss of user question semantics, atomic intents, program/degree
    bindings, applicant categories, numerical constraints, qualification routes,
    negation, and condition relationships between understanding and retrieval.
    """

    original_query: str
    resolved_query: str = ""
    institution_id: str | None = None
    program_binding: str | None = None
    degree_binding: str | None = None
    applicant_categories: tuple[str, ...] = ()
    numeric_constraints: tuple[NumericRequirement, ...] = ()
    qualifications: tuple[str, ...] = ()
    negation: bool = False
    time_limits: tuple[str, ...] = ()
    condition_relationships: tuple[SemanticCondition, ...] = ()
    intents: tuple[AtomicIntentBinding, ...] = ()
    is_multi_intent: bool = False
    preserved_terms: tuple[str, ...] = ()
    query_route: str = "simple_single_intent"

    def __post_init__(self) -> None:
        orig = _text(self.original_query)
        if not orig:
            raise ValueError("LosslessSemanticQuery.original_query cannot be empty.")
        object.__setattr__(self, "original_query", orig)
        object.__setattr__(self, "resolved_query", _text(self.resolved_query) or orig)
        object.__setattr__(self, "institution_id", _optional_text(self.institution_id))
        object.__setattr__(self, "program_binding", _optional_text(self.program_binding))
        object.__setattr__(self, "degree_binding", _optional_text(self.degree_binding))
        object.__setattr__(self, "applicant_categories", _texts(self.applicant_categories))
        object.__setattr__(self, "numeric_constraints", tuple(self.numeric_constraints))
        object.__setattr__(self, "qualifications", _texts(self.qualifications))
        object.__setattr__(self, "time_limits", _texts(self.time_limits))
        object.__setattr__(self, "condition_relationships", tuple(self.condition_relationships))
        object.__setattr__(self, "intents", tuple(self.intents))
        object.__setattr__(self, "preserved_terms", _texts(self.preserved_terms))
        object.__setattr__(self, "query_route", _text(self.query_route) or "simple_single_intent")


__all__ = [
    "ScalarValue",
    "ResolutionState",
    "VerificationState",
    "QueryRoute",
    "Intent",
    "Target",
    "Entity",
    "EntityMention",
    "Concept",
    "Relation",
    "Qualifier",
    "Constraint",
    "TemporalConstraint",
    "NumericRequirement",
    "Scope",
    "ListIntent",
    "Ambiguity",
    "RetrievalRequirement",
    "QueryRequirement",
    "QueryFacet",
    "Query",
    "SemanticQueryFrame",
    "SemanticCondition",
    "AtomicIntentBinding",
    "LosslessSemanticQuery",
]