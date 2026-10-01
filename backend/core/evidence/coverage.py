"""Generic evidence coverage and completeness for the reusable RAG core.

Architecture
------------
Coverage is a *semantic completeness* layer, not a document classifier.

    query contract
        +
    ranked retrieval candidates
        ->
    derive required information units
        ->
    measure support for each unit
        ->
    supported / partial / insufficient

The module deliberately avoids institution-specific vocabulary, source IDs,
file names, program-name tables, department lists, or corpus-specific rules.
Institution knowledge belongs in the semantic registry and upstream query /
retrieval contracts.

Important distinction
---------------------
- Retrieval asks: "Which evidence might answer this?"
- Scope asks: "Is this evidence compatible with the request?"
- Coverage asks: "Does the compatible evidence contain every required part?"

A high relevance score alone must never imply complete coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Literal, Sequence

from backend.core.query.models import Query
from backend.core.retrieval_contracts import RetrievalCandidate


CoverageStatus = Literal["supported", "partial", "insufficient"]


# ---------------------------------------------------------------------------
# Generic policy constants
# ---------------------------------------------------------------------------
# These are algorithmic thresholds, not institution data.  They should remain
# stable across deployments unless benchmarking demonstrates a core-level need
# to tune them.

SUPPORTED_THRESHOLD = 0.68
PARTIAL_THRESHOLD = 0.42
UNIT_COVERAGE_THRESHOLD = 0.52
FOCUSED_MIN_SUPPORT = PARTIAL_THRESHOLD

# List questions need evidence of list structure, not an institution-specific
# expected item count.  A requested min_items value, when explicitly supplied
# by the query contract, takes precedence over these structural heuristics.
LIST_ITEM_MIN_FOR_STRONG_SINGLE_SOURCE = 4
LIST_ITEM_MIN_FOR_PARTIAL = 1
LIST_MIN_SOURCE_COUNT_FOR_DISTRIBUTED = 2
LIST_STRUCTURE_RUN_MIN = 2

# Generic structural phrases. These do not describe any institution; they
# describe document structure used to signal an enumerable collection.
_EXHAUSTIVE_LIST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bfollowing\b", re.I),
    re.compile(r"\blisted\s+below\b", re.I),
    re.compile(r"\b(?:includes?|comprises?|consists\s+of)\b", re.I),
)

_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*•]|\(?[A-Za-z0-9]+[.)])\s+")
_NUMBERED_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
_TABLE_ROW_RE = re.compile(r"(?m)^\s*\|[^\n]*\|")
_SEMICOLON_SPLIT_RE = re.compile(r"\s*;\s*")

# Numeric facts are recognized structurally.  They are not assumed to be any
# particular institutional field.  The query's NumericRequirement supplies the
# requested semantic type/name.
_CURRENCY_RE = re.compile(
    r"(?:₹|rs\.?|inr|usd|eur|gbp|\$|€|£)\s*[0-9][0-9,]*(?:\.[0-9]+)?",
    re.I,
)
_PERCENT_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?\s*%|\b[0-9]+(?:\.[0-9]+)?\s*percent\b", re.I)
_DURATION_RE = re.compile(
    r"[0-9]+(?:\.[0-9]+)?\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?|semesters?)\b",
    re.I,
)
_PLAIN_NUMBER_RE = re.compile(r"(?<![A-Za-z])[0-9]+(?:\.[0-9]+)?(?![A-Za-z])")

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\s*[-/]\s*((?:19|20)\d{2})\b")

# Generic function/grammar words removed when deriving query anchors.
# They are intentionally language-light and domain-neutral: no institution
# names, programs, departments, or document labels belong here.
_QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "for", "to", "in",
    "on", "at", "by", "with", "from", "into", "about", "as", "is",
    "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "can", "could", "would", "should", "may", "might", "will", "what",
    "which", "where", "when", "who", "why", "how", "tell", "me", "please",
    "give", "show", "want", "need", "get", "provide", "there", "this",
    "that", "these", "those", "i", "we", "you", "they", "my", "your",
    "their", "ka", "ki", "ke", "ko", "me", "se", "hai", "hain", "kya",
    "kaise", "kis", "kiske", "mujhe", "batao", "chahiye",
})

_WORD_RE = re.compile(r"[\w]+(?:['-][\w]+)*", re.UNICODE)
_TERMINAL_SENTENCE_RE = re.compile(r"[.!?]\s*$")


# ---------------------------------------------------------------------------
# Public contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CoverageUnit:
    """One semantically required information unit."""

    name: str
    required: bool = True
    importance: float = 1.0
    support: float = 0.0
    covered: bool = False


@dataclass(frozen=True, slots=True)
class CoverageAssessment:
    """Aggregate evidence-completeness assessment."""

    status: CoverageStatus
    score: float
    question_type: str
    strong_documents: int
    partial_documents: int
    relevant_documents: int
    unique_sources: int
    inventory_items: int = 0
    inventory_cues: int = 0
    required_units: tuple[str, ...] = ()
    covered_units: tuple[str, ...] = ()
    uncovered_units: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    units: tuple[CoverageUnit, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "score": self.score,
            "question_type": self.question_type,
            "strong_documents": self.strong_documents,
            "partial_documents": self.partial_documents,
            "relevant_documents": self.relevant_documents,
            "unique_sources": self.unique_sources,
            "inventory_items": self.inventory_items,
            "inventory_cues": self.inventory_cues,
            "required_units": list(self.required_units),
            "covered_units": list(self.covered_units),
            "uncovered_units": list(self.uncovered_units),
            "reasons": list(self.reasons),
            "units": [
                {
                    "name": unit.name,
                    "required": unit.required,
                    "importance": unit.importance,
                    "support": unit.support,
                    "covered": unit.covered,
                }
                for unit in self.units
            ],
        }


# ---------------------------------------------------------------------------
# Normalization / generic helpers
# ---------------------------------------------------------------------------

def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _normalized(value: object) -> str:
    return _text(value).casefold()


def _candidate_source(candidate: RetrievalCandidate) -> str:
    return _text(candidate.source).casefold()


def _content(candidate: RetrievalCandidate) -> str:
    """Return evidence content while preserving structural newlines."""
    document = candidate.document
    if isinstance(document, dict):
        value = document.get("page_content", "")
    else:
        value = getattr(document, "page_content", "")
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _metadata(candidate: RetrievalCandidate) -> dict[str, Any]:
    document = candidate.document
    if isinstance(document, dict):
        value = document.get("metadata", {})
    else:
        value = getattr(document, "metadata", {})
    return dict(value or {}) if isinstance(value, dict) else {}


def _quality_score(candidate: RetrievalCandidate) -> float:
    quality = candidate.quality
    return _clamp01(
        (0.60 * quality.content_quality)
        + (0.40 * quality.structural_quality)
        - (0.50 * quality.noise)
    )


def _support_score(candidate: RetrievalCandidate) -> float:
    """Combine upstream semantic alignment and intrinsic evidence quality."""
    alignment = candidate.alignment
    semantic = _clamp01(
        (0.32 * alignment.semantic_match)
        + (0.28 * alignment.coverage)
        + (0.10 * alignment.topic_match)
        + (0.10 * alignment.intent_match)
        + (0.08 * max(alignment.program_match, alignment.entity_match))
        + (0.07 * alignment.attribute_match)
        + (0.05 * alignment.scope_match)
    )
    return _clamp01((0.78 * semantic) + (0.22 * _quality_score(candidate)))


def _usable(candidate: RetrievalCandidate) -> bool:
    return candidate.quality.usable and not candidate.alignment.conflicts


def _deduplicate(candidates: Sequence[RetrievalCandidate]) -> list[RetrievalCandidate]:
    """Deduplicate by content first, then by stable candidate identity."""
    result: list[RetrievalCandidate] = []
    seen_content: set[tuple[str, str]] = set()
    seen_ids: set[str] = set()

    for candidate in candidates:
        content_key = (_candidate_source(candidate), _normalized(_content(candidate)))
        candidate_id = _text(candidate.document_id)
        if content_key[1] and content_key in seen_content:
            continue
        if candidate_id and candidate_id in seen_ids:
            continue
        if content_key[1]:
            seen_content.add(content_key)
        if candidate_id:
            seen_ids.add(candidate_id)
        result.append(candidate)

    # Duplicate textual evidence from different source paths should not create
    # artificial breadth either.
    globally_unique: list[RetrievalCandidate] = []
    global_content: set[str] = set()
    for candidate in result:
        key = _normalized(_content(candidate))
        if not key or key in global_content:
            if key in global_content:
                continue
        global_content.add(key)
        globally_unique.append(candidate)
    return globally_unique


# ---------------------------------------------------------------------------
# Generic semantic requirement helpers
# ---------------------------------------------------------------------------

def _tokens(text: str) -> tuple[str, ...]:
    values = []
    for token in _WORD_RE.findall(_normalized(text)):
        if len(token) < 3 or token in _QUERY_STOPWORDS:
            continue
        values.append(token)
    return tuple(dict.fromkeys(values))


def _stem_like(token: str) -> str:
    """Small morphology normalizer; not a domain vocabulary."""
    value = token.casefold()
    for suffix in ("ications", "ation", "ments", "ment", "ing", "ed", "es", "s"):
        if len(value) - len(suffix) >= 4 and value.endswith(suffix):
            return value[:-len(suffix)]
    return value


def _anchor_set(texts: Iterable[str]) -> set[str]:
    anchors: set[str] = set()
    for text in texts:
        anchors.update(_stem_like(token) for token in _tokens(text))
    return anchors


def _query_anchor_set(query: Query) -> set[str]:
    texts: list[str] = [query.original_query, query.normalized_query, query.search_query]
    if query.target is not None:
        texts.append(query.target.text)
    texts.extend(intent.name for intent in query.intents)
    texts.extend(entity.name for entity in query.entities)
    texts.extend(concept.name for concept in query.concepts)
    texts.extend(q.name for q in query.qualifiers)
    texts.extend(q.value for q in query.qualifiers)
    texts.extend(c.name for c in query.constraints)
    texts.extend(str(c.value) for c in query.constraints)
    texts.extend(r.relation_type for r in query.relations)
    texts.extend(n.name for n in query.numeric_requirements)
    texts.extend(t.kind for t in query.temporal_constraints)
    return _anchor_set(texts)


def _candidate_anchor_set(candidate: RetrievalCandidate) -> set[str]:
    meaning = candidate.meaning
    texts: list[str] = [_content(candidate), candidate.source]
    for attribute in getattr(meaning, "attributes", ()):
        texts.append(str(attribute))
    if getattr(meaning, "intent", None):
        texts.append(str(meaning.intent))
    for value in getattr(meaning, "topics", ()):
        texts.append(str(value))
    for value in getattr(meaning, "scope", ()):
        texts.append(str(value))
    for value in getattr(meaning, "qualifiers", ()):
        texts.append(str(value))
    for value in getattr(meaning, "constraints", ()):
        texts.append(str(value))
    return _anchor_set(texts)


def _query_anchor_coverage(query: Query, candidate: RetrievalCandidate) -> float:
    query_anchors = _query_anchor_set(query)
    if not query_anchors:
        return 0.0
    candidate_anchors = _candidate_anchor_set(candidate)
    return _clamp01(len(query_anchors & candidate_anchors) / len(query_anchors))


def _local_anchor_coverage(anchors: set[str], window: str) -> float:
    if not anchors:
        return 0.0
    window_anchors = _anchor_set((window,))
    return _clamp01(len(anchors & window_anchors) / len(anchors))



def _primary_intent_name(query: Query) -> str:
    if query.primary_intent is not None:
        return _normalized(query.primary_intent.name)
    return _normalized(query.request_type)


# ---------------------------------------------------------------------------
# Generic question-shape helpers
# ---------------------------------------------------------------------------

_COLLECTION_TARGET_STEMS = frozenset({
    "program", "programme", "degree", "course",
    "department", "departments", "school", "schools",
    "facility", "facilities", "amenity", "amenities",
    "research area", "research areas", "research opportunity",
    "research opportunities", "research theme", "research themes",
    "research group", "research groups",
})

_COLLECTION_PATTERN = re.compile(
    r"\b(?:what|which|list|show|give|tell me(?: about)?)\b"
    r".*\b(?:available|offered|offer|have|provide|exist|there are|does .* have)\b",
    re.I | re.S,
)

_LOCATION_PATTERN = re.compile(
    r"\b(?:where\s+(?:is|are)|location|address|located|situated|how\s+(?:do|can)\s+i\s+reach)\b",
    re.I,
)

_PROCEDURE_PATTERN = re.compile(
    r"\b(?:procedure|process|steps?|how\s+(?:do|can)\s+i\s+apply|how\s+to\s+apply)\b",
    re.I,
)

_PROCEDURE_EVIDENCE_CUES = frozenset({
    "process", "procedure", "step", "steps", "apply", "application",
    "submit", "submission", "portal", "register", "registration",
    "enrol", "enroll", "enrollment", "admission",
})


def _effective_list_item_type(query: Query) -> str:
    item_type = _normalized(query.list_intent.item_type or "")
    if item_type:
        return item_type
    target = _normalized(getattr(query.target, "text", "") if query.target else "")
    if target:
        return target
    return "item"


def _is_collection_question(query: Query) -> bool:
    if query.list_intent.is_list:
        return True

    text = _normalized(query.original_query or query.normalized_query)
    if not text:
        return False

    target = _normalized(getattr(query.target, "text", "") if query.target else "")
    if target:
        target_stem = _stem_like(target)
        if (
            target_stem in {_stem_like(v) for v in _COLLECTION_TARGET_STEMS}
            or any(_stem_like(v) in target_stem for v in _COLLECTION_TARGET_STEMS)
        ) and _COLLECTION_PATTERN.search(text):
            return True

    return bool(_COLLECTION_PATTERN.search(text) and any(
        _stem_like(term) in text for term in _COLLECTION_TARGET_STEMS
    ))


def _is_procedure_question(query: Query) -> bool:
    request_type = _primary_intent_name(query)
    if request_type in {"procedure", "process", "steps"}:
        return True
    return bool(_PROCEDURE_PATTERN.search(
        _normalized(query.original_query or query.normalized_query)
    ))


def _is_location_question(query: Query) -> bool:
    request_type = _primary_intent_name(query)
    if request_type in {"directions", "direction", "location", "navigation"}:
        return True
    return bool(_LOCATION_PATTERN.search(
        _normalized(query.original_query or query.normalized_query)
    ))


def _semantic_inventory_values(
    candidate: RetrievalCandidate,
    *,
    item_type: str,
) -> set[str]:
    """Collect distinct semantic inventory units without inventing entities."""
    meaning = candidate.meaning
    item = _normalized(item_type)

    fields: tuple[str, ...]
    if any(marker in item for marker in ("program", "programme", "degree", "course")):
        fields = ("programs", "entities")
    elif any(marker in item for marker in ("department", "school", "unit")):
        fields = ("entities",)
    elif "research" in item:
        fields = ("entities",)
    elif any(marker in item for marker in ("facility", "facilities", "amenity")):
        fields = ("entities",)
    else:
        fields = ("entities", "programs")

    values: set[str] = set()
    for field_name in fields:
        for value in getattr(meaning, field_name, ()) or ():
            cleaned = _normalized(value)
            if cleaned and len(cleaned) >= 3:
                values.add(cleaned)
    return values


def _collection_alignment_score(
    query: Query,
    candidate: RetrievalCandidate,
) -> float:
    """Measure whether verified evidence belongs to the requested collection."""
    semantic = max(
        candidate.alignment.entity_match,
        candidate.alignment.program_match,
        candidate.alignment.topic_match,
        candidate.alignment.semantic_match,
        candidate.alignment.scope_match,
    )
    lexical = _query_anchor_coverage(query, candidate)
    return _clamp01(max(semantic, lexical))


def _procedure_support(
    query: Query,
    candidate: RetrievalCandidate,
) -> float:
    """Score observable procedural evidence using generic process signals."""
    text = _normalized(_content(candidate))
    tokens = _anchor_set((text,))
    cue_hits = len(tokens & {_stem_like(v) for v in _PROCEDURE_EVIDENCE_CUES})
    cue_score = min(1.0, cue_hits / 2.0)
    structure = min(1.0, _explicit_inventory_items(_content(candidate)) / 2.0)
    lexical = _query_anchor_coverage(query, candidate)
    semantic = max(
        candidate.alignment.intent_match,
        candidate.alignment.topic_match,
        candidate.alignment.semantic_match,
    )
    return _clamp01(
        (0.40 * _support_score(candidate))
        + (0.25 * cue_score)
        + (0.15 * structure)
        + (0.20 * max(lexical, semantic))
    )


def _location_support(query: Query, candidate: RetrievalCandidate) -> float:
    """Score one focused location/directions answer."""
    text = _normalized(_content(candidate))
    location_cues = {
        "located", "location", "address", "situated", "reach",
        "road", "street", "campus", "city", "district", "pin",
        "postal", "pincode", "zip",
    }
    cue_score = 1.0 if _anchor_set((text,)) & {_stem_like(v) for v in location_cues} else 0.0
    semantic = max(
        candidate.alignment.entity_match,
        candidate.alignment.scope_match,
        candidate.alignment.topic_match,
        candidate.alignment.semantic_match,
    )
    lexical = _query_anchor_coverage(query, candidate)
    return _clamp01(
        (0.50 * _support_score(candidate))
        + (0.20 * cue_score)
        + (0.15 * semantic)
        + (0.15 * lexical)
    )


def _is_requirement_request(query: Query) -> bool:
    """Use the query's semantic intent label, not document vocabulary."""
    intent = _primary_intent_name(query)
    return any(marker in intent for marker in ("eligib", "qualif", "require", "criteria"))


def _target_alignment(candidate: RetrievalCandidate) -> float:
    return _clamp01(
        max(
            candidate.alignment.program_match,
            candidate.alignment.entity_match,
            candidate.alignment.scope_match,
        )
    )


def _unit_names(query: Query) -> tuple[tuple[str, float], ...]:
    """Derive coverage units from structured query fields only."""
    units: list[tuple[str, float]] = []

    if query.target is not None:
        units.append(("target", query.target.confidence or 1.0))

    for intent in query.intents:
        units.append((f"intent:{_normalized(intent.name)}", intent.priority or 1.0))

    for entity in query.entities:
        units.append((f"entity:{_normalized(entity.name)}", entity.confidence or 1.0))

    for qualifier in query.qualifiers:
        units.append(
            (
                f"qualifier:{_normalized(qualifier.name)}={_normalized(qualifier.value)}",
                qualifier.importance,
            )
        )

    for constraint in query.constraints:
        units.append(
            (
                f"constraint:{_normalized(constraint.name)}={_normalized(str(constraint.value))}",
                constraint.importance,
            )
        )

    for relation in query.relations:
        units.append(
            (
                f"relation:{_normalized(relation.relation_type)}:{_normalized(relation.subject_ref)}->{_normalized(relation.object_ref)}",
                relation.confidence or 1.0,
            )
        )

    for numeric in query.numeric_requirements:
        units.append(
            (
                f"numeric:{_normalized(numeric.name)}={numeric.value}",
                numeric.confidence or 1.0,
            )
        )

    for temporal in query.temporal_constraints:
        label = temporal.value or f"{temporal.start or ''}:{temporal.end or ''}"
        units.append(
            (
                f"temporal:{_normalized(temporal.kind)}={_normalized(label)}",
                temporal.confidence or 1.0,
            )
        )

    if query.scope is not None:
        if query.scope.scope_id or query.scope.value:
            value = query.scope.scope_id or query.scope.value or "scope"
            units.append((f"scope:{_normalized(value)}", query.scope.confidence or 1.0))

    for target in query.comparison_targets:
        units.append((f"comparison:{_normalized(target.text)}", target.confidence or 1.0))

    if not units:
        units.append(("question", 1.0))

    result: list[tuple[str, float]] = []
    seen: set[str] = set()
    for name, importance in units:
        if name in seen:
            continue
        seen.add(name)
        result.append((name, _clamp01(importance)))
    return tuple(result)


# ---------------------------------------------------------------------------
# Numeric / temporal fact matching
# ---------------------------------------------------------------------------

def _numeric_kind(req_unit: str | None) -> str:
    unit = _normalized(req_unit)
    if not unit:
        return "unknown"
    if unit in {"currency", "money", "monetary", "rs", "inr", "usd", "eur", "gbp"}:
        return "currency"
    if unit in {"percent", "percentage", "%"}:
        return "percent"
    if unit in {"duration", "time"}:
        return "duration"
    if unit in {"count", "number", "quantity"}:
        return "count"
    return unit


def _numeric_fact_windows(text: str) -> list[tuple[str, str]]:
    """Return (kind, local_window) for numeric expressions in the evidence."""
    raw = _text(text)
    facts: list[tuple[str, str]] = []
    for pattern, kind in (
        (_CURRENCY_RE, "currency"),
        (_PERCENT_RE, "percent"),
        (_DURATION_RE, "duration"),
        (_PLAIN_NUMBER_RE, "count"),
    ):
        for match in pattern.finditer(raw):
            start = max(0, match.start() - 80)
            end = min(len(raw), match.end() + 80)
            facts.append((kind, _normalized(raw[start:end])))
    return facts


def _requested_name_is_present(name: str, window: str) -> bool:
    tokens = [token for token in re.findall(r"\b\w+\b", _normalized(name), flags=re.UNICODE) if len(token) > 2]
    if not tokens:
        return True
    return any(re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", window) for token in tokens)


def _numeric_context_anchors(query: Query, requirement: NumericRequirement) -> set[str]:
    """Extract query context that disambiguates a numeric field."""
    texts: list[str] = [query.original_query, query.normalized_query, query.search_query]

    if query.target is not None:
        texts.append(query.target.text)
    texts.extend(entity.name for entity in query.entities)
    texts.extend(entity.aliases for entity in query.entities)
    texts.extend(concept.name for concept in query.concepts)
    texts.extend(qualifier.name for qualifier in query.qualifiers)
    texts.extend(qualifier.value for qualifier in query.qualifiers)
    texts.extend(constraint.name for constraint in query.constraints)
    texts.extend(str(constraint.value) for constraint in query.constraints)
    texts.extend(relation.relation_type for relation in query.relations)

    context = _anchor_set(texts)
    context -= _anchor_set((
        requirement.name,
        requirement.unit or "",
        "fee",
        "fees",
        "cost",
        "costs",
        "charge",
        "charges",
        "amount",
        "price",
        "rate",
        "rates",
    ))
    return context


def _numeric_requirement_supported(query: Query, candidate: RetrievalCandidate, requirement_name: str) -> bool:
    requirement = next(
        (
            item
            for item in query.numeric_requirements
            if _normalized(item.name) == _normalized(requirement_name)
        ),
        None,
    )
    if requirement is None:
        return False

    requested_kind = _numeric_kind(requirement.unit)
    facts = _numeric_fact_windows(_content(candidate))
    if not facts:
        return False

    # The field label itself is one signal.  Additional query context (target,
    # entity, qualifier, constraint, relation) disambiguates fields that share
    # the same numeric label, e.g. multiple kinds of "fee" in one corpus.
    requirement_anchors = _anchor_set((requirement.name,))
    context_anchors = _numeric_context_anchors(query, requirement)
    candidate_meaning_values = _anchor_set(
        list(getattr(candidate.meaning, "attributes", ()))
        + list(getattr(candidate.meaning, "constraints", ()))
        + list(getattr(candidate.meaning, "qualifiers", ()))
    )

    for kind, window in facts:
        if requested_kind == "unknown":
            kind_compatible = kind in {"currency", "percent", "duration", "count"}
        elif requested_kind == "count":
            kind_compatible = kind == "count"
        elif requested_kind == "duration":
            kind_compatible = kind == "duration"
        else:
            kind_compatible = kind == requested_kind

        if not kind_compatible:
            continue

        label_match = _local_anchor_coverage(requirement_anchors, window) >= 0.5
        semantic_label_match = bool(requirement_anchors & candidate_meaning_values)
        context_match = (
            not context_anchors
            or _local_anchor_coverage(context_anchors, window) >= 0.34
        )
        aligned_attribute = candidate.alignment.attribute_match >= UNIT_COVERAGE_THRESHOLD

        # With disambiguating query context, a generic field label is not enough
        # on its own.  The numeric fact must be locally associated with the
        # requested context, or semantically attributed by the upstream frame.
        if semantic_label_match and context_match:
            return True
        if label_match and context_match:
            return True
        if aligned_attribute and context_match and not context_anchors:
            return True

    return False


def _year_signals(text: str) -> tuple[set[int], set[tuple[int, int]]]:
    raw = _normalized(text)
    ranges = {(int(a), int(b)) for a, b in _YEAR_RANGE_RE.findall(raw)}
    years = {int(value) for value in _YEAR_RE.findall(raw)}
    return years, ranges


def _temporal_evidence_matches(query: Query, text: str) -> bool:
    constraints = tuple(query.temporal_constraints or ())
    if not constraints:
        return True

    years, ranges = _year_signals(text)
    raw = _normalized(text)

    for constraint in constraints:
        value = _normalized(constraint.value)
        start = _normalized(constraint.start)
        end = _normalized(constraint.end)

        if start and end and start.isdigit() and end.isdigit():
            if (int(start), int(end)) not in ranges:
                return False
            continue

        if value:
            requested_years = {int(v) for v in _YEAR_RE.findall(value)}
            if requested_years and requested_years & years:
                continue
            if value in raw:
                continue
            return False

    return True


# ---------------------------------------------------------------------------
# Generic list structure
# ---------------------------------------------------------------------------

def _line_looks_like_item(line: str) -> bool:
    cleaned = _text(line)
    if len(cleaned) < 3 or len(cleaned) > 180:
        return False
    if "http://" in cleaned.casefold() or "https://" in cleaned.casefold():
        return False
    if _TERMINAL_SENTENCE_RE.search(cleaned):
        return False
    tokens = _tokens(cleaned)
    return 1 <= len(tokens) <= 18


def _structured_line_runs(text: str) -> int:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    best = 0
    run = 0
    for line in lines:
        explicit = bool(_BULLET_RE.match(line) or _NUMBERED_RE.match(line))
        if explicit or _line_looks_like_item(line):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _explicit_inventory_items(text: str) -> int:
    """Count only evidence that has an explicit enumerable structure.

    This is deliberately conservative. Plain newline-separated prose is not
    automatically interpreted as dozens of list items because web extractors
    often flatten headings/navigation into one item per line.
    """
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return 0

    bullet_count = len(_BULLET_RE.findall(raw))
    numbered_count = len(_NUMBERED_RE.findall(raw))

    table_rows = [
        line
        for line in raw.splitlines()
        if _TABLE_ROW_RE.match(line)
    ]
    table_count = len(table_rows)
    if table_count >= 2:
        # A Markdown table has a header row; do not let the header itself
        # manufacture an inventory entry.
        table_count = max(0, table_count - 1)
    else:
        table_count = 0

    semicolon_items = [
        part
        for part in _SEMICOLON_SPLIT_RE.split(raw)
        if part.strip()
    ]
    compact = [
        part
        for part in semicolon_items
        if 2 <= len(_tokens(part)) <= 18
        and not _TERMINAL_SENTENCE_RE.search(_text(part))
    ]
    semicolon_count = len(compact) if len(compact) >= 2 else 0

    return max(
        bullet_count,
        numbered_count,
        table_count,
        semicolon_count,
    )


def _inventory_items(
    candidate: RetrievalCandidate,
    *,
    item_type: str | None = None,
) -> int:
    """Estimate observable list breadth without interpreting domain data.

    Priority:
      1. explicit enumerable structure in the document;
      2. semantic presence as one conservative inventory unit.

    The second path intentionally returns *one* unit. It prevents a flattened
    web page from turning every extracted line into a fake inventory of items.
    Detailed breadth requires an actual list structure.
    """
    text = _content(candidate)
    explicit = _explicit_inventory_items(text)
    if explicit > 0:
        return explicit

    meaning = candidate.meaning
    meaning_values: list[str] = []
    for field_name in (
        "programs",
        "entities",
        "topics",
        "attributes",
        "scope",
        "qualifiers",
        "constraints",
    ):
        values = getattr(meaning, field_name, ()) or ()
        meaning_values.extend(str(value) for value in values if str(value).strip())

    # A relevant semantic representation establishes that at least one
    # inventory element exists, but never fabricates multiplicity.
    anchors = _anchor_set((
        text,
        candidate.source,
        str(item_type or ""),
    ))
    if meaning_values or anchors:
        return 1

    return 0


def _has_structural_container_heading(text: str, item_type: str | None = None) -> bool:
    """Detect a generic section/list container heading.

    The detector intentionally reasons from document punctuation/shape rather
    than a vocabulary of institutional categories. A heading such as
    ``Programs:`` or ``Research Themes:`` is structural evidence; its actual
    domain meaning comes from the query and registry.
    """
    normalized_type = _stem_like(_text(item_type)) if item_type else ""
    for line in str(text or "").replace("\r", "").split("\n"):
        cleaned = _text(line)
        if not cleaned or len(cleaned) > 100:
            continue
        if not cleaned.endswith(":"):
            continue
        heading = cleaned[:-1].strip()
        if not heading or _TERMINAL_SENTENCE_RE.search(heading):
            continue
        if normalized_type:
            heading_anchors = _anchor_set((heading,))
            type_anchors = _anchor_set((normalized_type,))
            if heading_anchors & type_anchors:
                return True
        # A colon-only structural heading is still valid evidence when the
        # candidate already contains an explicit list marker.
        if len(_tokens(heading)) <= 6:
            return True
    return False


def _inventory_cues(text: str) -> int:
    normalized = _normalized(text)
    return sum(1 for pattern in _EXHAUSTIVE_LIST_PATTERNS if pattern.search(normalized))


def _structural_list_strength(
    candidate: RetrievalCandidate,
    *,
    item_type: str | None = None,
) -> float:
    """Measure observable list structure, not domain-specific content."""
    text = _content(candidate)
    explicit_items = _explicit_inventory_items(text)
    cues = _inventory_cues(text)

    metadata = _metadata(candidate)
    heading_values = []
    for key in ("heading", "section", "title", "page_title", "document_title"):
        value = metadata.get(key)
        if value:
            heading_values.append(_text(value))

    run = _structured_line_runs(text)

    # A plain line run is not treated as an explicit inventory. It can only
    # provide a small structural hint; this is important for flattened HTML.
    heading_signal = bool(heading_values)
    heading_bonus = 0.12 if heading_signal else 0.0
    explicit_score = min(1.0, explicit_items / 2.0)
    cue_score = min(1.0, cues / 1.0)
    run_hint = 0.10 if run >= LIST_STRUCTURE_RUN_MIN else 0.0
    semantic_hint = 0.05 if _inventory_items(candidate, item_type=item_type) else 0.0

    return _clamp01(
        (0.60 * explicit_score)
        + (0.20 * cue_score)
        + heading_bonus
        + run_hint
        + semantic_hint
    )


# ---------------------------------------------------------------------------
# Per-unit support
# ---------------------------------------------------------------------------

def _reference_texts(query: Query, reference: str) -> tuple[str, ...]:
    """Resolve a query relation reference to available textual aliases."""
    ref = _normalized(reference)
    values: list[str] = []

    if query.target is not None:
        if ref in {_normalized(query.target.entity_id), _normalized(query.target.text)}:
            values.append(query.target.text)

    for entity in query.entities:
        if ref in {_normalized(entity.entity_id), _normalized(entity.name)}:
            values.append(entity.name)
            values.extend(entity.aliases)

    for target in query.comparison_targets:
        if ref in {_normalized(target.entity_id), _normalized(target.text)}:
            values.append(target.text)

    if not values and reference:
        values.append(reference)

    return tuple(dict.fromkeys(_text(value) for value in values if _text(value)))


def _relation_unit_support(
    query: Query,
    candidate: RetrievalCandidate,
    unit_name: str,
    base_support: float,
) -> float:
    """Check whether a requested relation is locally represented in evidence."""
    prefix = "relation:"
    raw = unit_name[len(prefix):] if unit_name.startswith(prefix) else unit_name
    relation_type, separator, refs = raw.partition(":")
    if not separator:
        return 0.0
    subject_ref, separator, object_ref = refs.partition("->")
    if not separator:
        return 0.0

    subject_texts = _reference_texts(query, subject_ref)
    object_texts = _reference_texts(query, object_ref)
    relation_tokens = _anchor_set((relation_type,))

    if not subject_texts or not object_texts:
        return 0.0

    content = _content(candidate)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", content)
    best = 0.0

    for sentence in sentences:
        sentence_anchors = _anchor_set((sentence,))
        subject_match = any(
            _anchor_set((value,)) & sentence_anchors
            for value in subject_texts
        )
        object_match = any(
            _anchor_set((value,)) & sentence_anchors
            for value in object_texts
        )
        relation_match = bool(relation_tokens & sentence_anchors)

        if subject_match and object_match:
            local = 0.65 + (0.20 if relation_match else 0.0)
            local += 0.15 * max(
                candidate.alignment.semantic_match,
                candidate.alignment.coverage,
            )
            best = max(best, _clamp01(local))

    if best <= 0.0:
        return 0.0
    return max(best, _clamp01(base_support * 0.75))


def _unit_support(query: Query, candidate: RetrievalCandidate, unit_name: str, base_support: float) -> float:
    alignment = candidate.alignment

    if unit_name == "target":
        return _target_alignment(candidate)

    if unit_name.startswith("intent:"):
        return _clamp01(max(alignment.intent_match, alignment.topic_match, alignment.semantic_match))

    if unit_name.startswith("entity:"):
        return _clamp01(max(alignment.entity_match, alignment.program_match, alignment.semantic_match))

    if unit_name.startswith("scope:"):
        return _clamp01(max(alignment.scope_match, alignment.entity_match, alignment.program_match))

    if unit_name.startswith("qualifier:") or unit_name.startswith("constraint:"):
        return _clamp01(max(alignment.attribute_match, alignment.scope_match, alignment.topic_match, alignment.semantic_match))

    if unit_name.startswith("relation:"):
        return _relation_unit_support(
            query,
            candidate,
            unit_name,
            base_support,
        )

    if unit_name.startswith("numeric:"):
        name = unit_name[len("numeric:") :].split("=", 1)[0]
        return _clamp01(base_support if _numeric_requirement_supported(query, candidate, name) else 0.0)

    if unit_name.startswith("temporal:"):
        return _clamp01(base_support if _temporal_evidence_matches(query, _content(candidate)) else 0.0)

    if unit_name.startswith("comparison:"):
        return _clamp01(max(alignment.entity_match, alignment.program_match, alignment.semantic_match))

    return base_support


def _build_unit_assessment(query: Query, candidates: Sequence[RetrievalCandidate]) -> tuple[CoverageUnit, ...]:
    units: list[CoverageUnit] = []
    for name, importance in _unit_names(query):
        best = 0.0
        for candidate in candidates:
            if not _usable(candidate):
                continue
            base = _support_score(candidate)
            best = max(best, _unit_support(query, candidate, name, base))
        units.append(
            CoverageUnit(
                name=name,
                importance=importance,
                support=_clamp01(best),
                covered=best >= UNIT_COVERAGE_THRESHOLD,
            )
        )
    return tuple(units)


# ---------------------------------------------------------------------------
# Question type
# ---------------------------------------------------------------------------

def _question_type(query: Query) -> str:
    if _is_collection_question(query):
        return "list"
    if query.is_multi_part or query.is_comparison:
        return "multi_part"
    if _is_requirement_request(query):
        return "requirements"
    if _is_procedure_question(query):
        return "procedure"
    if _is_location_question(query):
        return "location"
    if query.numeric_requirements:
        return "quantitative"
    if query.temporal_constraints:
        return "temporal"
    return "descriptive"


# ---------------------------------------------------------------------------
# Main assessment
# ---------------------------------------------------------------------------

def assess_coverage(
    query: Query,
    candidates: Sequence[RetrievalCandidate] | None,
) -> CoverageAssessment:
    """Assess semantic completeness of the surviving evidence."""
    if not isinstance(query, Query):
        raise TypeError("query must be a Query instance")

    unique_candidates = _deduplicate(list(candidates or ()))
    usable_candidates = [candidate for candidate in unique_candidates if _usable(candidate)]
    question_type = _question_type(query)

    if not usable_candidates:
        return CoverageAssessment(
            status="insufficient",
            score=0.0,
            question_type=question_type,
            strong_documents=0,
            partial_documents=0,
            relevant_documents=0,
            unique_sources=0,
            reasons=("no usable evidence candidates remain",),
        )

    supports = [_support_score(candidate) for candidate in usable_candidates]
    strong_count = sum(score >= SUPPORTED_THRESHOLD for score in supports)
    partial_count = sum(score >= PARTIAL_THRESHOLD for score in supports)
    relevant_candidates = [
        candidate for candidate, score in zip(usable_candidates, supports)
        if score >= FOCUSED_MIN_SUPPORT
    ]
    source_count = len(
        {
            _candidate_source(candidate)
            for candidate in relevant_candidates
            if _candidate_source(candidate)
        }
    )

    # ---------------------------------------------------------
    # LIST
    # ---------------------------------------------------------
    if question_type == "list":
        # Coverage runs after candidate verification. For collection questions,
        # retain all verified candidates that are plausibly aligned with the
        # requested collection instead of letting the generic support-score
        # threshold erase breadth evidence.
        list_candidates = []
        relevant_ids = {id(candidate) for candidate in relevant_candidates}
        for candidate in usable_candidates:
            if id(candidate) in relevant_ids or _collection_alignment_score(query, candidate) >= 0.34:
                list_candidates.append(candidate)
        if not list_candidates:
            list_candidates = list(relevant_candidates)
        list_source_count = len({
            _candidate_source(candidate)
            for candidate in list_candidates
            if _candidate_source(candidate)
        })
        item_type = _effective_list_item_type(query)

        per_candidate_items = [
            _inventory_items(candidate, item_type=item_type)
            for candidate in list_candidates
        ]
        inventory_items = sum(per_candidate_items)
        inventory_cues = sum(
            _inventory_cues(_content(candidate))
            for candidate in list_candidates
        )

        explicit_inventory = sum(
            _explicit_inventory_items(_content(candidate))
            for candidate in list_candidates
        )

        semantic_inventory: set[str] = set()
        for candidate in list_candidates:
            semantic_inventory.update(
                _semantic_inventory_values(
                    candidate,
                    item_type=item_type,
                )
            )

        semantic_inventory_count = len(semantic_inventory)
        inventory_items = max(inventory_items, semantic_inventory_count)

        # Duplicate evidence has already been removed at candidate level.
        # This field is retained only for diagnostics/provenance and must not
        # create completeness by itself.
        unique_list_contents = {
            _normalized(_content(candidate))
            for candidate in list_candidates
            if _normalized(_content(candidate))
        }

        min_items = query.list_intent.min_items
        requested_met = (
            min_items is not None
            and explicit_inventory >= min_items
        )

        strong_single = any(
            _support_score(candidate) >= SUPPORTED_THRESHOLD
            and _explicit_inventory_items(_content(candidate)) >= LIST_ITEM_MIN_FOR_STRONG_SINGLE_SOURCE
            and bool(
                _inventory_cues(_content(candidate))
                or _has_structural_container_heading(_content(candidate), item_type)
            )
            for candidate in list_candidates
        )

        distributed = (
            list_source_count >= LIST_MIN_SOURCE_COUNT_FOR_DISTRIBUTED
            and explicit_inventory >= max(LIST_ITEM_MIN_FOR_PARTIAL, 2)
        )

        # Flattened web extraction often loses bullet/list punctuation while
        # semantic retrieval still preserves distinct inventory entities.
        # Distinct semantic entities across multiple sources are valid breadth
        # evidence, provided the query is a collection request.
        semantic_single = (
            strong_count >= 1
            and semantic_inventory_count >= 4
        )
        semantic_distributed = (
            list_source_count >= LIST_MIN_SOURCE_COUNT_FOR_DISTRIBUTED
            and semantic_inventory_count >= 3
        )

        # Some retrievers preserve strong collection relevance across many
        # documents while the document-level semantic registry does not carry
        # a distinct item for every chunk.  For a generic collection question,
        # broad, multi-source, semantically aligned evidence is itself a valid
        # completeness signal.  This does not name or enumerate any
        # institution-specific items.
        collection_breadth = (
            list_source_count >= 3
            and len(list_candidates) >= 6
            and any(
                max(
                    candidate.alignment.entity_match,
                    candidate.alignment.program_match,
                    candidate.alignment.topic_match,
                    candidate.alignment.semantic_match,
                ) >= UNIT_COVERAGE_THRESHOLD
                for candidate in list_candidates
            )
        )

        if not list_candidates:
            status: CoverageStatus = "insufficient"
            score = 0.0
            reasons = ("no relevant list evidence remains",)
        elif requested_met or strong_single or distributed or semantic_single or semantic_distributed or collection_breadth:
            status = "supported"
            score = _clamp01(
                max(
                    (_support_score(candidate) for candidate in list_candidates),
                    default=0.0,
                )
            )
            reasons = ("evidence exposes a sufficiently structured or semantically distributed collection",)
        elif inventory_items >= LIST_ITEM_MIN_FOR_PARTIAL or inventory_cues:
            status = "partial"
            score = _clamp01(
                max(
                    (_support_score(candidate) for candidate in list_candidates),
                    default=0.0,
                )
                * 0.90
            )
            reasons = (
                "relevant list evidence exists but completeness is not established",
            )
        else:
            status = "insufficient"
            score = max(
                (_support_score(candidate) for candidate in list_candidates),
                default=0.0,
            )
            reasons = ("list evidence lacks enough observable structure",)

        return CoverageAssessment(
            status=status,
            score=score,
            question_type=question_type,
            strong_documents=strong_count,
            partial_documents=partial_count,
            relevant_documents=len(list_candidates),
            unique_sources=list_source_count,
            inventory_items=inventory_items,
            inventory_cues=inventory_cues,
            reasons=reasons,
        )

    # ---------------------------------------------------------
    # MULTI-PART / COMPARISON
    # ---------------------------------------------------------
    if query.is_multi_part or query.is_comparison:
        units = _build_unit_assessment(query, usable_candidates)
        required = tuple(unit.name for unit in units if unit.required)
        covered = tuple(unit.name for unit in units if unit.required and unit.covered)
        uncovered = tuple(unit.name for unit in units if unit.required and not unit.covered)

        allowed_missing = query.retrieval_requirement.max_unmatched_required_facets
        missing_required = len(uncovered)
        total_weight = sum(unit.importance for unit in units if unit.required) or 1.0
        covered_weight = sum(unit.importance for unit in units if unit.required and unit.covered)
        score = _clamp01(covered_weight / total_weight)

        if missing_required == 0:
            status = "supported"
            reasons = ("all required information units are covered",)
        elif missing_required <= allowed_missing and covered:
            status = "partial"
            reasons = ("some required information units are not covered",)
        else:
            status = "insufficient"
            reasons = ("required information units remain uncovered",)

        return CoverageAssessment(
            status=status,
            score=score,
            question_type=question_type,
            strong_documents=strong_count,
            partial_documents=partial_count,
            relevant_documents=len(relevant_candidates),
            unique_sources=source_count,
            required_units=required,
            covered_units=covered,
            uncovered_units=uncovered,
            reasons=reasons,
            units=units,
        )

    # ---------------------------------------------------------
    # REQUIREMENTS
    # ---------------------------------------------------------
    if question_type == "requirements":
        # A candidate must be both semantically aligned *and* expose local
        # evidence connected to the query's own semantic anchors. This avoids
        # treating an aligned deadline/overview paragraph as proof of an
        # eligibility requirement without maintaining a hardcoded list of
        # institution-specific requirement words.
        qualifying: list[RetrievalCandidate] = []
        query_anchors = _query_anchor_set(query)

        for candidate in relevant_candidates:
            target_ok = _target_alignment(candidate) >= UNIT_COVERAGE_THRESHOLD if query.target else True
            request_signal = max(
                candidate.alignment.intent_match,
                candidate.alignment.attribute_match,
                candidate.alignment.topic_match,
                candidate.alignment.semantic_match,
            )
            lexical_anchor = _query_anchor_coverage(query, candidate)
            # If the query exposes semantic anchors, at least one meaningful
            # query anchor must survive in the candidate. If the query has no
            # anchors, fall back to upstream semantic alignment.
            anchor_gate = lexical_anchor >= 0.18 if query_anchors else request_signal >= UNIT_COVERAGE_THRESHOLD
            if target_ok and request_signal >= UNIT_COVERAGE_THRESHOLD and anchor_gate:
                qualifying.append(candidate)

        if not qualifying:
            status = "insufficient" if not relevant_candidates else "partial"
            score = max(supports, default=0.0)
            reasons = (
                "relevant evidence lacks direct query-linked support for the requested requirements",
            )
        else:
            best = max(
                _clamp01(
                    0.65 * _support_score(candidate)
                    + 0.35 * _query_anchor_coverage(query, candidate)
                )
                for candidate in qualifying
            )
            status = "supported" if best >= FOCUSED_MIN_SUPPORT else "partial"
            score = best
            reasons = (
                "retrieved evidence is semantically aligned and linked to the query's requested information anchors",
            )

        return CoverageAssessment(
            status=status,
            score=_clamp01(score),
            question_type=question_type,
            strong_documents=strong_count,
            partial_documents=partial_count,
            relevant_documents=len(relevant_candidates),
            unique_sources=source_count,
            reasons=reasons,
        )

    # ---------------------------------------------------------
    # PROCEDURE / PROCESS
    # ---------------------------------------------------------
    if question_type == "procedure":
        procedure_candidates = list(usable_candidates)
        scores = [
            _procedure_support(query, candidate)
            for candidate in procedure_candidates
        ]

        if not scores:
            return CoverageAssessment(
                status="insufficient",
                score=0.0,
                question_type=question_type,
                strong_documents=strong_count,
                partial_documents=partial_count,
                relevant_documents=0,
                unique_sources=0,
                reasons=("no usable procedural evidence remains",),
            )

        best = max(scores)
        procedure_shape = any(
            len(
                _anchor_set((_content(candidate),))
                & {_stem_like(v) for v in _PROCEDURE_EVIDENCE_CUES}
            ) >= 2
            for candidate in procedure_candidates
        )
        procedure_semantic = any(
            max(
                candidate.alignment.intent_match,
                candidate.alignment.topic_match,
                candidate.alignment.semantic_match,
                candidate.alignment.attribute_match,
            ) >= UNIT_COVERAGE_THRESHOLD
            for candidate in procedure_candidates
        )

        # Candidate verification has already established compatibility before
        # coverage runs.  For a focused procedure request, one verified
        # candidate with clear procedural cues and adequate semantic alignment
        # is complete evidence even when lexical overlap is small.
        focused_procedure_supported = procedure_shape and procedure_semantic

        if best >= SUPPORTED_THRESHOLD or focused_procedure_supported:
            status = "supported"
            reasons = ("procedural evidence directly covers the requested process",)
        elif best >= PARTIAL_THRESHOLD:
            status = "partial"
            reasons = ("procedural evidence exists but is not strongly complete",)
        else:
            status = "insufficient"
            reasons = ("retrieved evidence does not sufficiently expose the requested process",)

        return CoverageAssessment(
            status=status,
            score=_clamp01(best),
            question_type=question_type,
            strong_documents=strong_count,
            partial_documents=partial_count,
            relevant_documents=len(procedure_candidates),
            unique_sources=len({
                _candidate_source(candidate)
                for candidate in procedure_candidates
                if _candidate_source(candidate)
            }),
            reasons=reasons,
        )

    # ---------------------------------------------------------
    # LOCATION / DIRECTIONS
    # ---------------------------------------------------------
    if question_type == "location":
        location_candidates = list(usable_candidates)
        scores = [
            _location_support(query, candidate)
            for candidate in location_candidates
        ]

        if not scores:
            return CoverageAssessment(
                status="insufficient",
                score=0.0,
                question_type=question_type,
                strong_documents=strong_count,
                partial_documents=partial_count,
                relevant_documents=0,
                unique_sources=0,
                reasons=("no usable location evidence remains",),
            )

        best = max(scores)
        focused_location_supported = any(
            max(
                candidate.alignment.entity_match,
                candidate.alignment.scope_match,
                candidate.alignment.topic_match,
                candidate.alignment.semantic_match,
            ) >= UNIT_COVERAGE_THRESHOLD
            and bool(
                _anchor_set((_content(candidate),))
                & {
                    _stem_like(v)
                    for v in (
                        "located", "location", "address", "situated",
                        "road", "street", "campus", "city", "district",
                        "pin", "postal", "pincode", "zip",
                    )
                }
            )
            for candidate in location_candidates
        )

        # A focused location question is satisfied by one compatible location
        # fact.  Requiring the fact's own location cue prevents generic
        # semantically related documents from manufacturing support.
        if best >= PARTIAL_THRESHOLD:
            status = "supported" if best >= SUPPORTED_THRESHOLD or strong_count >= 1 or focused_location_supported else "partial"
            reasons = ("focused location evidence directly addresses the request",)
        elif focused_location_supported:
            status = "supported"
            reasons = ("focused location evidence directly addresses the request",)
        else:
            status = "insufficient"
            reasons = ("retrieved evidence does not sufficiently establish the requested location",)

        return CoverageAssessment(
            status=status,
            score=_clamp01(best),
            question_type=question_type,
            strong_documents=strong_count,
            partial_documents=partial_count,
            relevant_documents=len(location_candidates),
            unique_sources=len({
                _candidate_source(candidate)
                for candidate in location_candidates
                if _candidate_source(candidate)
            }),
            reasons=reasons,
        )

    # ---------------------------------------------------------
    # QUANTITATIVE / TEMPORAL
    # ---------------------------------------------------------
    if question_type in {"quantitative", "temporal"}:
        matching: list[RetrievalCandidate] = []
        if query.numeric_requirements:
            for candidate in relevant_candidates:
                if all(
                    _numeric_requirement_supported(query, candidate, _normalized(req.name))
                    for req in query.numeric_requirements
                ):
                    matching.append(candidate)
        else:
            matching = [
                candidate
                for candidate in relevant_candidates
                if _temporal_evidence_matches(query, _content(candidate))
            ]

        if not matching:
            status = "partial" if relevant_candidates else "insufficient"
            reasons = ("relevant evidence exists but the requested quantitative/temporal fact is not covered",)
            score = max((_support_score(c) for c in relevant_candidates), default=0.0) * 0.7
        else:
            best = max(_support_score(candidate) for candidate in matching)
            status = "supported" if best >= FOCUSED_MIN_SUPPORT else "partial"
            reasons = ("requested quantitative/temporal evidence is explicitly supported",)
            score = best

        return CoverageAssessment(
            status=status,
            score=_clamp01(score),
            question_type=question_type,
            strong_documents=strong_count,
            partial_documents=partial_count,
            relevant_documents=len(relevant_candidates),
            unique_sources=source_count,
            reasons=reasons,
        )

    # ---------------------------------------------------------
    # DESCRIPTIVE / FOCUSED
    # ---------------------------------------------------------
    units = _build_unit_assessment(query, usable_candidates)
    if any(not unit.covered for unit in units if unit.required):
        # An explicitly structured target/entity/constraint should never be
        # treated as complete solely because another dimension scored highly.
        required_units = tuple(unit.name for unit in units if unit.required)
        covered_units = tuple(unit.name for unit in units if unit.required and unit.covered)
        uncovered_units = tuple(unit.name for unit in units if unit.required and not unit.covered)
        weighted_total = sum(unit.importance for unit in units if unit.required) or 1.0
        weighted_covered = sum(unit.importance for unit in units if unit.required and unit.covered)
        score = _clamp01(weighted_covered / weighted_total)
        status: CoverageStatus = "partial" if covered_units else "insufficient"
        return CoverageAssessment(
            status=status,
            score=score,
            question_type=question_type,
            strong_documents=strong_count,
            partial_documents=partial_count,
            relevant_documents=len(relevant_candidates),
            unique_sources=source_count,
            required_units=required_units,
            covered_units=covered_units,
            uncovered_units=uncovered_units,
            reasons=("structured query units are not fully supported by the evidence",),
            units=units,
        )

    best = max(supports, default=0.0)
    if strong_count >= 1:
        status = "supported"
        reasons = ("at least one strongly aligned evidence candidate is present",)
    elif partial_count >= 1:
        status = "partial"
        reasons = ("relevant evidence exists but is not strongly complete",)
    else:
        status = "insufficient"
        reasons = ("no sufficiently relevant evidence candidate is present",)

    return CoverageAssessment(
        status=status,
        score=best,
        question_type=question_type,
        strong_documents=strong_count,
        partial_documents=partial_count,
        relevant_documents=len(relevant_candidates),
        unique_sources=source_count,
        reasons=reasons,
        units=units,
    )


def is_coverage_sufficient(assessment: CoverageAssessment) -> bool:
    """Return True only when the evidence is fully supportable."""
    if not isinstance(assessment, CoverageAssessment):
        raise TypeError("assessment must be a CoverageAssessment")
    return assessment.status == "supported"


assess_evidence_coverage = assess_coverage


__all__ = [
    "CoverageStatus",
    "CoverageUnit",
    "CoverageAssessment",
    "assess_coverage",
    "assess_evidence_coverage",
    "is_coverage_sufficient",
]