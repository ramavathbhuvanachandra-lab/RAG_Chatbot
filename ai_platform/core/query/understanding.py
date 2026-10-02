"""Generic semantic query understanding for the reusable assistant core.

This module consolidates the useful behavior from the legacy semantic layers:

* structured semantic interpretation;
* deterministic, conservative retrieval-text construction;
* literal preservation of unknown terms;
* structural validation before semantic retrieval is trusted;
* an optional second safety check only when uncertainty warrants it;
* fail-safe fallback to the original user wording.

The module is institution-agnostic. It does not import an institution package
and it never decides factual answers.

Migration boundary
------------------
The legacy implementation exposed ``SemanticFrame``, ``QueryUnderstanding``
and ``SemanticQueryFrame``-style data. The new public contract is ``Query``
from ``backend.core.query.models``. A small compatibility helper
``understand_query_frame`` is retained so the migration can proceed without
breaking downstream code prematurely.
"""

from __future__ import annotations

from functools import lru_cache
import json
import re
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ai_platform.core.query.models import (
    Ambiguity,
    Constraint,
    Entity,
    EntityMention,
    Intent,
    ListIntent,
    Query,
    QueryFacet,
    Qualifier,
    RetrievalRequirement,
    SemanticQueryFrame,
    Target,
)


# ---------------------------------------------------------------------------
# Stable limits learned from the legacy implementation.
# ---------------------------------------------------------------------------

MAX_FACETS = 12
MAX_LIST_ITEMS = 16
MAX_PRESERVED_TERMS = 24
MAX_SEMANTIC_QUERY_TOKENS = 24

LIST_PATTERNS: tuple[str, ...] = (
    r"\bwhat all\b",
    r"\bwhich all\b",
    r"\blist\b",
    r"\btypes?\b",
    r"\bcategories?\b",
    r"\bmodes?\b",
    r"\boptions?\b",
    r"\bkaun kaun\b",
    r"\bkon kon\b",
    r"\bkitne\b",
    r"\bkitni\b",
)

# Strong generic lexical cues are used only as a deterministic backstop.
# They are institution-agnostic: they describe common information-seeking
# language, not any particular college, program, or policy.
REQUEST_TYPE_BACKSTOP_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "eligibility",
        r"\b(?:eligibility|eligible|qualification|qualifications|qualifying|criteria)\b",
    ),
    (
        "procedure",
        r"\b(?:how\s+(?:do|can)\s+i\s+apply|application\s+(?:process|procedure)|procedure\s+for\s+applying|process\s+for\s+applying)\b",
    ),
    (
        "cost",
        r"\b(?:fee|fees|tuition|cost|costs|charge|charges|price|amount|rent|stipend|payment|payments)\b",
    ),
    (
        "directions",
        r"\b(?:where\s+is|where\s+are|located|location|address|directions?|how\s+(?:do|can)\s+i\s+(?:reach|get\s+to))\b",
    ),
    (
        "comparison",
        r"\b(?:compare|comparison|difference(?:s)?\s+between|versus|vs\.?|better\s+than)\b",
    ),
    (
        "registration",
        r"\b(?:registration|register|enrol(?:lment)?|enrollment)\b",
    ),
    (
        "research",
        r"\b(?:research\s+(?:area|areas|opportunit(?:y|ies)|theme|themes)|research)\b",
    ),
    (
        "admission",
        r"\b(?:application\s+deadlines?|admission|admissions|admit|entry)\b",
    ),
)

# Generic institutional targets. The longest literal match wins. These are
# semantic nouns/properties, not institution-specific entities.
GENERIC_TARGET_PATTERNS: tuple[str, ...] = (
    r"\bacademic\s+programs?\b",
    r"\badmission\s+requirements?\b",
    r"\badmission\s+(?:fees?|charges?)\b",
    r"\badmission\s+(?:process|procedure)\b",
    r"\bapplication\s+(?:deadlines?|process|procedure)\b",
    r"\bresearch\s+opportunit(?:y|ies)\b",
    r"\bresearch\s+areas?\b",
    r"\bresearch\s+(?:themes?|topics?)\b",
    r"\bhostel\s+facilities?\b",
    r"\bhostel\s+fees?\b",
    r"\btuition\s+fees?\b",
    r"\beligibility\s+requirements?\b",
    r"\bapplication\s+deadlines?\b",
    r"\bdepartments?\b",
    r"\bprograms?\b",
    r"\bfacilities?\b",
    r"\bfees?\b",
    r"\beligibility\b",
    r"\bapplications?\b",
    r"\badmission\b",
    r"\binstitute\b",
)

REQUESTED_ATTRIBUTE_LABELS: tuple[tuple[str, str], ...] = (
    ("eligibility", "eligibility"),
    ("procedure", "procedure"),
    ("cost", "fee"),
    ("directions", "location"),
    ("registration", "registration"),
    ("research", "research"),
    ("admission", "admission"),
)

VALID_INTERPRETATION_STATUSES = {
    "trusted",
    "review",
    "rejected",
}


# ---------------------------------------------------------------------------
# LLM protocols
# ---------------------------------------------------------------------------

class StructuredModel(Protocol):
    """Minimal interface needed by the semantic understanding stage."""

    def invoke(self, input: Any) -> Any:
        ...


class StructuredModelFactory(Protocol):
    """Protocol for LangChain-style structured-output capable models."""

    def with_structured_output(self, schema: Any, **kwargs: Any) -> StructuredModel:
        ...


# ---------------------------------------------------------------------------
# Internal structured-output contracts.
# ---------------------------------------------------------------------------

class FacetPayload(BaseModel):
    """LLM-produced named semantic facet."""

    model_config = ConfigDict(extra="ignore")

    name: str
    value: str
    required: bool = True
    importance: float = Field(default=1.0, ge=0.0, le=1.0)


class SemanticInterpretation(BaseModel):
    """Model output describing the user's information need only."""

    model_config = ConfigDict(extra="ignore")

    semantic_query: str = ""
    target: str | None = None
    request_type: str | None = None
    facets: list[FacetPayload] = Field(default_factory=list)
    qualifiers: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    temporal_context: list[str] = Field(default_factory=list)
    comparison_targets: list[str] = Field(default_factory=list)
    preserved_terms: list[str] = Field(default_factory=list)
    is_list_question: bool = False
    is_comparison_question: bool = False
    is_multi_part: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_clarification: bool = False
    clarification_reason: str = ""
    language: str | None = None
    ambiguous: bool = False
    unknown_terms: list[str] = Field(default_factory=list)


class SemanticSafetyCheck(BaseModel):
    """Read-only semantic safety decision; it cannot rewrite the query."""

    model_config = ConfigDict(extra="ignore")

    approved: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SEMANTIC_SYSTEM_PROMPT = r"""
You are the semantic query-understanding component of a retrieval system.

Your job is ONLY to understand what the user is asking.

Do NOT answer the question.
Do NOT retrieve information.
Do NOT invent institutional facts.
Do NOT assume facts that are absent from the user's wording.

Preserve:
- the actual target/topic;
- person/category qualifiers;
- program or object expressions;
- requested attributes/actions;
- quantities and units;
- duration and time constraints;
- important conditions;
- relationships between requested concepts;
- list intent;
- comparison intent;
- unknown/opaque user terms when they truly cannot be interpreted safely.

For Hindi, Hinglish, Romanized Hindi, English, mixed-language, slang,
spelling variation, and informal wording:
- infer meaning only when reasonably supported by the wording;
- preserve unusual literal terms;
- do not invent a canonical institutional entity name.

Unknown-term rule:
Mark a term as unknown only when its meaning cannot reasonably be inferred
from the wording. Do not mark ordinary nouns, program names, categories, or
normal question words as unknown merely because the institution-specific
details are unknown.

For exact factual requests, preserve both the requested target and the
requested attribute. Do not collapse a detailed question into a generic
label such as "student", "registration", or "information".

Semantic paraphrasing is allowed, but factual interpretation must come from
the user wording alone.
""".strip()

SEMANTIC_SAFETY_PROMPT_TEMPLATE = r"""
Check whether the semantic interpretation below is supported by the user's
wording alone.

Do not use institutional knowledge.
Do not answer the user's question.
Do not rewrite the interpretation.
Do not add facts.

Reject the interpretation when:
- an opaque user phrase is assigned a specific unsupported meaning;
- an important user condition is dropped;
- the target/topic is invented rather than supported by the query;
- a specific institutional fact is introduced that the user did not state.

A normal paraphrase is valid. A program name or ordinary noun is not
automatically an unknown term.

USER QUERY:
{query}

SEMANTIC INTERPRETATION:
{payload}
""".strip()


# ---------------------------------------------------------------------------
# Generic normalization helpers.
# ---------------------------------------------------------------------------

def normalize_text(value: object) -> str:
    """Collapse whitespace without otherwise rewriting user wording."""
    return re.sub(r"\s+", " ", str(value or "").strip())


def clean_item(value: object) -> str:
    """Normalize a semantic fragment and remove only edge punctuation."""
    return normalize_text(value).strip(" \t\r\n.,;:!?-")


def dedupe_text(values: Sequence[object], limit: int | None = None) -> tuple[str, ...]:
    """Case-insensitive stable deduplication of textual values."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = clean_item(value)
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if limit is not None and len(result) >= limit:
            break
    return tuple(result)


def appears_in_query(term: str, query: str) -> bool:
    """Return whether a literal semantic term occurs in the user query."""
    term = clean_item(term)
    query = normalize_text(query)
    return bool(term) and term.casefold() in query.casefold()


def detect_list_question(query: str) -> bool:
    """Detect collection/list intent without treating every ``what`` question as a list."""
    normalized = normalize_text(query).casefold()
    if any(re.search(pattern, normalized) for pattern in LIST_PATTERNS):
        return True

    # Natural English often places the collection noun between ``what`` and
    # ``are`` (for example, ``what departments are there?``).
    if re.search(r"\bwhat\b", normalized) and re.search(r"\b(?:are|available|there|exist)\b", normalized):
        target = _deterministic_target(normalized)
        if target:
            plural_or_collection = bool(
                re.search(
                    r"\b(?:departments?|programs?|facilities?|research\s+areas?|research\s+opportunit(?:y|ies)|modes?|options?|categories?|types?)\b",
                    target,
                    flags=re.IGNORECASE,
                )
            )
            if plural_or_collection:
                return True

    # Hindi/Hinglish collection phrasing is intentionally narrow.
    if re.search(r"\b(?:kaun\s+kaun|kon\s+kon|kitne|kitni)\b", normalized):
        return True

    return False


def _deterministic_request_type(query: str) -> str | None:
    """Infer a generic request type only from strong lexical query cues."""
    normalized = normalize_text(query).casefold()
    for request_type, pattern in REQUEST_TYPE_BACKSTOP_PATTERNS:
        if re.search(pattern, normalized):
            return request_type
    return None


def _deterministic_target(query: str) -> str | None:
    """Extract the strongest generic information target present in the query."""
    normalized = normalize_text(query)
    matches: list[tuple[int, int, str]] = []
    for pattern in GENERIC_TARGET_PATTERNS:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        value = clean_item(match.group(0))
        if not value:
            continue
        matches.append((len(value.split()), match.start(), value))

    if not matches:
        return None

    matches.sort(key=lambda item: (-item[0], item[1], -len(item[2])))
    return matches[0][2]


def _deterministic_requested_attribute(request_type: str | None, query: str) -> str | None:
    """Return a generic requested property when the wording makes it explicit."""
    normalized = normalize_text(query).casefold()
    if request_type:
        for candidate_type, label in REQUESTED_ATTRIBUTE_LABELS:
            if request_type.casefold() == candidate_type:
                return label

    if re.search(r"\b(?:fee|fees|tuition|cost|charge|charges|rent|amount|price)\b", normalized):
        return "fee"
    if re.search(r"\b(?:deadline|deadlines)\b", normalized):
        return "deadline"
    if re.search(r"\b(?:eligibility|eligible|qualification|qualifications|criteria)\b", normalized):
        return "eligibility"
    if re.search(r"\b(?:process|procedure|how\s+(?:do|can)\s+i\s+apply|application)\b", normalized):
        return "procedure"
    if re.search(r"\b(?:where|located|location|address|directions?)\b", normalized):
        return "location"
    return None


def _apply_deterministic_backstop(
    interpretation: SemanticInterpretation,
    original_query: str,
) -> SemanticInterpretation:
    """Merge strong lexical signals without replacing richer LLM meaning."""
    list_intent = bool(interpretation.is_list_question or detect_list_question(original_query))

    lexical_request_type = _deterministic_request_type(original_query)
    current_request_type = clean_item(interpretation.request_type) or None
    generic_current_types = {"", "information", "other", "general_information"}

    if lexical_request_type and (current_request_type or "").casefold() in generic_current_types:
        current_request_type = lexical_request_type

    target = clean_item(interpretation.target) or None
    lexical_target = _deterministic_target(original_query)
    generic_targets = {
        "information",
        "general information",
        "process",
        "procedure",
        "criteria",
        "cost",
        "fee",
        "fees",
        "tuition",
        "tuition fee",
        "tuition fees",
    }
    requested_attribute_hint = _deterministic_requested_attribute(current_request_type, original_query)
    target_is_contextually_generic = (
        target.casefold() in generic_targets
        or (target.casefold() == "admission" and requested_attribute_hint == "fee")
    ) if target else True
    if (
        target_is_contextually_generic
        and lexical_target
        and lexical_target.casefold() not in {"institute", "college", "university"}
    ):
        target = lexical_target

    # Separate an explicit entity from a generic requested-property suffix.
    target = _canonicalize_composite_target(target or "", original_query) or None

    explicit_context_target = _extract_explicit_target_from_context(original_query)
    if (
        explicit_context_target
        and (
            not target
            or target.casefold() in generic_targets
            or target.casefold().startswith("admission ")
            or target.casefold().startswith("application ")
        )
    ):
        target = explicit_context_target

    # A direct process/application wording outranks a broad model label such
    # as ``application``.
    current_request_type = _force_explicit_request_type(
        original_query,
        lexical_request_type,
        current_request_type or "",
    ) or None

    facets = list(interpretation.facets)
    requested_attribute = _deterministic_requested_attribute(current_request_type, original_query)
    if requested_attribute and not any(
        facet.name.casefold() == "requested_attribute"
        for facet in facets
    ):
        facets.append(
            FacetPayload(
                name="requested_attribute",
                value=requested_attribute,
                required=True,
                importance=1.0,
            )
        )

    confidence = clamp_confidence(interpretation.confidence)
    strong_structure = bool(target and (current_request_type or lexical_request_type or list_intent))
    if strong_structure:
        deterministic_floor = 0.92 if lexical_request_type and target else 0.86
        confidence = max(confidence, deterministic_floor)

    return SemanticInterpretation(
        semantic_query=interpretation.semantic_query,
        target=target,
        request_type=current_request_type,
        facets=facets[:MAX_FACETS],
        qualifiers=interpretation.qualifiers,
        conditions=interpretation.conditions,
        relations=interpretation.relations,
        temporal_context=interpretation.temporal_context,
        comparison_targets=interpretation.comparison_targets,
        preserved_terms=interpretation.preserved_terms,
        is_list_question=list_intent,
        is_comparison_question=bool(interpretation.is_comparison_question),
        is_multi_part=bool(interpretation.is_multi_part),
        confidence=confidence,
        needs_clarification=bool(interpretation.needs_clarification),
        clarification_reason=interpretation.clarification_reason,
        language=interpretation.language,
        ambiguous=bool(interpretation.ambiguous),
        unknown_terms=interpretation.unknown_terms,
    )


def tokenise(text: str) -> tuple[str, ...]:
    """Tokenize text for conservative grounding checks."""
    return tuple(re.findall(r"[\w]+", normalize_text(text).casefold(), flags=re.UNICODE))


def token_overlap_supported(fragment: str, query: str, threshold: float = 0.80) -> bool:
    """Check that most meaningful tokens of a fragment are grounded in query."""
    fragment_tokens = {token for token in tokenise(fragment) if len(token) >= 2}
    if not fragment_tokens:
        return False
    query_tokens = set(tokenise(query))
    matched = len(fragment_tokens & query_tokens)
    return matched / len(fragment_tokens) >= threshold


def contiguous_tokens_supported(fragment: str, query: str) -> bool:
    """Require the fragment's token sequence to occur contiguously in the query."""
    fragment_tokens = tuple(tokenise(fragment))
    query_tokens = tuple(tokenise(query))
    if not fragment_tokens or not query_tokens or len(fragment_tokens) > len(query_tokens):
        return False
    width = len(fragment_tokens)
    return any(
        query_tokens[index:index + width] == fragment_tokens
        for index in range(len(query_tokens) - width + 1)
    )


def clamp_confidence(value: object) -> float:
    """Convert malformed confidence values into a safe [0, 1] range."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric != numeric:  # NaN
        return 0.0
    return min(1.0, max(0.0, numeric))


# ---------------------------------------------------------------------------
# Deterministic semantic serialization.
# ---------------------------------------------------------------------------

def build_semantic_query(frame: SemanticInterpretation) -> str:
    """Serialize structured meaning into concise retrieval text deterministically."""
    parts: list[str] = []

    target = clean_item(frame.target)
    if target:
        parts.append(target)

    request_type = clean_item(frame.request_type)
    if request_type and request_type.casefold() not in {"information", "other"}:
        parts.append(request_type)

    for facet in frame.facets[:MAX_FACETS]:
        parts.append(clean_item(facet.value))

    parts.extend(clean_item(item) for item in frame.qualifiers[:MAX_LIST_ITEMS])
    parts.extend(clean_item(item) for item in frame.conditions[:MAX_LIST_ITEMS])
    parts.extend(clean_item(item) for item in frame.relations[:MAX_LIST_ITEMS])
    parts.extend(clean_item(item) for item in frame.temporal_context[:MAX_LIST_ITEMS])
    parts.extend(clean_item(item) for item in frame.comparison_targets[:MAX_LIST_ITEMS])

    deduped = list(dedupe_text(parts))
    if not deduped:
        return ""

    return " ".join(deduped[:MAX_SEMANTIC_QUERY_TOKENS * 2])


# ---------------------------------------------------------------------------
# Deterministic query backstops.
# ---------------------------------------------------------------------------

_RESEARCH_TARGET_PATTERNS = (
    (r"\bresearch\s+areas?\b", "research areas"),
    (r"\bresearch\s+opportunities?\b", "research opportunities"),
)

_COLLECTION_TARGET_PATTERNS = (
    (r"\b(?:academic\s+)?programs?\b", "academic programs"),
    (r"\bdepartments?\b", "departments"),
    (r"\bfacilities\b", "facilities"),
    (r"\bhostel\s+facilities\b", "hostel facilities"),
)

# Generic target suffixes describe the requested property rather than the
# entity being asked about.  They are deliberately domain-level language
# patterns, not institution-specific vocabulary.
_TARGET_ATTRIBUTE_SUFFIXES = (
    "eligibility requirements",
    "eligibility criteria",
    "admission requirements",
    "admission routes",
    "admission process",
    "application procedure",
    "application process",
    "registration procedure",
    "registration process",
    "tuition fees",
    "tuition fee",
)

_GENERIC_TARGET_LABELS = {
    "admission",
    "procedure",
    "process",
    "information",
    "general information",
    "requirements",
    "criteria",
    "routes",
    "fees",
    "fee",
    "tuition",
    "tuition fee",
    "tuition fees",
}


def _query_has_any(query: str, patterns: Sequence[str]) -> bool:
    normalized = normalize_text(query).casefold()
    return any(re.search(pattern, normalized) for pattern in patterns)


def _first_literal_match(query: str, patterns: Sequence[tuple[str, str]]) -> str:
    normalized = normalize_text(query).casefold()
    for pattern, value in patterns:
        if re.search(pattern, normalized):
            return value
    return ""


def _detect_backstop_target(query: str) -> str:
    normalized = normalize_text(query).casefold()

    for pattern, value in _RESEARCH_TARGET_PATTERNS:
        if re.search(pattern, normalized):
            return value

    for pattern, value in _COLLECTION_TARGET_PATTERNS:
        if re.search(pattern, normalized):
            return value

    if re.search(r"\bhostel\s+(?:fees?|fee|cost|costs|charges?)\b", normalized):
        return "hostel fees"

    if re.search(r"\btuition\s+fees?\b|\btuition\b", normalized):
        return "tuition fee"

    if re.search(r"\badmission\s+fees?\b", normalized):
        return "admission fee"

    if re.search(r"\bapplication\s+deadlines?\b|\badmission\s+deadlines?\b", normalized):
        return "application deadlines"

    if re.search(r"\bwhere\s+is\s+(?:the\s+)?institute\s+(?:located|situated)\b", normalized):
        return "institute"

    if re.search(r"\b(?:how|where)\s+(?:do|can)\s+I\s+apply\b|\bprocedure\s+for\s+applying\b|\badmission\s+process\b", normalized):
        return "admission"

    if re.search(r"\beligibility\s+(?:requirements?|criteria)\b", normalized):
        # Program/entity extraction is intentionally delegated to the active
        # institution registry in the orchestration layer. The generic query
        # module must not contain a fixed degree vocabulary.
        return ""

    return ""


def _canonicalize_composite_target(
    target: str,
    query: str,
) -> str:
    """Reduce a model target to the entity when its suffix is an attribute.

    The query remains authoritative.  We only strip a known generic
    information-property suffix when the remaining prefix is explicitly
    grounded in the user's wording.
    """
    current = clean_item(target)
    if not current:
        return current

    normalized_target = normalize_text(current).casefold()
    normalized_query = normalize_text(query)

    for suffix in _TARGET_ATTRIBUTE_SUFFIXES:
        suffix_norm = normalize_text(suffix).casefold()
        marker = f" {suffix_norm}"
        if not normalized_target.endswith(marker):
            continue

        prefix = current[: len(current) - len(suffix) ].strip(" .,:;-—–")
        if not prefix:
            continue

        if prefix.casefold() in _GENERIC_TARGET_LABELS:
            continue

        if appears_in_query(prefix, normalized_query) or contiguous_tokens_supported(prefix, normalized_query):
            return prefix

    return current


def _extract_explicit_target_from_context(
    query: str,
) -> str:
    """Extract an explicit entity/program used as the target of a generic request.

    This is intentionally conservative.  It only considers a short phrase
    after a relationship marker such as ``for``/``of``/``in`` when that phrase
    reaches the end of the user's question.  It therefore does not invent an
    institution-specific entity.
    """
    normalized = normalize_text(query)

    match = re.search(
        r"\b(?:for|of|in)\s+(?P<value>[A-Za-z0-9][A-Za-z0-9.&'()/\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9.&'()/\-]*){0,5})\s*[?!.,:;]*$",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""

    value = clean_item(match.group("value"))
    if not value:
        return ""

    if value.casefold() in {
        "students",
        "student",
        "admission",
        "admissions",
        "application",
        "applications",
        "institute",
        "college",
        "university",
        "the institute",
        "the college",
        "the university",
    }:
        return ""

    return value


def _force_explicit_request_type(
    query: str,
    detected_request: str | None,
    current_request: str,
) -> str:
    """Prefer an unambiguous literal process/cost/eligibility signal.

    A structured model label such as ``application`` must not override a
    direct user phrase such as ``procedure for applying``.
    """
    normalized = normalize_text(query).casefold()

    if re.search(
        r"\b(?:eligibility|eligible|qualifications?|requirements?|criteria)\b",
        normalized,
    ):
        return "eligibility"

    if re.search(
        r"\b(?:procedure|process)\s+for\s+applying\b|\bhow\s+(?:do|can)\s+i\s+apply\b|\bhow\s+to\s+apply\b|\badmission\s+process\b",
        normalized,
    ):
        return "procedure"

    return detected_request or current_request


def _detect_backstop_request_type(query: str, target: str) -> str | None:
    normalized = normalize_text(query).casefold()

    if _query_has_any(normalized, (r"\beligib", r"\beligibility\b", r"\brequirements?\b", r"\bcriteria\b")):
        return "eligibility"

    if _query_has_any(normalized, (r"\btuition\b", r"\bfees?\b", r"\bcosts?\b", r"\bcharges?\b", r"\bprice\b", r"\bamount\b")):
        return "cost"

    if _query_has_any(normalized, (r"\bwhere\s+is\b.*\b(?:located|situated)\b", r"\blocation\b", r"\baddress\b")):
        return "directions"

    if _query_has_any(normalized, (r"\bdeadline\b", r"\bdeadlines\b")):
        return "admission"

    if _query_has_any(normalized, (r"\bhow\s+do\s+i\s+apply\b", r"\bhow\s+to\s+apply\b", r"\bprocedure\s+for\s+applying\b", r"\badmission\s+process\b")):
        return "procedure"

    if target in {"research areas", "research opportunities"}:
        return "research"

    if target in {"departments", "academic programs", "facilities", "hostel facilities"}:
        return "information"

    return None


def _detect_requested_attribute(query: str) -> str | None:
    normalized = normalize_text(query).casefold()

    if re.search(r"\beligibility\s+(?:requirements?|criteria)\b|\beligibility\b", normalized):
        return "eligibility requirements"
    if re.search(r"\bhostel\s+(?:fees?|fee|cost|costs|charges?)\b", normalized):
        return "hostel fees"
    if re.search(r"\btuition\s+fees?\b|\btuition\b", normalized):
        return "tuition fee"
    if re.search(r"\badmission\s+fees?\b", normalized):
        return "admission fee"
    if re.search(r"\bapplication\s+deadlines?\b|\badmission\s+deadlines?\b", normalized):
        return "application deadlines"
    if re.search(r"\b(?:how|where)\s+(?:do|can)\s+I\s+apply\b|\bprocedure\s+for\s+applying\b|\badmission\s+process\b", normalized):
        return "application procedure"
    if re.search(r"\bwhere\s+is\s+(?:the\s+)?institute\s+(?:located|situated)\b|\b(?:location|address)\b", normalized):
        return "location"
    if target := _detect_backstop_target(normalized):
        if target in {"departments", "academic programs", "research areas", "research opportunities", "facilities", "hostel facilities", "institute"}:
            return None
    return None


def _is_collection_question(query: str, target: str) -> bool:
    if target not in {
        "departments",
        "academic programs",
        "research areas",
        "research opportunities",
        "facilities",
        "hostel facilities",
    }:
        return False

    normalized = normalize_text(query).casefold()
    if re.search(r"\b(?:list|which|what all|which all|types?|categories?|modes?|options?|kaun kaun|kon kon|kitne|kitni)\b", normalized):
        return True

    return bool(
        re.search(r"\bwhat\s+(?:are|all)\b", normalized)
        and not re.search(r"\b(?:eligibility|requirements?|criteria|fees?|fee|cost|deadlines?)\b", normalized)
    )


def apply_deterministic_backstops(
    interpretation: SemanticInterpretation,
    original_query: str,
) -> SemanticInterpretation:
    """Recover obvious query semantics the structured model under-specified."""
    query = normalize_text(original_query)
    target = clean_item(interpretation.target)
    request_type = clean_item(interpretation.request_type)

    detected_target = _detect_backstop_target(query)

    # First recover an obvious target supplied by deterministic lexical
    # patterns when the model gave us no target or a generic label.
    if (
        not target
        or target.casefold() in {"information", "general information", "admission", "procedure", "student"}
    ) and detected_target:
        target = detected_target

    # If the model returned a composite ``entity + requested property`` target,
    # keep the entity as the target and represent the property separately as a
    # requested-attribute facet.
    target = _canonicalize_composite_target(target, query)

    # Generic request labels such as ``admission routes`` or ``admission
    # requirements`` may omit an explicit entity that is clearly stated in the
    # user's question (for example: ``... for M.Sc.``). Recover that entity
    # without maintaining a college-specific program dictionary.
    explicit_context_target = _extract_explicit_target_from_context(query)
    if (
        explicit_context_target
        and (
            not target
            or target.casefold() in _GENERIC_TARGET_LABELS
            or target.casefold().startswith("admission ")
            or target.casefold().startswith("application ")
        )
    ):
        target = explicit_context_target

    detected_request = _detect_backstop_request_type(query, target)
    request_type = _force_explicit_request_type(
        query,
        detected_request,
        request_type,
    )

    if request_type.casefold() in {"information", "other"} and target in {"research areas", "research opportunities"}:
        request_type = "research"

    # A strong literal cost/eligibility/procedure signal outranks a generic
    # model label, while a specific model intent remains authoritative.
    attribute = _detect_requested_attribute(query)
    facets = list(interpretation.facets)
    if attribute:
        replaced = False
        for index, facet in enumerate(facets):
            if facet.name.casefold() == "requested_attribute":
                facets[index] = FacetPayload(
                    name="requested_attribute",
                    value=attribute,
                    required=True,
                    importance=max(float(facet.importance), 0.95),
                )
                replaced = True
                break
        if not replaced:
            facets.append(
                FacetPayload(
                    name="requested_attribute",
                    value=attribute,
                    required=True,
                    importance=0.95,
                )
            )

    is_list = bool(
        interpretation.is_list_question
        or _is_collection_question(query, target)
    )

    confidence = float(interpretation.confidence)
    if detected_target or detected_request or attribute or is_list:
        confidence = max(confidence, 0.95)

    semantic_query = clean_item(interpretation.semantic_query)
    if request_type in {"cost", "eligibility", "procedure", "directions", "admission", "research"}:
        semantic_query_parts = [target, request_type]
        if attribute:
            semantic_query_parts.append(attribute)
        semantic_query = " ".join(item for item in dedupe_text(semantic_query_parts) if item)
    elif target and (
        not semantic_query
        or semantic_query.casefold() in {"information", "general information", "admission", "procedure"}
    ):
        semantic_query = target

    return interpretation.model_copy(
        update={
            "semantic_query": semantic_query,
            "target": target or interpretation.target,
            "request_type": request_type or interpretation.request_type,
            "facets": facets,
            "is_list_question": is_list,
            "confidence": confidence,
        }
    )


# ---------------------------------------------------------------------------
# Normalization + structural checks.
# ---------------------------------------------------------------------------

def normalize_interpretation(
    interpretation: SemanticInterpretation,
    original_query: str,
) -> SemanticInterpretation:
    """Clean model output without inventing replacements."""
    cleaned_target = clean_item(interpretation.target)
    if cleaned_target and not (
        appears_in_query(cleaned_target, original_query)
        or contiguous_tokens_supported(cleaned_target, original_query)
    ):
        cleaned_target = ""

    cleaned_facets: list[FacetPayload] = []
    for facet in interpretation.facets[:MAX_FACETS]:
        name = clean_item(facet.name)
        value = clean_item(facet.value)
        if not name or not value:
            continue
        normalized_name = name.casefold()
        if normalized_name in {"target", "subject", "entity"} and not (
            appears_in_query(value, original_query)
            or contiguous_tokens_supported(value, original_query)
        ):
            continue
        cleaned_facets.append(
            FacetPayload(
                name=name,
                value=value,
                required=bool(facet.required),
                importance=clamp_confidence(facet.importance),
            )
        )

    unknown_terms = dedupe_text(
        [item for item in interpretation.unknown_terms[:MAX_PRESERVED_TERMS] if appears_in_query(item, original_query)]
    )

    comparison_targets = [
        clean_item(item)
        for item in interpretation.comparison_targets[:MAX_LIST_ITEMS]
        if (
            appears_in_query(clean_item(item), original_query)
            or contiguous_tokens_supported(clean_item(item), original_query)
        )
    ]

    preserved_terms = dedupe_text(
        [
            item
            for item in (
                list(interpretation.preserved_terms)
                + ([interpretation.target] if interpretation.target else [])
                + [facet.value for facet in cleaned_facets]
            )
            if appears_in_query(item, original_query)
        ],
        MAX_PRESERVED_TERMS,
    )

    is_list = bool(interpretation.is_list_question or detect_list_question(original_query))
    is_ambiguous = bool(interpretation.ambiguous or unknown_terms or interpretation.needs_clarification)

    return SemanticInterpretation(
        semantic_query=clean_item(interpretation.semantic_query),
        target=cleaned_target or None,
        request_type=clean_item(interpretation.request_type) or None,
        facets=cleaned_facets,
        qualifiers=list(dedupe_text(interpretation.qualifiers, MAX_LIST_ITEMS)),
        conditions=list(dedupe_text(interpretation.conditions, MAX_LIST_ITEMS)),
        relations=list(dedupe_text(interpretation.relations, MAX_LIST_ITEMS)),
        temporal_context=list(dedupe_text(interpretation.temporal_context, MAX_LIST_ITEMS)),
        comparison_targets=comparison_targets,
        preserved_terms=list(preserved_terms),
        is_list_question=is_list,
        is_comparison_question=bool(interpretation.is_comparison_question),
        is_multi_part=bool(interpretation.is_multi_part),
        confidence=clamp_confidence(interpretation.confidence),
        needs_clarification=bool(interpretation.needs_clarification),
        clarification_reason=clean_item(interpretation.clarification_reason),
        language=clean_item(interpretation.language) or None,
        ambiguous=is_ambiguous,
        unknown_terms=list(unknown_terms),
    )


def structural_validate(
    interpretation: SemanticInterpretation,
    original_query: str,
    semantic_query: str,
) -> tuple[bool, str]:
    """Validate interpretation structure before allowing it to guide retrieval."""
    if not interpretation.target and not semantic_query:
        return False, "missing_semantic_target"

    if len(tokenise(semantic_query)) > MAX_SEMANTIC_QUERY_TOKENS:
        return False, "semantic_query_too_long"

    for term in interpretation.unknown_terms:
        if not appears_in_query(term, original_query):
            return False, "unknown_term_not_literal"

    # A purported target should never be fabricated by the interpretation LLM.
    if interpretation.target and not (
        appears_in_query(interpretation.target, original_query)
        or contiguous_tokens_supported(interpretation.target, original_query)
    ):
        return False, "target_not_grounded_in_user_query"

    return True, "structurally_valid"


def should_run_safety_check(interpretation: SemanticInterpretation) -> bool:
    """Preserve the legacy fast path: second model call only when warranted."""
    return bool(
        interpretation.ambiguous
        or interpretation.unknown_terms
        or interpretation.confidence < 0.90
        or (interpretation.request_type or "").casefold() == "definition"
    )


# ---------------------------------------------------------------------------
# Safety + conversion helpers.
# ---------------------------------------------------------------------------

_REQUEST_TYPE_ATTRIBUTE = {
    "eligibility": "eligibility",
    "cost": "fee",
    "directions": "location",
    "research": "research",
    "procedure": "procedure",
    "registration": "registration",
    "admission": "admission",
}


def derive_retrieval_requirement(
    interpretation: SemanticInterpretation,
) -> RetrievalRequirement:
    """Derive evidence strictness from generic query structure."""
    has_target = bool(interpretation.target)
    request_type = (interpretation.request_type or "").casefold()
    has_requested_attribute = any(
        facet.name.casefold() in {"requested_attribute", "attribute", "requested attribute"}
        or facet.value.casefold() in set(_REQUEST_TYPE_ATTRIBUTE.values())
        for facet in interpretation.facets
    ) or request_type in _REQUEST_TYPE_ATTRIBUTE
    has_scope_facet = any(
        facet.name.casefold() in {"location", "category", "person_type", "scope"}
        for facet in interpretation.facets
    )

    if interpretation.is_list_question:
        return RetrievalRequirement(
            mode="broad",
            require_target_alignment=False,
            require_attribute_alignment=False,
            require_scope_alignment=False,
            reject_explicit_conflict=True,
            allow_partial_evidence=True,
            max_unmatched_required_facets=1,
        )

    if interpretation.is_comparison_question:
        return RetrievalRequirement(
            mode="focused",
            require_target_alignment=has_target,
            require_attribute_alignment=has_requested_attribute,
            require_scope_alignment=has_scope_facet,
            reject_explicit_conflict=True,
            allow_partial_evidence=True,
            max_unmatched_required_facets=0,
        )

    if interpretation.is_multi_part:
        return RetrievalRequirement(
            mode="focused",
            require_target_alignment=has_target,
            require_attribute_alignment=has_requested_attribute,
            require_scope_alignment=has_scope_facet,
            reject_explicit_conflict=True,
            allow_partial_evidence=True,
            max_unmatched_required_facets=1,
        )

    if has_target and has_requested_attribute:
        return RetrievalRequirement(
            mode="exact",
            require_target_alignment=True,
            require_attribute_alignment=True,
            require_scope_alignment=has_scope_facet,
            reject_explicit_conflict=True,
            allow_partial_evidence=False,
            max_unmatched_required_facets=0,
        )

    if has_target or interpretation.request_type or interpretation.facets:
        return RetrievalRequirement(
            mode="focused",
            require_target_alignment=has_target,
            require_attribute_alignment=has_requested_attribute,
            require_scope_alignment=has_scope_facet,
            reject_explicit_conflict=True,
            allow_partial_evidence=True,
            max_unmatched_required_facets=1,
        )

    return RetrievalRequirement()


def _interpretation_payload(frame: SemanticInterpretation) -> dict[str, Any]:
    return {
        "target": frame.target,
        "request_type": frame.request_type,
        "facets": [facet.model_dump() for facet in frame.facets],
        "qualifiers": frame.qualifiers,
        "conditions": frame.conditions,
        "relations": frame.relations,
        "temporal_context": frame.temporal_context,
        "comparison_targets": frame.comparison_targets,
        "unknown_terms": frame.unknown_terms,
        "ambiguous": frame.ambiguous,
        "is_list_question": frame.is_list_question,
        "is_comparison_question": frame.is_comparison_question,
        "is_multi_part": frame.is_multi_part,
        "confidence": frame.confidence,
    }


def run_safety_check(
    query: str,
    interpretation: SemanticInterpretation,
    safety_model: StructuredModel,
) -> tuple[bool, str]:
    """Run the optional semantic safety check and fail closed on exceptions."""
    payload = _interpretation_payload(interpretation)
    prompt = SEMANTIC_SAFETY_PROMPT_TEMPLATE.format(
        query=query,
        payload=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )
    try:
        result = safety_model.invoke(prompt)
    except Exception:
        return False, "semantic_safety_verification_exception"

    if isinstance(result, dict):
        approved = bool(result.get("approved", False))
        reason = clean_item(result.get("reason", ""))
    else:
        approved = bool(getattr(result, "approved", False))
        reason = clean_item(getattr(result, "reason", ""))

    if not approved:
        return False, reason or "semantic_frame_rejected"
    return True, reason or "semantic_frame_approved"


def interpretation_to_query(
    original_query: str,
    interpretation: SemanticInterpretation,
    semantic_query: str,
    *,
    verification_status: str,
    verification_reason: str,
    semantic_grounded: bool,
) -> Query:
    """Convert normalized semantic meaning into the canonical query contract."""
    facets = tuple(
        QueryFacet(
            name=facet.name,
            value=facet.value,
            required=facet.required,
            importance=facet.importance,
        )
        for facet in interpretation.facets
    )

    intents = (
        Intent(
            name=interpretation.request_type or "information",
            confidence=interpretation.confidence,
        ),
    ) if interpretation.request_type else ()

    target = (
        Target(
            text=interpretation.target,
            confidence=interpretation.confidence,
            resolution_state="unresolved",
        )
        if interpretation.target
        else None
    )

    entities = ()
    mentions = ()
    if interpretation.target:
        mentions = (
            EntityMention(
                text=interpretation.target,
                normalized_text=interpretation.target,
                confidence=interpretation.confidence,
                resolution_state="unresolved",
            ),
        )

    qualifiers = tuple(
        Qualifier(name="qualifier", value=value)
        for value in interpretation.qualifiers
    )
    constraints = tuple(
        Constraint(name="condition", value=value)
        for value in interpretation.conditions
    )

    ambiguity = Ambiguity(
        is_ambiguous=interpretation.ambiguous,
        terms=tuple(interpretation.unknown_terms),
        reason=interpretation.clarification_reason,
    )

    retrieval_requirement = derive_retrieval_requirement(interpretation)

    return Query(
        original_query=original_query,
        normalized_query=original_query,
        search_query=semantic_query,
        intents=intents,
        target=target,
        entities=entities,
        entity_mentions=mentions,
        concepts=(),
        relations=(),
        qualifiers=qualifiers,
        constraints=constraints,
        temporal_constraints=(),
        numeric_requirements=(),
        scope=None,
        request_type=interpretation.request_type,
        list_intent=ListIntent(is_list=interpretation.is_list_question),
        ambiguity=ambiguity,
        unknown_terms=tuple(interpretation.unknown_terms),
        resolution_state="unresolved",
        verification_status=verification_status,
        confidence=interpretation.confidence,
        language=interpretation.language,
        is_comparison=interpretation.is_comparison_question,
        is_multi_part=interpretation.is_multi_part,
        comparison_targets=tuple(
            Target(
                text=value,
                confidence=interpretation.confidence,
                resolution_state="unresolved",
            )
            for value in interpretation.comparison_targets
        ),
        preserved_terms=tuple(interpretation.preserved_terms),
        needs_clarification=interpretation.needs_clarification,
        clarification_reason=(interpretation.clarification_reason or verification_reason),
        retrieval_requirement=retrieval_requirement,
        semantic_grounded=semantic_grounded,
    )


# ---------------------------------------------------------------------------
# Fail-safe fallback.
# ---------------------------------------------------------------------------

def fallback_query(original_query: str, reason: str) -> Query:
    """Construct a conservative query when semantic interpretation fails."""
    original_query = normalize_text(original_query)
    if not original_query:
        raise ValueError("original_query cannot be empty.")

    return Query(
        original_query=original_query,
        normalized_query=original_query,
        search_query="",
        intents=(),
        target=None,
        ambiguity=Ambiguity(is_ambiguous=False, reason=""),
        unknown_terms=(),
        resolution_state="unresolved",
        verification_status="failed",
        confidence=0.0,
        preserved_terms=(),
        needs_clarification=False,
        clarification_reason=reason,
        semantic_grounded=False,
    )


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def _load_default_structured_models() -> tuple[StructuredModel, StructuredModel]:
    """Lazily construct structured-output models from the runtime adapter."""
    from ai_platform.runtime.llm import query_understanding_llm

    structured_semantic = query_understanding_llm.with_structured_output(
        SemanticInterpretation,
        method="json_schema",
        include_raw=False,
    )
    structured_safety = query_understanding_llm.with_structured_output(
        SemanticSafetyCheck,
        method="json_schema",
        include_raw=False,
    )
    return structured_semantic, structured_safety


def _understand_query_uncached(
    query: str,
    *,
    semantic_model: StructuredModel | None,
    safety_model: StructuredModel | None,
) -> Query:
    original_query = normalize_text(query)
    if not original_query:
        raise ValueError("query cannot be empty.")

    if semantic_model is None or safety_model is None:
        default_semantic, default_safety = _load_default_structured_models()
        semantic_model = semantic_model or default_semantic
        safety_model = safety_model or default_safety

    try:
        response = semantic_model.invoke(
            [
                ("system", SEMANTIC_SYSTEM_PROMPT),
                (
                    "human",
                    f"<USER_QUERY>\n{original_query}\n</USER_QUERY>",
                ),
            ]
        )
        interpretation = (
            response
            if isinstance(response, SemanticInterpretation)
            else SemanticInterpretation.model_validate(response)
        )
        interpretation = apply_deterministic_backstops(
            interpretation,
            original_query,
        )
    except Exception:
        return fallback_query(
            original_query,
            "structured_semantic_model_exception",
        )

    try:
        interpretation = normalize_interpretation(
            interpretation,
            original_query,
        )
        interpretation = _apply_deterministic_backstop(
            interpretation,
            original_query,
        )
        semantic_query = build_semantic_query(interpretation)
    except Exception:
        return fallback_query(
            original_query,
            "semantic_frame_normalization_exception",
        )

    grounded, structural_reason = structural_validate(
        interpretation,
        original_query,
        semantic_query,
    )

    if not grounded:
        return interpretation_to_query(
            original_query,
            interpretation,
            semantic_query,
            verification_status="rejected",
            verification_reason=structural_reason,
            semantic_grounded=False,
        )

    verification_status = "skipped"
    verification_reason = "clear_semantic_frame"

    if should_run_safety_check(interpretation):
        approved, safety_reason = run_safety_check(
            original_query,
            interpretation,
            safety_model,
        )
        if not approved:
            return interpretation_to_query(
                original_query,
                interpretation,
                semantic_query,
                verification_status="rejected",
                verification_reason=safety_reason,
                semantic_grounded=True,
            )
        verification_status = "approved"
        verification_reason = safety_reason

    safe = bool(
        interpretation.confidence >= 0.80
        and not interpretation.ambiguous
        and grounded
    )

    return interpretation_to_query(
        original_query,
        interpretation,
        semantic_query,
        verification_status=verification_status,
        verification_reason=verification_reason,
        semantic_grounded=True,
    )


@lru_cache(maxsize=256)
def _understand_query_cached(query: str) -> Query:
    return _understand_query_uncached(
        query,
        semantic_model=None,
        safety_model=None,
    )


def understand_query(
    query: str,
    *,
    semantic_model: StructuredModel | None = None,
    safety_model: StructuredModel | None = None,
) -> Query:
    """Public semantic-understanding API returning the canonical ``Query`` contract."""
    normalized = normalize_text(query)
    if not normalized:
        raise ValueError("query cannot be empty.")

    if semantic_model is None and safety_model is None:
        return _understand_query_cached(normalized)

    return _understand_query_uncached(
        normalized,
        semantic_model=semantic_model,
        safety_model=safety_model,
    )


def understand_query_frame(
    query: str,
    *,
    semantic_model: StructuredModel | None = None,
    safety_model: StructuredModel | None = None,
) -> SemanticQueryFrame:
    """Temporary migration helper exposing a legacy-compatible frame."""
    canonical = understand_query(
        query,
        semantic_model=semantic_model,
        safety_model=safety_model,
    )

    target = canonical.target.text if canonical.target else None
    qualifiers = tuple(item.value for item in canonical.qualifiers)
    conditions = tuple(str(item.value) for item in canonical.constraints)
    facets = tuple(
        QueryFacet(
            name="qualifier",
            value=item.value,
            required=item.required,
            importance=item.importance,
        )
        for item in canonical.qualifiers
    )
    if canonical.request_type:
        facets = facets + (
            QueryFacet(
                name="request_type",
                value=canonical.request_type,
                required=False,
                importance=0.8,
            ),
        )

    return SemanticQueryFrame(
        original_query=canonical.original_query,
        normalized_query=canonical.normalized_query,
        semantic_query=canonical.search_query,
        target=target,
        request_type=canonical.request_type,
        facets=facets,
        qualifiers=qualifiers,
        conditions=conditions,
        relations=(),
        temporal_context=(),
        comparison_targets=tuple(item.text for item in canonical.comparison_targets),
        preserved_terms=canonical.preserved_terms,
        is_list_question=canonical.list_intent.is_list,
        is_comparison_question=canonical.is_comparison,
        is_multi_part=canonical.is_multi_part,
        requirement=canonical.retrieval_requirement,
        confidence=canonical.confidence,
        needs_clarification=canonical.needs_clarification,
        clarification_reason=canonical.clarification_reason,
        language=canonical.language,
        interpretation_status=(
            "trusted" if canonical.semantic_grounded and canonical.verification_status in {"approved", "not_run"}
            else "review"
        ),
        intents=canonical.intents,
        entities=canonical.entities,
        entity_mentions=canonical.entity_mentions,
        concepts=canonical.concepts,
        structured_qualifiers=canonical.qualifiers,
        constraints=canonical.constraints,
        temporal_constraints=canonical.temporal_constraints,
        numeric_requirements=canonical.numeric_requirements,
        scope=canonical.scope,
        ambiguity=canonical.ambiguity,
        unknown_terms=canonical.unknown_terms,
        resolution_state=canonical.resolution_state,
        verification_status=canonical.verification_status,
        semantic_grounded=canonical.semantic_grounded,
    )


__all__ = [
    "FacetPayload",
    "SemanticInterpretation",
    "SemanticSafetyCheck",
    "SEMANTIC_SYSTEM_PROMPT",
    "SEMANTIC_SAFETY_PROMPT_TEMPLATE",
    "build_semantic_query",
    "structural_validate",
    "should_run_safety_check",
    "fallback_query",
    "understand_query",
    "understand_query_frame",
]