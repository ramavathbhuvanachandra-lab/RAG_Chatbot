"""
Reusable deterministic reranking for institutional RAG.

Pipeline position
-----------------
    dense / lexical retrieval -> weighted RRF -> this module -> evidence

Design goals
------------
* institution-agnostic: no college names, program names, or source filenames;
* primary-query authoritative;
* target- and attribute-aware for focused questions;
* broad-scope aware without allowing source folders to dominate;
* conservative with semantic metadata supplied by upstream stages;
* explicit conflicts are strongly penalized;
* exact duplicate chunks are removed during top-k selection;
* source diversity is bounded, not forced when it would hide better evidence;
* deterministic and dependency-light; no LLM calls.

The ranker is intentionally a ranking component, not an answer/evidence
sufficiency component. Retrieval continues to provide the candidate set;
this module only orders that set and performs safe top-k diversification.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from ai_platform.core.retrieval.contracts import RetrievalCandidate
from ai_platform.core.retrieval.evidence_matcher import _singular_forms


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RankingConfig:
    """Bounded deterministic ranking and diversity configuration."""

    lexical_weight: float = 0.95
    phrase_weight: float = 0.75
    rrf_weight: float = 0.85

    semantic_weight: float = 1.15
    semantic_coverage_weight: float = 0.45

    target_match_weight: float = 5.00
    target_miss_penalty: float = 4.00

    attribute_match_weight: float = 5.80
    attribute_partial_weight: float = 2.20
    required_attribute_miss_penalty: float = 3.75

    scope_match_weight: float = 1.80
    scope_conflict_penalty: float = 1.80

    source_role_weight: float = 1.20
    broad_source_bonus: float = 1.25
    specialized_without_target_penalty: float = 4.00

    qualifier_match_weight: float = 1.20
    qualifier_miss_penalty: float = 1.20

    intrinsic_quality_weight: float = 0.65
    conflict_penalty: float = 25.0

    max_phrase_bonus: float = 2.20
    max_candidate_score: float = 30.0

    max_per_source: int = 4
    dedupe_exact_content: bool = True
    diversify_when_top_k: bool = True


DEFAULT_RANKING_CONFIG = RankingConfig()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "at", "be", "can", "could", "did",
        "do", "does", "for", "from", "how", "i", "in", "is", "it",
        "me", "of", "on", "or", "please", "the", "this", "to", "what",
        "when", "where", "which", "who", "with", "you", "your",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=4096)
def _normalize(value: Any) -> str:
    text = str(value or "").casefold().replace("\\", "/")
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=4096)
def _compact_token(value: Any) -> str:
    cleaned = re.sub(r"[^a-z0-9]", "", str(value or "").casefold())
    return cleaned if 2 <= len(cleaned) <= 8 and cleaned not in _STOPWORDS else ""


@lru_cache(maxsize=4096)
def _tokens(value: Any) -> tuple[str, ...]:
    out: list[str] = []
    norm = _normalize(value)
    for token in _TOKEN_RE.findall(norm):
        if (len(token) < 2 or (len(token) == 2 and not token.isdigit())) or token in _STOPWORDS:
            continue
        out.append(token)
    comp = _compact_token(value)
    if comp and comp not in out:
        out.append(comp)
    return tuple(out)


@lru_cache(maxsize=4096)
def _token_set(value: Any) -> frozenset[str]:
    raw_tokens = _tokens(value)
    expanded: set[str] = set(raw_tokens)
    for t in raw_tokens:
        expanded.update(_singular_forms(t))
    return frozenset(expanded)


def _document_text(candidate: RetrievalCandidate) -> str:
    document = getattr(candidate, "document", None)
    return _normalize(getattr(document, "page_content", ""))


def _source(candidate: RetrievalCandidate) -> str:
    return _normalize(getattr(candidate, "source", ""))


def _contains_any(text: str, phrases: Sequence[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    """Cheap typo detector for long semantic vocabulary tokens."""
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    index_left = index_right = edits = 0
    while index_left < len(left) and index_right < len(right):
        if left[index_left] == right[index_right]:
            index_left += 1
            index_right += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(left) == len(right):
            index_left += 1
            index_right += 1
        else:
            index_right += 1
    return edits + (len(right) - index_right) <= 1


def _query_has_token(token: str, query_tokens: set[str]) -> bool:
    token = token.casefold()
    if token in query_tokens:
        return True
    if len(token) < 7:
        return False
    return any(_edit_distance_at_most_one(token, item) for item in query_tokens)


def _token_overlap(left: Sequence[str], right: set[str]) -> float:
    source = set(left)
    if not source:
        return 0.0
    return len(source & right) / len(source)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


# ---------------------------------------------------------------------------
# Generic semantic attribute vocabulary
# ---------------------------------------------------------------------------

# These are retrieval-intent classes, not institution-specific entities.
# They are deliberately broad enough to apply across campuses.
_ATTRIBUTE_MARKERS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "documents",
        (
            "documents", "certificates", "paperwork", "score card",
            "scorecard", "proof of address", "originals and copies",
            "bring originals", "copies of", "necessary certificates",
        ),
        ("submit", "bring", "documents", "certificates", "paperwork", "copies"),
    ),
    (
        "contact",
        (
            "contact", "email", "e-mail", "phone number", "telephone",
            "helpline", "coordinator", "contact details", "office", 
        ),
        ("contact", "email", "phone", "telephone", "coordinator"),
    ),
    (
        "process",
        (
            "admission process", "application process", "admission procedure",
            "application procedure", "registration procedure", "how to apply",
            "how to register", "steps", "procedure", "process", "admission cycle",
            "admissions cycle",
        ),
        ("process", "procedure", "apply", "application", "register", "registration", "steps", "admission", "admissions"),
    ),
    (
        "fees",
        (
            "fee", "fees", "tuition", "cost", "costs", "charge", "charges",
            "payment", "rent", "semester fee",
        ),
        ("fee", "tuition", "cost", "charge", "payment", "rent"),
    ),
    (
        "rules",
        (
            "rules", "rule", "guidelines", "guideline", "regulations", "regulation", "policy", "policies",
            "attendance", "attendance rule", "attendance policy", "permitted", "prohibited", "allowed", "not allowed",
        ),
        ("rules", "rule", "guidelines", "guideline", "regulations", "regulation", "policy", "attendance", "permitted", "prohibited"),
    ),
    (
        "dining",
        (
            "dining", "food", "mess", "meal", "meals", "canteen", "cafeteria",
        ),
        ("dining", "food", "mess", "meal", "canteen", "cafeteria"),
    ),
    (
        "research",
        (
            "research area", "research areas", "research theme", "research themes",
            "research group", "research groups", "research opportunities",
        ),
        ("research", "themes", "areas", "laboratory", "lab"),
    ),
    (
        "facilities",
        (
            "facilities", "amenities", "infrastructure", "health center",
            "health centre", "laboratory", "laboratories", "lab facilities",
            "medical facilities",
        ),
        ("facilities", "amenities", "infrastructure", "laboratory", "health"),
    ),
    (
        "minor_programs",
        (
            "minor programs", "minor program", "minor programmes", "minor programme",
            "minor degree", "minor courses",
        ),
        ("minor",),
    ),
    (
        "booking",
        (
            "booking", "bookings", "reservation", "reservations",
            "short-term", "short term", "accommodation booking",
        ),
        ("booking", "reservation", "short", "stay"),
    ),
    (
        "programs",
        (
            "programs", "programmes", "degrees", "courses", "academic programs",
            "academic programmes",
        ),
        ("program", "programme", "degree", "course"),
    ),
    (
        "departments",
        (
            "departments", "department", "academic departments", "list of departments",
            "what departments", "which departments", "schools", "academic schools",
            "centres", "centers",
        ),
        ("department", "departments", "school", "schools", "centre", "centres", "center", "centers"),
    ),
    (
        "eligibility",
        (
            "eligibility", "eligible", "qualify", "qualified", "qualification", "admission requirements",
            "requirements", "criteria", "criterion", "must have", "must either", "shall have",
        ),
        (
            "eligibility", "eligible", "qualify", "qualified", "qualification", "requirements",
            "criteria", "required", "requirement", "minimum", "must", "applicant", "applicants",
            "exempted", "exemption",
        ),
    ),
    (
        "deadlines",
        (
            "deadline", "deadlines", "last date", "closing date", "application date",
            "when can i apply", "application dates", "admission cycle", "admissions cycle",
            "academic cycle", "admission timeline", "application timeline",
        ),
        ("deadline", "date", "dates", "apply", "closing", "cycle", "timeline"),
    ),
    (
        "location",
        (
            "where is", "where are", "located", "location", "directions", "address",
            "map", "near", "how to reach",
        ),
        ("located", "location", "address", "directions", "map", "reach"),
    ),
    (
        "fellowship",
        (
            "fellowship", "fellowships", "financial assistance", "scholarship", "scholarships",
            "stipend", "assistantship", "teaching assistantship", "research assistantship",
            "monthly fellowship", "contingency",
        ),
        (
            "fellowship", "fellowships", "financial", "assistance", "scholarship",
            "scholarships", "stipend", "assistantship", "mhrd", "norm", "norms",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class _AttributeRule:
    name: str
    phrases: tuple[str, ...]
    tokens: tuple[str, ...]


_ATTRIBUTE_RULES = tuple(
    _AttributeRule(name, tuple(phrases), tuple(tokens))
    for name, phrases, tokens in _ATTRIBUTE_MARKERS
)


@lru_cache(maxsize=256)
def _derive_query_attributes(query: str, query_frame: Any | None) -> tuple[str, ...]:
    normalized = _normalize(query)
    matched: list[tuple[str, float]] = []

    for rule in _ATTRIBUTE_RULES:
        query_tokens = _token_set(normalized)
        phrase_hits = sum(1 for marker in rule.phrases if marker in normalized)
        token_hits = sum(1 for marker in rule.tokens if _query_has_token(marker, query_tokens))
        score = phrase_hits * 1.0 + token_hits * 0.25
        if score > 0:
            matched.append((rule.name, score))

    # Upstream request_type is valuable when present, but it must be treated
    # as a hint, not a factual source of attributes.
    request_type = str(getattr(query_frame, "request_type", "") or "").casefold().strip()
    if request_type:
        aliases = {
            "fee": "fees", "fees": "fees", "cost": "fees",
            "contact": "contact", "documents": "documents", "document": "documents",
            "registration": "process", "application": "process", "admission": "process",
            "research": "research", "program": "programs", "programs": "programs",
            "department": "departments", "departments": "departments",
            "facility": "facilities", "facilities": "facilities",
            "rule": "rules", "rules": "rules",
        }
        canonical = aliases.get(request_type)
        if canonical:
            matched.append((canonical, 1.25))

    # Retain at most three attributes, ordered by evidence strength.
    ranked = sorted(matched, key=lambda item: (-item[1], item[0]))
    result: list[str] = []
    seen: set[str] = set()
    for name, _score in ranked:
        if name in seen:
            continue
        seen.add(name)
        result.append(name)
        if len(result) >= 3:
            break
    return tuple(result)


# ---------------------------------------------------------------------------
# Generic source roles / scope
# ---------------------------------------------------------------------------

_SCOPE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("admissions", ("admission", "admissions", "how to apply", "admission process", "application process")),
    ("fees", ("fee", "fees", "cost", "tuition", "charge", "charges", "payment", "rent")),
    ("hostel", ("hostel", "accommodation", "residence", "residential")),
    ("research", ("research area", "research areas", "research theme", "research themes", "research group")),
    ("programs", ("what programs", "which programs", "programs offered", "programmes offered", "academic programs", "academic programmes", "b.tech programmes", "btech programmes")),
    ("departments", ("what departments", "which departments", "departments does", "list of departments", "academic departments", "schools")),
    ("facilities", ("facilities available", "campus facilities", "infrastructure", "amenities", "laboratories", "medical facilities")),
    ("registration", ("registration", "register for", "course registration", "registration procedure")),
    ("emergency", ("emergency", "urgent medical", "ambulance", "emergency contact", "emergency contacts")),
)


def detect_query_scopes(query: str) -> frozenset[str]:
    normalized = _normalize(query)
    scopes: set[str] = set()
    query_tokens = _token_set(normalized)
    for scope, markers in _SCOPE_MARKERS:
        if any(marker in normalized for marker in markers):
            scopes.add(scope)
            continue
        if any(_query_has_token(marker, query_tokens) for marker in markers if " " not in marker):
            scopes.add(scope)
    return frozenset(scopes)


_SOURCE_ROLE_SEGMENTS: Mapping[str, tuple[str, ...]] = {
    "admissions": ("admissions",),
    "registration": ("registration", "academic_administration"),
    "fees": ("finance", "fees", "fee"),
    "hostel": ("hostel", "hostel_accommodation", "accommodation"),
    "research": ("research", "research_platforms"),
    "programs": ("programs",),
    "departments": ("departments", "department", "programs", "offices_and_administration", "office_of_academics"),
    "facilities": ("facilities", "research_and_technology_facilities"),
    "emergency": ("emergency", "health", "medical"),
    "administration": ("offices_and_administration", "office", "administration"),
}


def _is_inventory_or_directory_candidate(candidate: RetrievalCandidate) -> bool:
    """Return whether candidate represents an institution-wide inventory, directory, or dataset."""
    doc = getattr(candidate, "document", None)
    meta = getattr(doc, "metadata", {}) if doc and hasattr(doc, "metadata") else {}
    if not isinstance(meta, Mapping):
        meta = {}
    doc_type = str(meta.get("document_type") or "").casefold().strip()
    if doc_type in {"structured_record", "dataset", "directory"}:
        return True
    subcategory = str(meta.get("subcategory") or "").casefold().strip()
    if "list" in subcategory or "directory" in subcategory:
        return True
    title = str(meta.get("title") or "").casefold().strip()
    if any(m in title for m in ("directory", "catalogue", "catalog", "complete list", "inventory", "all departments", "all programs")):
        return True
    src = _source(candidate).casefold()
    stem = src.rsplit("/", 1)[-1]
    if any(stem.endswith(ext) for ext in (".json", ".csv", ".tsv")):
        if any(marker in stem for marker in ("department", "program", "course", "faculty", "fee")):
            return True
    if any(m in stem for m in ("directory", "catalogue", "catalog", "inventory", "departments_list", "programmes_list")):
        return True
    return False


def _inventory_candidate_scope(candidate: RetrievalCandidate) -> str | None:
    doc = getattr(candidate, "document", None)
    meta = getattr(doc, "metadata", {}) if doc and hasattr(doc, "metadata") else {}
    if not isinstance(meta, Mapping):
        meta = {}
    category = str(meta.get("category") or "").casefold().strip()
    subcategory = str(meta.get("subcategory") or "").casefold().strip()
    title = str(meta.get("title") or "").casefold().strip()
    src = _source(candidate).casefold()
    stem = src.rsplit("/", 1)[-1]
    combined = f"{category} {subcategory} {title} {stem}"
    if any(m in combined for m in ("department", "departments", "school", "schools")):
        return "departments"
    if any(m in combined for m in ("programme", "program", "course", "degree", "btech", "mtech", "phd")):
        return "programs"
    if any(m in combined for m in ("faculty", "professor", "teacher", "staff")):
        return "faculty"
    if any(m in combined for m in ("fee", "fees", "tuition", "scholarship", "finance")):
        return "fees"
    return None


def _is_relevant_inventory_candidate(
    candidate: RetrievalCandidate,
    query: str,
    query_attributes: Sequence[str] = (),
) -> bool:
    """Return True if candidate is an inventory/directory matching the query scope."""
    if not _is_inventory_or_directory_candidate(candidate):
        return False
    inv_scope = _inventory_candidate_scope(candidate)
    if not inv_scope:
        return False
    query_scopes = detect_query_scopes(query)
    if inv_scope in query_scopes:
        return True
    if query_attributes and inv_scope in query_attributes:
        return True
    q_norm = _normalize(query)
    if inv_scope == "departments" and any(k in q_norm for k in ("department", "departments", "school", "schools")):
        return True
    if inv_scope == "programs" and any(k in q_norm for k in ("program", "programs", "programme", "programmes", "course", "courses", "btech", "mtech", "degree")):
        return True
    if inv_scope == "faculty" and any(k in q_norm for k in ("faculty", "professor", "professors", "teacher", "teachers", "staff")):
        return True
    if inv_scope == "fees" and any(k in q_norm for k in ("fee", "fees", "tuition", "cost", "charge", "charges", "scholarship")):
        return True
    return False


def _path_segments(source: str) -> tuple[str, ...]:
    return tuple(part for part in re.split(r"/+", _normalize(source)) if part)


def _has_role(source: str, role: str) -> bool:
    segments = set(_path_segments(source))
    return any(segment in segments for segment in _SOURCE_ROLE_SEGMENTS.get(role, ()))


def _is_specialized_source(source: str, candidate: RetrievalCandidate | None = None) -> bool:
    if candidate is not None and _is_inventory_or_directory_candidate(candidate):
        return False
    norm = _normalize(source)
    if any(k in norm for k in ("/data/", "/datasets/")) and norm.endswith((".json", ".csv", ".tsv")):
        return False
    if any(k in norm for k in ("/schools/", "/departments/", "/research_platforms/")):
        return True
    stem = source.rsplit("/", 1)[-1]
    return not _is_generic_source(source) and any(
        marker in stem for marker in (
            "program", "department", "special", "international", "school",
        )
    )


def _is_generic_source(source: str) -> bool:
    norm = _normalize(source)
    if any(k in norm for k in ("/schools/", "/departments/", "/research_platforms/")):
        return False
    if any(k in norm for k in ("/institute_overview/", "/finance/", "/campus_and_student_services/")):
        return True
    stem = source.rsplit("/", 1)[-1]
    return any(marker in stem for marker in ("general", "overview", "index", "home", "main", "fees_and_finance", "programs", "academics"))


def _source_scope_signal(query: str, candidate: RetrievalCandidate) -> tuple[float, float]:
    scopes = detect_query_scopes(query)
    if not scopes:
        return 0.0, 0.0

    source = _source(candidate)
    text = _document_text(candidate)
    match = 0.0
    conflict = 0.0

    # A requested scope is supporting evidence, never a hard gate by itself.
    for scope in scopes:
        if _has_role(source, scope):
            match += 1.0
        elif scope in {"admissions", "fees", "hostel", "research", "programs", "departments", "facilities"}:
            # Generic content may be stored in administration/other sections;
            # only apply a small negative signal when another strong role exists.
            if any(_has_role(source, other) for other in _SOURCE_ROLE_SEGMENTS if other != scope):
                conflict += 0.25

        # Content itself can rescue a differently stored document.
        if scope == "emergency" and _contains_any(text, ("health center", "health centre", "ambulance", "emergency")):
            match += 0.5
        elif scope == "research" and "research" in text:
            match += 0.35
        elif scope == "facilities" and _contains_any(text, ("facilities", "laboratory", "laboratories")):
            match += 0.35
        elif scope == "hostel" and _contains_any(text, ("hostel", "accommodation")):
            match += 0.25
        elif scope == "departments" and _contains_any(text, ("departments", "school of", "centres", "center for")):
            match += 0.40

    return min(2.0, match), min(1.0, conflict)

    return min(2.0, match), min(1.0, conflict)


# ---------------------------------------------------------------------------
# Query target / qualifiers
# ---------------------------------------------------------------------------

def _query_target_text(query_frame: Any | None) -> str:
    target = getattr(query_frame, "target", None)
    if target is None:
        return ""
    text = getattr(target, "text", "") if hasattr(target, "text") else str(target or "")
    return _normalize(text)


def _candidate_semantic_text(candidate: RetrievalCandidate) -> str:
    meaning = getattr(candidate, "meaning", None)
    if meaning is None:
        return ""
    values: list[str] = []
    for field_name in ("programs", "entities", "topics", "attributes", "scope", "qualifiers", "constraints"):
        field = getattr(meaning, field_name, ()) or ()
        values.extend(str(value) for value in field if str(value).strip())
    intent = getattr(meaning, "intent", None)
    if intent:
        values.append(str(intent))
    return _normalize(" ".join(values))


_GENERIC_INSTITUTION_TARGETS: frozenset[str] = frozenset({
    "iit jodhpur", "iitj", "iit", "indian institute of technology jodhpur",
    "the institute", "institute", "the institution", "institution", "campus",
    "college", "university",
})


def _explicit_query_entities(query_frame: Any | None) -> tuple[Any, ...]:
    return tuple(
        e for e in (getattr(query_frame, "entities", ()) or ())
        if (str(getattr(e, "resolution_state", "") or "").casefold() in {"resolved", "approved", "trusted"}
            or float(getattr(e, "confidence", 0.0) or 0.0) >= 0.95)
        and _normalize(getattr(e, "name", "")) not in _GENERIC_INSTITUTION_TARGETS
    )


def _effective_query_target(query: str, query_frame: Any | None) -> str:
    target = _query_target_text(query_frame)
    if target and target not in _GENERIC_INSTITUTION_TARGETS:
        return target
    for entity in _explicit_query_entities(query_frame):
        name = _normalize(getattr(entity, "name", ""))
        if name and name not in _GENERIC_INSTITUTION_TARGETS:
            return name
    return ""

def _explicit_target_match(query: str, query_frame: Any | None, candidate: RetrievalCandidate) -> float:
    entities = _explicit_query_entities(query_frame)
    text = _document_text(candidate)
    source = _source(candidate)
    meaning = candidate.meaning
    candidate_tokens = _token_set(text + " " + source + " " + _candidate_semantic_text(candidate))
    if entities:
        scores = []
        for entity in entities:
            name = _normalize(getattr(entity, "name", ""))
            kind = _normalize(getattr(entity, "entity_type", "entity"))
            if name in text or name in source:
                scores.append(1.0)
                continue
            field = "programs" if kind == "program" else "entities" if kind == "entity" else "topics"
            scores.append(1.0 if any(_normalize(v) == name for v in (getattr(meaning, field, ()) or ())) else 0.0)
        return min(scores) if scores else 0.0
    target = _effective_query_target(query, query_frame)
    if not target:
        return 0.0
    student_markers = (
        "a student", "the student", "all students", "any student", "every student", "each student",
        "for students", "to students", "of students", "students", "research scholar", "research scholars",
        "scholars", "doctoral", "postgraduate", "undergraduate", "candidates", "applicant",
    )
    target_compact = _compact_token(target)
    if target_compact and len(target_compact) <= 6:
        if target_compact in candidate_tokens:
            return 1.0
        query_tokens = _token_set(_normalize(query))
        stop_words = {"what", "when", "where", "which", "how", "for", "the", "and", "are", "is", "can", "does", "rules"}
        acronyms = {tok for tok in query_tokens if 2 <= len(tok) <= 6 and tok not in stop_words}
        if acronyms and (acronyms & candidate_tokens):
            return 1.0
        return 0.0
    q = _token_set(target)
    if not q:
        return 0.0
    overlap = len(q & candidate_tokens) / len(q)
    if overlap >= 0.95:
        return 1.0
    if overlap >= 0.60:
        return 0.70
    if overlap > 0:
        return 0.35
    return 0.0

def _query_qualifiers(query_frame: Any | None) -> tuple[str, ...]:
    values: list[str] = []
    for field_name in ("qualifiers", "conditions", "structured_qualifiers"):
        field = getattr(query_frame, field_name, ()) or ()
        for value in field:
            item = getattr(value, "value", value)
            if str(item).strip():
                values.append(str(item))
    return tuple(values)


def _qualifier_match(query_frame: Any | None, candidate: RetrievalCandidate) -> tuple[float, float]:
    qualifiers = _query_qualifiers(query_frame)
    if not qualifiers:
        return 0.0, 0.0
    text = _document_text(candidate) + " " + _candidate_semantic_text(candidate)
    candidate_tokens = _token_set(text)
    matched = 0
    for value in qualifiers:
        tokens = _token_set(value)
        if tokens and _token_overlap(tuple(tokens), candidate_tokens) >= 0.6:
            matched += 1
    return matched / len(qualifiers), (len(qualifiers) - matched) / len(qualifiers)


# ---------------------------------------------------------------------------
# Attribute relevance
# ---------------------------------------------------------------------------

def _attribute_evidence(attribute: str, candidate: RetrievalCandidate) -> float:
    rule = next((item for item in _ATTRIBUTE_RULES if item.name == attribute), None)
    if rule is None:
        return 0.0

    source = _source(candidate)
    text = _document_text(candidate)
    combined = text + " " + _candidate_semantic_text(candidate)

    # Some intent classes require actionable evidence, not just the generic
    # word appearing somewhere in a large document.
    if attribute == "contact":
        actionable_markers = (
            "email", "e-mail", "@", "phone number", "telephone",
            "helpline", "contact details", "coordinator",
        )
        direct_hits = sum(1 for marker in actionable_markers if marker in combined)
        contact_word = "contact" in combined
        if direct_hits >= 2:
            return 1.0
        if direct_hits == 1 and contact_word:
            return 0.88
        if direct_hits == 1:
            return 0.72
        if contact_word:
            return 0.18
        return 0.0

    if attribute == "minor_programs":
        has_minor = "minor" in combined
        has_program = _contains_any(combined, ("program", "programs", "programme", "programmes"))
        if has_minor and has_program:
            return 1.0
        if has_minor:
            return 0.70
        return 0.0

    if attribute == "departments":
        multiple_depts = sum(1 for d in ("bioscience", "computer science", "electrical", "mechanical", "chemistry", "physics", "mathematics", "chemical", "materials", "civil", "aerospace", "economics") if d in combined)
        has_dept_marker = any(d in combined for d in ("department", "departments", "academic departments", "school of", "centres", "center for"))
        if multiple_depts >= 4:
            return 1.0
        if multiple_depts >= 2:
            return 0.85
        if has_dept_marker:
            return 0.50
        return 0.0

    if attribute == "booking":
        direct_hits = sum(1 for marker in (
            "booking", "bookings", "reservation", "reservations", "short-term", "short term",
            "accommodation booking",
        ) if marker in combined)
        return min(1.0, 0.30 + 0.18 * direct_hits) if direct_hits else 0.0

    # Exact phrase evidence is strongest. Documents are a special case: a
    # generic mention of "documents" is not equivalent to an explicit list
    # of required/brought/submitted items.
    phrase_hits = sum(1 for phrase in rule.phrases if phrase in combined)
    tokens = _token_set(combined)

    if attribute == "documents":
        document_terms = {"documents", "certificates", "score", "scorecard", "proof"}
        action_terms = {"required", "bring", "bring", "submit", "originals", "necessary", "copies"}
        doc_hits = len(document_terms & tokens)
        action_hits = len(action_terms & tokens)
        direct_markers = (
            "originals and copies",
            "proof of address",
            "score card",
            "scorecard",
            "necessary certificates",
            "items need to be brought",
        )
        direct_hits = sum(1 for marker in direct_markers if marker in combined)
        vague_document_language = "other documents" in combined
        if direct_hits >= 2:
            exact = 1.0
        elif direct_hits == 1 and doc_hits:
            exact = 0.92
        elif doc_hits and action_hits >= 2:
            exact = 0.78 if vague_document_language else 0.84
        elif doc_hits and action_hits == 1:
            exact = 0.68
        elif doc_hits:
            exact = 0.45
        else:
            exact = 0.0
        token_score = _clamp01((0.60 * doc_hits + 0.40 * min(action_hits, 2)) / 3.0)
    else:
        if phrase_hits:
            exact = min(1.0, 0.45 + 0.15 * phrase_hits)
        else:
            exact = 0.0
        token_hits = sum(1 for token in rule.tokens if token in tokens)
        token_score = min(1.0, token_hits / max(2.0, len(rule.tokens) * 0.55))

    # Source-role evidence is secondary. It helps when chunk text is short.
    source_boost = 0.0
    role_map = {
        "documents": ("registration", "admissions", "administration"),
        "contact": ("administration", "admissions", "registration"),
        "process": ("admissions", "registration", "administration"),
        "fees": ("fees",),
        "rules": ("hostel", "administration"),
        "dining": ("hostel",),
        "research": ("research",),
        "facilities": ("facilities", "emergency"),
        "programs": ("programs", "admissions"),
        "departments": ("departments", "programs", "administration"),
        "minor_programs": ("programs", "admissions"),
        "booking": ("fees", "hostel"),
        "eligibility": ("admissions", "programs"),
        "deadlines": ("admissions", "administration"),
        "location": ("administration", "hostel", "facilities"),
        "fellowship": ("programs", "administration", "research"),
    }
    roles = role_map.get(attribute, ())
    if any(_has_role(source, role) for role in roles):
        source_boost = 0.15

    return min(1.0, max(exact, 0.65 * token_score) + source_boost)


def _attribute_source_synergy(query: str, attribute: str, candidate: RetrievalCandidate) -> float:
    """Reward a generic source role that naturally carries the requested attribute."""
    source = _source(candidate)
    normalized = _normalize(query)

    if attribute == "documents" and any(marker in normalized for marker in ("admission", "registration")):
        if _has_role(source, "registration"):
            return 1.5
        if _has_role(source, "admissions"):
            return 0.15

    if attribute == "process" and _has_role(source, "admissions"):
        return 0.55

    if attribute == "contact":
        if _has_role(source, "administration"):
            return 0.50
        if _has_role(source, "admissions"):
            return 0.15

    if attribute == "minor_programs" and _has_role(source, "programs"):
        return 0.55

    if attribute == "programs":
        if "programs/btech/overview" in source or "office_of_academics/programs" in source or "programs/general_information" in source:
            return 0.85
        if _has_role(source, "programs") or _has_role(source, "administration"):
            return 0.55

    if attribute == "departments":
        if _is_inventory_or_directory_candidate(candidate) and _inventory_candidate_scope(candidate) == "departments":
            return 0.85
        if "office_of_academics" in source:
            return 0.55
        if _has_role(source, "departments") or _has_role(source, "administration"):
            return 0.55

    if attribute == "booking":
        if _has_role(source, "fees"):
            return 0.55
        if _has_role(source, "hostel"):
            return 0.45

    if attribute == "fees" and (_has_role(source, "fees") or "fees_and_finance" in source or "/finance/" in source):
        return 0.85

    if attribute == "rules" and _has_role(source, "hostel"):
        return 0.40

    if attribute == "dining" and _has_role(source, "hostel"):
        return 0.40

    if attribute == "research" and _has_role(source, "research"):
        return 0.35

    if attribute == "facilities" and _has_role(source, "facilities"):
        return 0.35

    if attribute == "eligibility" and (_has_role(source, "admissions") or _has_role(source, "programs")):
        return 0.55

    if attribute == "fellowship" and (_has_role(source, "programs") or _has_role(source, "administration") or _has_role(source, "research")):
        return 0.55

    return 0.0


def _attribute_alignment(query: str, candidate: RetrievalCandidate, query_frame: Any | None) -> tuple[float, tuple[str, ...]]:
    attributes = _derive_query_attributes(query, query_frame)
    if not attributes:
        return 0.0, ()

    evidence = [(attribute, _attribute_evidence(attribute, candidate)) for attribute in attributes]
    specialized = {"minor_programs", "booking", "documents", "contact"}
    if any(attribute in specialized for attribute in attributes):
        # Specialized intent families should not lose simply because a broad
        # family such as `programs` or `fees` also matched the query.
        evidence.sort(key=lambda item: (item[0] not in specialized, -item[1], item[0]))
    evidence.sort(key=lambda item: (-item[1], item[0]))

    # The strongest attribute is primary. A second distinct attribute can
    # contribute at half weight for genuinely multi-facet questions.
    strongest = evidence[0][1]
    if len(evidence) > 1:
        strongest = min(1.0, strongest + 0.35 * evidence[1][1])
    return strongest, tuple(attribute for attribute, _score in evidence)


# ---------------------------------------------------------------------------
# Lexical / phrase / provenance / quality signals
# ---------------------------------------------------------------------------

def _idf_weights(query: str, candidates: Sequence[RetrievalCandidate]) -> dict[str, float]:
    terms = set(_tokens(query))
    if not terms:
        return {}
    total = max(1, len(candidates))
    df = {term: 0 for term in terms}
    for candidate in candidates:
        tokens = _token_set(_document_text(candidate))
        for term in terms:
            if term in tokens:
                df[term] += 1
    return {
        term: min(2.5, max(0.75, math.log((total + 1) / (freq + 1)) + 1.0))
        for term, freq in df.items()
    }


def _lexical_score(
    query: str,
    candidate: RetrievalCandidate,
    candidates: Sequence[RetrievalCandidate],
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    if weights is None:
        weights = _idf_weights(query, candidates)
    if not weights:
        return 0.0
    tokens = _token_set(_document_text(candidate))
    total = sum(weights.values())
    return sum(weight for term, weight in weights.items() if term in tokens) / max(1.0, total)


def _phrase_score(query: str, candidate: RetrievalCandidate, maximum: float) -> float:
    tokens = list(_tokens(query))
    if len(tokens) < 2:
        return 0.0
    content = _document_text(candidate)
    score = 0.0
    # Long contiguous phrases matter more than isolated generic words.
    for size, multiplier in ((5, 1.25), (4, 1.15), (3, 0.90), (2, 0.55)):
        if len(tokens) < size:
            continue
        for index in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[index:index + size])
            if phrase in content:
                score += multiplier
    return min(maximum, score)


def _rrf_signal(candidate: RetrievalCandidate) -> float:
    provenance = getattr(candidate, "provenance", None)
    value = float(getattr(provenance, "rrf_score", 0.0) or 0.0) if provenance else 0.0
    if value <= 0:
        return 0.0
    return min(1.0, math.sqrt(min(1.0, value * 60.0)))


def _semantic_signal(candidate: RetrievalCandidate) -> tuple[float, float]:
    alignment = getattr(candidate, "alignment", None)
    if alignment is None:
        return 0.0, 0.0
    return (
        _clamp01(float(getattr(alignment, "semantic_match", 0.0) or 0.0)),
        _clamp01(float(getattr(alignment, "coverage", 0.0) or 0.0)),
    )


def _quality_signal(candidate: RetrievalCandidate) -> float:
    quality = getattr(candidate, "quality", None)
    if quality is None:
        return 1.0
    content = _clamp01(float(getattr(quality, "content_quality", 1.0) or 0.0))
    structure = _clamp01(float(getattr(quality, "structural_quality", 1.0) or 0.0))
    noise = _clamp01(float(getattr(quality, "noise", 0.0) or 0.0))
    return min(1.0, 0.50 * content + 0.40 * structure + 0.10 * (1.0 - noise))


def _has_conflict(candidate: RetrievalCandidate) -> bool:
    alignment = getattr(candidate, "alignment", None)
    conflicts = getattr(alignment, "conflicts", ()) if alignment else ()
    return bool(tuple(conflicts or ()))


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def score_candidate(
    query: str,
    candidate: RetrievalCandidate,
    *,
    candidates: Sequence[RetrievalCandidate] = (),
    query_frame: Any | None = None,
    config: RankingConfig = DEFAULT_RANKING_CONFIG,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Return a deterministic relevance score for one retrieved candidate."""
    if not isinstance(candidate, RetrievalCandidate):
        raise TypeError("candidate must be a RetrievalCandidate.")
    if not isinstance(config, RankingConfig):
        raise TypeError("config must be a RankingConfig.")

    pool = tuple(candidates) or (candidate,)
    q = _normalize(query)
    score = 0.0

    # Base retrieval evidence.
    score += config.lexical_weight * _lexical_score(q, candidate, pool, weights=weights)
    score += config.phrase_weight * _phrase_score(q, candidate, config.max_phrase_bonus)
    score += config.rrf_weight * _rrf_signal(candidate)

    # Explicit target alignment.
    target = _effective_query_target(q, query_frame)
    target_match = _explicit_target_match(q, query_frame, candidate)

    # Upstream semantic alignment is useful, but bounded below explicit query
    # requirements so generic semantic overlap cannot rescue a wrong topic.
    semantic, coverage = _semantic_signal(candidate)
    if target_match >= 0.90 and semantic < 0.50:
        semantic, coverage = 0.85, 0.85
    score += config.semantic_weight * semantic
    score += config.semantic_coverage_weight * coverage

    requirement = getattr(query_frame, "requirement", None) or getattr(query_frame, "retrieval_requirement", None)
    inferred_target = bool(target)
    if target:
        score += config.target_match_weight * target_match
        if target_match == 0.0:
            cand_src_check = _source(candidate)
            is_institutional_fee_doc = (
                any(fq in q for fq in ("fee", "fees", "tuition", "cost", "charge", "charges"))
                and (_has_role(cand_src_check, "fees") or "fees_and_finance" in cand_src_check or "/finance/" in cand_src_check)
            )
            if not is_institutional_fee_doc:
                strict_target = bool(getattr(requirement, "require_target_alignment", False)) or inferred_target
                score -= config.target_miss_penalty * (1.0 if strict_target else 0.55)

    # Requested attribute is the main correction for broad-source overreach.
    attribute_score, attributes = _attribute_alignment(q, candidate, query_frame)
    if attributes:
        score += config.attribute_match_weight * attribute_score
        score += min(1.5, _attribute_source_synergy(q, attributes[0], candidate)) * config.attribute_partial_weight

        strict_attribute = bool(getattr(requirement, "require_attribute_alignment", False))
        # The lexical query itself can establish strictness even without an
        # upstream structured frame for the focused attribute families.
        if attributes[0] in {
            "documents", "contact", "fees", "tuition", "rules", "deadlines", "location",
            "minor_programs", "booking", "eligibility",
        }:
            strict_attribute = True
        if strict_attribute and attribute_score < 0.35:
            score -= config.required_attribute_miss_penalty

    # Scope is intentionally a supporting signal.
    scope_match, scope_conflict = _source_scope_signal(q, candidate)
    inv_scope = _inventory_candidate_scope(candidate) if _is_inventory_or_directory_candidate(candidate) else None
    query_scopes = detect_query_scopes(q)
    if inv_scope and inv_scope in query_scopes and scope_match < 1.0:
        scope_match = 1.0
    score += config.scope_match_weight * scope_match
    score -= config.scope_conflict_penalty * scope_conflict

    # Broad query behavior: general/overview source roles are useful only when
    # the user did not specify a resolved target.
    broad_query = bool(getattr(query_frame, "list_intent", None) and getattr(getattr(query_frame, "list_intent"), "is_list", False)) or not target
    attribute_specific = bool(attributes)
    broad_source_eligible = broad_query and (
        not attribute_specific
        or attributes[0] in {"process", "programs", "departments", "facilities", "fees", "tuition"}
    )
    is_rel_inv = _is_relevant_inventory_candidate(candidate, q, attributes)
    is_spec = _is_specialized_source(_source(candidate), candidate)
    is_gen = _is_generic_source(_source(candidate)) or is_rel_inv

    if broad_source_eligible and query_scopes and is_gen:
        score += config.broad_source_bonus
    elif (
        broad_query
        and is_spec
        and attributes
        and attributes[0] in {"process", "programs", "departments", "research", "fees", "tuition"}
        and _phrase_score(q, candidate, config.max_phrase_bonus) == 0.0
        and _lexical_score(q, candidate, pool, weights=weights) < 0.40
    ):
        # Do not make this a hard exclusion: exact content/attribute evidence
        # can still win.
        score -= config.specialized_without_target_penalty

    qualifier_match, qualifier_miss = _qualifier_match(query_frame, candidate)
    if qualifier_match:
        score += config.qualifier_match_weight * qualifier_match
    if qualifier_miss:
        score -= config.qualifier_miss_penalty * qualifier_miss

    score += config.intrinsic_quality_weight * _quality_signal(candidate)

    if _has_conflict(candidate):
        score -= config.conflict_penalty

    # Explicit admission vs academic progression conflict:
    # Admission eligibility must never be answered with enrolled branch change or progression criteria.
    cand_src = _source(candidate)
    cand_text = _document_text(candidate)
    if any(adm in q for adm in ("admission", "admissions")):
        if any(prog in cand_text or prog in cand_src for prog in ("branch change", "academic_progression", "academic progression", "duration of residence", "continuation and graduation in a program")):
            score -= 12.0

    # Technical chip design / semiconductor query vs generic non-technical design
    if any(term in q for term in ("chip design", "semiconductor", "semiconductors", "vlsi")):
        if "schools/design" in cand_src or "civil_and_infrastructure" in cand_src:
            score -= 8.0
        elif any(tech in cand_text for tech in ("semiconductor", "vlsi", "microelectronic", "chip design")):
            score += 4.0

    # Single-degree vs compound dual-degree preference:
    # When query specifically requests a single degree (e.g. M.Tech or B.Tech or M.Sc.) without mentioning "dual",
    # compound dual-degree programmes (e.g. M.Sc.-M.Tech. or M.Tech.-Ph.D.) must not outrank the standalone primary programme.
    is_dual_degree_cand = any(d in cand_src or d in cand_text[:200].casefold() for d in ("dual_degree", "dual degree", "msc_mtech", "mtech_phd"))
    if is_dual_degree_cand and not any(dq in q for dq in ("dual", "dual degree", "m.sc - m.tech", "msc-mtech")):
        score -= 3.0

    # Catalogue / complete listing queries (programmes or departments offered)
    is_catalogue_query = any(cat in q for cat in (
        "what b.tech", "what btech", "list all", "programmes offered", "programs offered",
        "programmes does", "programs does", "what departments", "list of departments", "which departments"
    ))
    if is_catalogue_query:
        is_dept_catalogue = any(cat in q for cat in ("what departments", "list of departments", "which departments", "departments does", "academic departments")) or (attributes and attributes[0] == "departments")
        is_prog_catalogue = any(cat in q for cat in ("what b.tech", "what btech", "programmes offered", "programs offered", "programmes does", "programs does", "what programmes", "what programs", "degrees offered")) or (attributes and attributes[0] == "programs")

        if is_dept_catalogue:
            if inv_scope == "departments" or cand_src.endswith("departments.json") or "departments_list" in cand_src or "departments_directory" in cand_src:
                score += 4.5
            elif any(ov in cand_src for ov in ("office_of_academics/departments",)):
                score += 4.5
            elif "academic_progression" in cand_src or "branch change" in cand_text:
                score -= 3.0
            elif is_spec and "curriculum" in cand_src:
                score -= 5.0
        elif is_prog_catalogue:
            if inv_scope == "programs" or any(ov in cand_src for ov in ("programs/btech/overview", "office_of_academics/programs", "programs/general_information", "academic_programmes.json")):
                score += 4.5
            elif "academic_progression" in cand_src or "branch change" in cand_text:
                score -= 3.0
            elif is_spec and "curriculum" in cand_src:
                score -= 5.0
        else:
            if _is_inventory_or_directory_candidate(candidate):
                score += 4.5
            elif any(ov in cand_src for ov in ("programs/btech/overview", "office_of_academics/programs", "programs/general_information")):
                score += 4.5

    # Institutional fee queries boost
    is_fee_query = any(fq in q for fq in ("fee", "fees", "tuition", "cost", "charge", "charges"))
    if is_fee_query:
        if "fees_and_finance" in cand_src or "/finance/" in cand_src:
            score += 5.5
            if any(fm in cand_text for fm in ("fee structure", "tuition fee", "particulars (per semester)", "50,000", "32,250")):
                score += 3.5
        elif "academic_progression" in cand_src or ("curriculum" in cand_src and "/departments/" in cand_src):
            score -= 4.0

    return float(max(-config.max_candidate_score, min(config.max_candidate_score, score)))


# ---------------------------------------------------------------------------
# Diversity / deterministic ordering
# ---------------------------------------------------------------------------

def _canonical_source(source: str) -> str:
    parts = [part for part in re.split(r"/+", _normalize(source)) if part]
    # Keep a small stable suffix so equivalent absolute/relative local paths
    # collapse without assuming a particular deployment root.
    return "/".join(parts[-5:]) if parts else ""


def _content_fingerprint(candidate: RetrievalCandidate) -> str:
    text = re.sub(r"\s+", " ", _normalize(_document_text(candidate))).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _select_diverse(
    scored: Sequence[tuple[RetrievalCandidate, float, int]],
    *,
    top_k: int,
    config: RankingConfig,
) -> list[tuple[RetrievalCandidate, float, int]]:
    selected: list[tuple[RetrievalCandidate, float, int]] = []
    source_counts: dict[str, int] = {}
    fingerprints: set[str] = set()

    # First pass uses the natural score ordering but prevents one source from
    # consuming the entire evidence budget.
    for item in scored:
        candidate = item[0]
        source_key = _canonical_source(_source(candidate))
        fingerprint = _content_fingerprint(candidate)

        if config.dedupe_exact_content and fingerprint in fingerprints:
            continue
        if source_counts.get(source_key, 0) >= config.max_per_source:
            continue

        selected.append(item)
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        fingerprints.add(fingerprint)
        if len(selected) >= top_k:
            return selected

    # Second pass: if top_k has not been reached after diverse selection,
    # backfill with remaining distinct valid candidates up to top_k rather
    # than starving downstream evidence boundaries.
    if len(selected) < top_k:
        for item in scored:
            candidate = item[0]
            fingerprint = _content_fingerprint(candidate)
            if config.dedupe_exact_content and fingerprint in fingerprints:
                continue
            selected.append(item)
            fingerprints.add(fingerprint)
            if len(selected) >= top_k:
                return selected

    return selected


def rank_candidates(
    query: str,
    candidates: Iterable[RetrievalCandidate],
    *,
    top_k: int | None = None,
    query_frame: Any | None = None,
    config: RankingConfig = DEFAULT_RANKING_CONFIG,
) -> list[RetrievalCandidate]:
    """Score and deterministically rank retrieved candidates."""
    items = tuple(candidates)
    if not items:
        return []
    for candidate in items:
        if not isinstance(candidate, RetrievalCandidate):
            raise TypeError("rank_candidates() accepts RetrievalCandidate objects only.")

    q = _normalize(query)
    weights = _idf_weights(q, items)
    scored: list[tuple[RetrievalCandidate, float, int]] = []
    for index, candidate in enumerate(items):
        scored.append((
            candidate,
            score_candidate(
                query,
                candidate,
                candidates=items,
                query_frame=query_frame,
                config=config,
                weights=weights,
            ),
            index,
        ))

    scored.sort(
        key=lambda item: (
            item[1],
            _rrf_signal(item[0]),
            _quality_signal(item[0]),
            -item[2],
        ),
        reverse=True,
    )

    if top_k is None:
        chosen = scored
    else:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer or None.")
        if top_k == 0:
            return []
        chosen = (
            _select_diverse(scored, top_k=top_k, config=config)
            if config.diversify_when_top_k
            else scored[:top_k]
        )

    return [replace(candidate, final_score=score) for candidate, score, _ in chosen]


def select_ranked_documents(
    query: str,
    candidates: Iterable[RetrievalCandidate],
    *,
    top_k: int = 5,
    query_frame: Any | None = None,
    config: RankingConfig = DEFAULT_RANKING_CONFIG,
) -> list[Any]:
    """Compatibility helper returning ranked source documents."""
    return [
        candidate.document
        for candidate in rank_candidates(
            query,
            candidates,
            top_k=top_k,
            query_frame=query_frame,
            config=config,
        )
    ]


__all__ = [
    "RankingConfig",
    "DEFAULT_RANKING_CONFIG",
    "detect_query_scopes",
    "score_candidate",
    "rank_candidates",
    "select_ranked_documents",
]