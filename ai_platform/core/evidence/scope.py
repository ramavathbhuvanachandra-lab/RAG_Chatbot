"""Generic evidence scope and conflict filtering.

This module is part of the reusable college-AI core.

Responsibilities
----------------
- Detect explicit semantic scope expressed by a query and a candidate.
- Reject evidence only when an explicit, material conflict exists.
- Preserve useful narrow evidence for broad questions; completeness belongs
  to the evidence-coverage layer.
- Prefer structured semantic metadata when available.
- Fall back to conservative text-derived signals when metadata is absent.

Non-responsibilities
--------------------
- retrieval
- reranking
- answer generation
- institution-specific vocabulary
- LLM calls

The design intentionally treats unknown terms as neutral. Lack of a
recognized scope signal is not evidence of a conflict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Generic college-domain scope vocabulary.
# These are product-level concepts, not institution-specific entities.
# ---------------------------------------------------------------------------

_TOPIC_ALIASES: dict[str, frozenset[str]] = {
    "admission": frozenset({
        "admission", "admissions", "eligibility", "eligible",
        "qualification", "qualifications", "requirement", "requirements",
    }),
    "application": frozenset({
        "application", "applications", "apply", "applying", "submission",
        "submit", "registration", "register", "documents", "certificates",
    }),
    "fees": frozenset({
        "fee", "fees", "cost", "costs", "charge", "charges", "rent",
        "tuition", "payment", "payments",
    }),
    "hostel": frozenset({
        "hostel", "hostels", "accommodation", "residence", "residential",
        "dorm", "dormitory",
    }),
    "research": frozenset({
        "research", "researches", "research area", "research areas",
        "research theme", "research themes", "research group", "research groups",
    }),
    "programs": frozenset({
        "program", "programs", "programme", "programmes", "degree", "degrees",
        "course", "courses", "curriculum",
    }),
    "facilities": frozenset({
        "facility", "facilities", "amenity", "amenities", "infrastructure",
    }),
    "faculty": frozenset({
        "faculty", "professor", "professors", "faculty member", "faculty members",
    }),
    "rules": frozenset({
        "rule", "rules", "regulation", "regulations", "policy", "policies",
    }),
    "contact": frozenset({
        "contact", "contacts", "email", "phone", "telephone", "office",
    }),
}

_MODE_ALIASES: dict[str, frozenset[str]] = {
    "regular": frozenset({"regular", "full time", "full-time"}),
    "part_time": frozenset({"part time", "part-time"}),
    "sponsored": frozenset({"sponsored", "sponsor"}),
    "external": frozenset({"external"}),
    "executive": frozenset({"executive"}),
}

_PROGRAM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:b|m)\s*\.?\s*tech\b", re.I),
    re.compile(r"\b(?:b|m)\s*\.?\s*sc\b", re.I),
    re.compile(r"\bph\s*\.?\s*d\b", re.I),
    re.compile(r"\b(?:b|m)\s*\.?\s*s\b", re.I),
    re.compile(r"\b(?:mba|mca|bba|bca|bed|med)\b", re.I),
    re.compile(r"\b(?:doctorate|doctoral|masters?|bachelors?)\b", re.I),
)

_ORG_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:department|school|centre|center)\s+of\s+(.+?)"
        r"(?=\s+(?:research|researches|admission|admissions|eligibility|"
        r"requirements?|program|programs|programme|programmes|course|courses|"
        r"facilities?|faculty|fees?|rules?|regulations?|contact|office)\b|[,.!?;:]|$)",
        re.I,
    ),
    re.compile(
        r"\b(?:in|at|from|within|under)\s+(?:the\s+)?(.+?)"
        r"(?=\s+(?:research|researches|admission|admissions|eligibility|"
        r"requirements?|program|programs|programme|programmes|course|courses|"
        r"facilities?|faculty|fees?|rules?|regulations?|contact|office)\b|[,.!?;:]|$)",
        re.I,
    ),
)

_YEAR_RANGE_RE = re.compile(r"\b(20\d{2})\s*[-/]\s*(20\d{2})\b")
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

_STOP_WORDS = frozenset({
    "the", "a", "an", "for", "to", "of", "in", "at", "from", "and",
    "or", "with", "about", "what", "which", "how", "can", "could", "do",
    "does", "is", "are", "be", "on", "by", "within", "under", "through",
    "student", "students", "college", "university", "institute",
})


def _normalize(value: Any) -> str:
    text = str(value or "").casefold()
    replacements = {
        "b.tech.": "btech",
        "b.tech": "btech",
        "m.tech.": "mtech",
        "m.tech": "mtech",
        "m.sc.": "msc",
        "m.sc": "msc",
        "ph.d.": "phd",
        "ph.d": "phd",
        "b.s.": "bs",
        "b.s": "bs",
        "m.s.": "ms",
        "m.s": "ms",
        "_": " ",
        "/": " ",
        "-": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^\w\s.%]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return [token for token in _normalize(text).split() if token and token not in _STOP_WORDS]


def _term_present(text: str, term: str) -> bool:
    normalized = _normalize(text)
    term_normalized = _normalize(term)
    if not normalized or not term_normalized:
        return False
    return re.search(r"(?<!\w)" + re.escape(term_normalized) + r"(?!\w)", normalized) is not None


def _extract_topics(text: str) -> frozenset[str]:
    normalized = _normalize(text)
    if not normalized:
        return frozenset()
    found: set[str] = set()
    for topic, aliases in _TOPIC_ALIASES.items():
        if any(_term_present(normalized, alias) for alias in aliases):
            found.add(topic)
    return frozenset(found)


def _canonical_program(value: str) -> str:
    normalized = _normalize(value)
    normalized = re.sub(r"\b(b|m)\s*tech\b", lambda m: m.group(1) + "tech", normalized)
    normalized = normalized.replace("ph d", "phd")
    normalized = normalized.replace("m sc", "msc")
    normalized = normalized.replace("b sc", "bsc")
    normalized = normalized.replace("m s", "ms")
    normalized = normalized.replace("b s", "bs")
    return normalized


def _extract_programs(text: str) -> frozenset[str]:
    normalized = _normalize(text)
    found: set[str] = set()
    if not normalized:
        return frozenset()
    for pattern in _PROGRAM_PATTERNS:
        for match in pattern.finditer(normalized):
            found.add(_canonical_program(match.group(0)))
    # Catch normalized compact degree tokens directly.
    for token in normalized.split():
        if token in {"btech", "mtech", "msc", "bsc", "phd", "bs", "ms", "mba", "mca", "bba", "bca", "bed", "med"}:
            found.add(token)
    return frozenset(found)


def _clean_org(value: str) -> str:
    normalized = _normalize(value)
    parts = [p for p in normalized.split() if p not in _STOP_WORDS]
    return " ".join(parts[:8]).strip()


def _extract_orgs(text: str) -> frozenset[str]:
    normalized = _normalize(text)
    if not normalized:
        return frozenset()
    found: set[str] = set()
    for pattern in _ORG_PATTERNS:
        for match in pattern.finditer(normalized):
            cleaned = _clean_org(match.group(1))
            if cleaned:
                found.add(cleaned)
    return frozenset(found)


def _extract_modes(text: str) -> frozenset[str]:
    found: set[str] = set()
    normalized = _normalize(text)
    for mode, aliases in _MODE_ALIASES.items():
        if any(_term_present(normalized, alias) for alias in aliases):
            found.add(mode)
    return frozenset(found)


def _extract_years(text: str) -> frozenset[int]:
    raw = str(text or "").casefold()
    ranges = {(int(m.group(1)), int(m.group(2))) for m in _YEAR_RANGE_RE.finditer(raw)}
    range_years = {year for pair in ranges for year in pair}
    years = {int(match.group(1)) for match in _YEAR_RE.finditer(raw)}
    # Keep endpoint years as explicit years too; this is useful for ordinary
    # year queries while exact ranges retain their own stricter check.
    return frozenset(years | range_years)


def _extract_year_ranges(text: str) -> frozenset[tuple[int, int]]:
    raw = str(text or "").casefold()
    return frozenset(
        (int(m.group(1)), int(m.group(2)))
        for m in _YEAR_RANGE_RE.finditer(raw)
    )


def _structured_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value} if value.strip() else set()
    if isinstance(value, Mapping):
        values: list[Any] = []
        for key in ("name", "value", "id", "label"):
            if key in value:
                values.append(value[key])
        return {str(item).strip() for item in values if str(item or "").strip()}
    try:
        return {str(item).strip() for item in value if str(item or "").strip()}
    except TypeError:
        return {str(value).strip()} if str(value or "").strip() else set()


def _field(source: Any, name: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


@dataclass(frozen=True, slots=True)
class ScopeSignals:
    topics: frozenset[str] = field(default_factory=frozenset)
    programs: frozenset[str] = field(default_factory=frozenset)
    organizations: frozenset[str] = field(default_factory=frozenset)
    modes: frozenset[str] = field(default_factory=frozenset)
    years: frozenset[int] = field(default_factory=frozenset)
    year_ranges: frozenset[tuple[int, int]] = field(default_factory=frozenset)

    @property
    def is_explicit(self) -> bool:
        return bool(
            self.topics
            or self.programs
            or self.organizations
            or self.modes
            or self.years
            or self.year_ranges
        )


@dataclass(frozen=True, slots=True)
class ScopeDecision:
    compatible: bool
    hard_conflict: bool
    reasons: tuple[str, ...] = ()
    query_signals: ScopeSignals = ScopeSignals()
    document_signals: ScopeSignals = ScopeSignals()


def signals_from_text(text: str) -> ScopeSignals:
    """Extract conservative scope signals from natural-language text."""
    return ScopeSignals(
        topics=_extract_topics(text),
        programs=_extract_programs(text),
        organizations=_extract_orgs(text),
        modes=_extract_modes(text),
        years=_extract_years(text),
        year_ranges=_extract_year_ranges(text),
    )


def signals_from_semantic(source: Any) -> ScopeSignals:
    """Build signals from a Query/DocumentMeaning-like structured object.

    Canonical ``Query.target`` is part of the structured contract and must be
    respected even when the original user wording is underspecified. A target
    with an explicit generic entity type is mapped to the corresponding scope
    dimension without introducing institution-specific knowledge.
    """
    topics = {str(v).casefold().strip() for v in _structured_values(_field(source, "topics")) if str(v).strip()}
    programs = {_canonical_program(v) for v in _structured_values(_field(source, "programs"))}
    organizations = {_clean_org(v) for v in _structured_values(_field(source, "entities")) if _clean_org(v)}
    modes = {str(v).casefold().strip().replace("-", "_") for v in _structured_values(_field(source, "qualifiers"))}
    scope_value = _field(source, "scope")
    topics.update(str(v).casefold().strip() for v in _structured_values(scope_value) if str(v).strip())

    target = _field(source, "target")
    target_text = str(_field(target, "text", "") or "").strip()
    target_type = str(_field(target, "entity_type", "") or "").casefold().replace("-", "_").strip()
    if target_text:
        if target_type in {"program", "programme", "degree", "course", "academic_program", "academic_programme"}:
            programs.add(_canonical_program(target_text))
        elif target_type in {
            "department", "school", "center", "centre", "organization", "organisation",
            "office", "unit", "faculty",
        }:
            organizations.add(_clean_org(target_text))

    return ScopeSignals(
        topics=frozenset(topics),
        programs=frozenset(programs),
        organizations=frozenset(organizations),
        modes=frozenset(modes & set(_MODE_ALIASES)),
        years=frozenset(),
        year_ranges=frozenset(),
    )


def merge_signals(text_signals: ScopeSignals, semantic_signals: ScopeSignals) -> ScopeSignals:
    return ScopeSignals(
        topics=frozenset(text_signals.topics | semantic_signals.topics),
        programs=frozenset(text_signals.programs | semantic_signals.programs),
        organizations=frozenset(text_signals.organizations | semantic_signals.organizations),
        modes=frozenset(text_signals.modes | semantic_signals.modes),
        years=frozenset(text_signals.years | semantic_signals.years),
        year_ranges=frozenset(text_signals.year_ranges | semantic_signals.year_ranges),
    )


def _signals_for_document(document: Any) -> ScopeSignals:
    """Extract document scope without allowing metadata to contradict text.

    The document content is the primary source of explicit scope. Structured
    metadata may enrich dimensions that the text does not expose, but it must
    not override a concrete textual program, organisation, mode, or topic.
    This prevents stale/incorrect metadata from laundering an explicitly
    conflicting document into a compatible candidate.
    """
    content = str(getattr(document, "page_content", "") or "")
    metadata = getattr(document, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    text_signals = signals_from_text(content)

    semantic_candidates = (
        metadata.get("meaning"),
        metadata.get("document_meaning"),
        metadata.get("semantic_alignment"),
    )
    semantic = ScopeSignals()
    for candidate in semantic_candidates:
        if candidate is not None:
            semantic = merge_signals(semantic, signals_from_semantic(candidate))

    # Text is authoritative whenever it exposes an explicit dimension.
    return ScopeSignals(
        topics=text_signals.topics or semantic.topics,
        programs=text_signals.programs or semantic.programs,
        organizations=text_signals.organizations or semantic.organizations,
        modes=text_signals.modes or semantic.modes,
        years=text_signals.years or semantic.years,
        year_ranges=text_signals.year_ranges or semantic.year_ranges,
    )


# Topics such as fees/contact are generally attributes, while admission,
# hostel, research, programs, facilities, faculty and rules represent broader
# domain scopes. When a query names multiple domains, a candidate that exposes
# a different explicit domain is a material scope mismatch even if it shares an
# attribute like ``fees`` with the query.
_DOMAIN_TOPICS = frozenset({
    "admission",
    "hostel",
    "research",
    "programs",
    "facilities",
    "faculty",
    "rules",
})


def _domain_topics(signals: ScopeSignals) -> frozenset[str]:
    return frozenset(signals.topics & _DOMAIN_TOPICS)


def _same_identity(query_values: Iterable[str], document_values: Iterable[str]) -> bool:
    query_list = [v for v in query_values if v]
    doc_list = [v for v in document_values if v]
    for q in query_list:
        q_tokens = q.split()
        for d in doc_list:
            d_tokens = d.split()
            if q == d:
                return True
            if len(q_tokens) <= len(d_tokens):
                for i in range(len(d_tokens) - len(q_tokens) + 1):
                    if d_tokens[i:i + len(q_tokens)] == q_tokens:
                        return True
    return False


def assess_scope(
    query: Any,
    document: Any,
    *,
    query_semantics: Any = None,
) -> ScopeDecision:
    """Assess whether a candidate evidence document conflicts with the query.

    Only explicit material conflicts are hard failures. Broad questions may
    still legitimately use narrow evidence; completeness is assessed later.
    """
    query_text = query if isinstance(query, str) else str(_field(query, "original_query", "") or _field(query, "normalized_query", ""))
    query_signals = signals_from_text(query_text)
    if query_semantics is not None:
        query_signals = merge_signals(query_signals, signals_from_semantic(query_semantics))
    elif not isinstance(query, str):
        query_signals = merge_signals(query_signals, signals_from_semantic(query))

    document_signals = _signals_for_document(document)
    reasons: list[str] = []

    # Explicit program mismatch is a hard conflict.
    if query_signals.programs and document_signals.programs:
        if not (query_signals.programs & document_signals.programs):
            reasons.append("explicit_program_mismatch")

    # Explicit organization mismatch is a hard conflict only when both sides
    # name organizations and none match. Absence of a query organization is
    # intentionally permissive for broad questions.
    if query_signals.organizations and document_signals.organizations:
        if not _same_identity(query_signals.organizations, document_signals.organizations):
            reasons.append("explicit_organization_mismatch")

    # When both sides explicitly name incompatible semantic topics, reject.
    # Prefer domain-topic agreement over incidental shared attributes such as
    # ``fees``. A query for hostel fees should not become compatible with an
    # admissions-fee document merely because both contain the word "fees".
    query_domains = _domain_topics(query_signals)
    document_domains = _domain_topics(document_signals)
    if query_domains and document_domains:
        if not (query_domains & document_domains):
            reasons.append("explicit_topic_mismatch")
    elif query_signals.topics and document_signals.topics:
        if not (query_signals.topics & document_signals.topics):
            reasons.append("explicit_topic_mismatch")

    # Explicit admission mode mismatch.
    if query_signals.modes and document_signals.modes:
        if not (query_signals.modes & document_signals.modes):
            reasons.append("explicit_mode_mismatch")

    # Explicit year range mismatch. A query with an exact academic year range
    # should not silently accept a different explicit range.
    if query_signals.year_ranges and document_signals.year_ranges:
        if not any(q == d for q in query_signals.year_ranges for d in document_signals.year_ranges):
            reasons.append("explicit_year_range_mismatch")
    elif query_signals.years and document_signals.years:
        if not (query_signals.years & document_signals.years):
            # If the document has a year range that contains the requested year,
            # keep it compatible for ordinary year requests.
            if not any(start <= year <= end for year in query_signals.years for start, end in document_signals.year_ranges):
                reasons.append("explicit_year_mismatch")

    conflict = bool(reasons)
    return ScopeDecision(
        compatible=not conflict,
        hard_conflict=conflict,
        reasons=tuple(reasons),
        query_signals=query_signals,
        document_signals=document_signals,
    )


def is_scope_compatible(query: Any, document: Any, *, query_semantics: Any = None) -> bool:
    return assess_scope(query, document, query_semantics=query_semantics).compatible


def filter_scope_conflicts(
    query: Any,
    documents: Sequence[Any],
    *,
    query_semantics: Any = None,
) -> list[Any]:
    """Remove only explicit scope conflicts; preserve order."""
    return [
        document
        for document in documents
        if is_scope_compatible(query, document, query_semantics=query_semantics)
    ]
