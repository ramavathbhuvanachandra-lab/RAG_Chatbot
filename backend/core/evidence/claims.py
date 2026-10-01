"""Generic claim extraction and claim-to-evidence auditing.

E5: Claim extraction + claim/evidence audit for the reusable evidence layer.

Design goals
------------
* Institution agnostic: no college/program/source vocabulary is defined here.
* Structured-query first: the Query contract is authoritative.
* Evidence is decomposed into small, traceable units before matching.
* High-risk factual values require local attachment to the requested field.
* Relations require co-location of their semantic parts; separate mentions do
  not manufacture a relationship.
* Conflicting evidence is surfaced instead of silently choosing one value.
* Deterministic and dependency-light. No LLM calls.

The older answer-claim-audit and claim-context-filter modules are behavioral
references only. Their institution-specific detectors and routing are not
copied into this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Iterable, Sequence

from backend.core.query.models import (
    Constraint,
    NumericRequirement,
    Query,
    TemporalConstraint,
)
from backend.core.retrieval_contracts import RetrievalCandidate


# ---------------------------------------------------------------------------
# Normalization and tokenization
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}

_MONEY_RE = re.compile(r"(?:₹|rs\.?|inr)\s*\d[\d,]*(?:\.\d+)?", re.I)
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_RATIO_RE = re.compile(r"\b\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_ACADEMIC_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\s*[-–—]\s*(?:19|20)?\d{2}\b")
_DURATION_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|years?|semesters?)\b", re.I
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•▪◦]|\d+[.)])\s+(.+?)\s*$")
_HASH_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
_COLON_HEADING_RE = re.compile(r"^\s*([^.!?\n]{2,100}):\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"[\w₹%]+", re.UNICODE)
_SCOPE_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)+|[a-z0-9]{2,}", re.I)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "how", "i", "in", "is", "it", "its", "me", "of", "on", "or", "please",
    "should", "that", "the", "their", "there", "this", "those", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your", "tell", "give", "get", "about", "into", "than",
}


def normalize_text(text: Any) -> str:
    value = str(text or "")
    for word, number in _NUMBER_WORDS.items():
        value = re.sub(rf"\b{re.escape(word)}\b", number, value, flags=re.I)
    value = value.casefold().replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(normalize_text(text))
        if len(token) > 2 and token not in _STOPWORDS
    }


def _scope_tokens(text: str) -> set[str]:
    """Tokenize local scope while preserving dotted/hyphenated identifiers."""
    return {
        token.casefold()
        for token in _SCOPE_TOKEN_RE.findall(normalize_text(text))
        if len(token) > 2 and token.casefold() not in _STOPWORDS
    }


def _overlap(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _dedupe_preserve(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = str(value or "").strip().casefold()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return tuple(result)


# ---------------------------------------------------------------------------
# Factual markers
# ---------------------------------------------------------------------------

def _normalize_marker(marker: str) -> str:
    return normalize_text(marker)


def _normalize_money(marker: str) -> str:
    value = _normalize_marker(marker)
    value = re.sub(r"^(?:inr|rs\.?|₹)\s*", "rs ", value)
    return re.sub(r"\s+", " ", value).strip()


def factual_markers(text: str) -> dict[str, tuple[str, ...]]:
    normalized = normalize_text(text)
    return {
        "money": _dedupe_preserve(_normalize_money(m) for m in _MONEY_RE.findall(normalized)),
        "percentages": _dedupe_preserve(_normalize_marker(m) for m in _PERCENT_RE.findall(normalized)),
        "ratios": _dedupe_preserve(_normalize_marker(m) for m in _RATIO_RE.findall(normalized)),
        "years": _dedupe_preserve(_YEAR_RE.findall(normalized)),
        "academic_years": _dedupe_preserve(_normalize_marker(m) for m in _ACADEMIC_YEAR_RE.findall(normalized)),
        "durations": _dedupe_preserve(_normalize_marker(m) for m in _DURATION_RE.findall(normalized)),
    }


def extract_factual_markers(text: str) -> tuple[str, ...]:
    groups = factual_markers(text)
    return _dedupe_preserve(marker for values in groups.values() for marker in values)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Claim:
    """One atomic information requirement derived from a Query."""

    claim_id: str
    text: str
    kind: str
    name: str | None = None
    value: Any = None
    unit: str | None = None
    operator: str | None = None
    required: bool = True
    importance: float = 1.0
    anchors: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not str(self.claim_id).strip():
            raise ValueError("Claim.claim_id cannot be empty.")
        if not str(self.text).strip():
            raise ValueError("Claim.text cannot be empty.")
        object.__setattr__(self, "claim_id", str(self.claim_id).strip())
        object.__setattr__(self, "text", " ".join(str(self.text).split()))
        object.__setattr__(self, "kind", str(self.kind).strip().casefold())
        object.__setattr__(self, "name", str(self.name).strip() if self.name else None)
        object.__setattr__(self, "unit", str(self.unit).strip().casefold() if self.unit else None)
        object.__setattr__(self, "operator", str(self.operator).strip() if self.operator else None)
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "importance", _clamp01(float(self.importance)))
        object.__setattr__(self, "anchors", _dedupe_preserve(self.anchors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "kind": self.kind,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "operator": self.operator,
            "required": self.required,
            "importance": self.importance,
            "anchors": list(self.anchors),
        }


@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    """Small, traceable evidence unit extracted from a retrieval candidate."""

    evidence_id: str
    text: str
    source: str
    position: int
    heading: str | None = None
    factual_markers: tuple[str, ...] = field(default_factory=tuple)
    token_set: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not str(self.evidence_id).strip():
            raise ValueError("EvidenceUnit.evidence_id cannot be empty.")
        if not str(self.text).strip():
            raise ValueError("EvidenceUnit.text cannot be empty.")
        object.__setattr__(self, "evidence_id", str(self.evidence_id).strip())
        object.__setattr__(self, "text", " ".join(str(self.text).split()))
        object.__setattr__(self, "source", str(self.source or "unknown").strip())
        object.__setattr__(self, "position", int(self.position))
        object.__setattr__(self, "heading", " ".join(str(self.heading).split()) if self.heading else None)
        object.__setattr__(self, "factual_markers", _dedupe_preserve(self.factual_markers))
        object.__setattr__(self, "token_set", frozenset(self.token_set or _tokens(self.text)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "text": self.text,
            "source": self.source,
            "position": self.position,
            "heading": self.heading,
            "factual_markers": list(self.factual_markers),
        }


@dataclass(frozen=True, slots=True)
class ClaimEvidenceMatch:
    claim_id: str
    evidence_id: str
    score: float
    status: str
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "evidence_id": self.evidence_id,
            "score": self.score,
            "status": self.status,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class ClaimAudit:
    claims: tuple[Claim, ...]
    matches: tuple[ClaimEvidenceMatch, ...]
    supported_claim_ids: tuple[str, ...]
    partial_claim_ids: tuple[str, ...]
    unsupported_claim_ids: tuple[str, ...]
    conflicting_claim_ids: tuple[str, ...]

    @property
    def status(self) -> str:
        if self.conflicting_claim_ids:
            return "conflict"
        required = {claim.claim_id for claim in self.claims if claim.required}
        supported = set(self.supported_claim_ids)
        partial = set(self.partial_claim_ids)
        if required and required <= supported:
            return "supported"
        if required & (supported | partial):
            return "partial"
        return "insufficient"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "claims": [claim.to_dict() for claim in self.claims],
            "matches": [match.to_dict() for match in self.matches],
            "supported_claim_ids": list(self.supported_claim_ids),
            "partial_claim_ids": list(self.partial_claim_ids),
            "unsupported_claim_ids": list(self.unsupported_claim_ids),
            "conflicting_claim_ids": list(self.conflicting_claim_ids),
        }


# ---------------------------------------------------------------------------
# Query -> claims
# ---------------------------------------------------------------------------

def _claim_id(kind: str, name: str, ordinal: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", f"{kind}_{name}".casefold()).strip("_")
    return f"{slug or kind}_{ordinal}"


def _query_anchors(query: Query) -> tuple[str, ...]:
    values: list[str] = []
    if query.target is not None:
        values.append(query.target.text)
        if query.target.entity_type:
            values.append(query.target.entity_type)
    values.extend(entity.name for entity in query.entities)
    values.extend(
        mention.normalized_text or mention.text for mention in query.entity_mentions
    )
    values.extend(concept.name for concept in query.concepts)
    values.extend(qualifier.name for qualifier in query.qualifiers)
    values.extend(qualifier.value for qualifier in query.qualifiers)
    values.extend(constraint.name for constraint in query.constraints)
    values.extend(str(constraint.value) for constraint in query.constraints)
    values.extend(relation.relation_type for relation in query.relations)
    if query.scope is not None:
        if query.scope.scope_type:
            values.append(query.scope.scope_type)
        if query.scope.value:
            values.append(query.scope.value)
    if query.request_type:
        values.append(query.request_type)
    return _dedupe_preserve(values)


def _claim_from_numeric(req: NumericRequirement, ordinal: int, anchors: Sequence[str]) -> Claim:
    unit = str(req.unit or "").strip().casefold() or None
    text = f"{req.name} {req.operator} {req.value}" + (f" {unit}" if unit else "")
    return Claim(
        claim_id=_claim_id("numeric", req.name, ordinal),
        text=text,
        kind="numeric",
        name=req.name,
        value=req.value,
        unit=unit,
        operator=req.operator,
        required=True,
        importance=req.confidence or 1.0,
        anchors=(req.name, *(a for a in anchors if a.casefold() != str(req.name).casefold())),
    )


def _claim_from_temporal(req: TemporalConstraint, ordinal: int, anchors: Sequence[str]) -> Claim:
    value = req.value or "-".join(part for part in (req.start, req.end) if part)
    return Claim(
        claim_id=_claim_id("temporal", req.kind, ordinal),
        text=f"{req.kind} {value}",
        kind="temporal",
        name=req.kind,
        value=value,
        required=True,
        importance=req.confidence or 1.0,
        anchors=(req.kind, value, *anchors),
    )


def extract_claims(query: Query) -> tuple[Claim, ...]:
    """Extract atomic claims directly from the structured Query contract."""
    if not isinstance(query, Query):
        raise TypeError("query must be a Query instance")

    claims: list[Claim] = []
    ordinal = 1
    anchors = _query_anchors(query)

    for relation in query.relations:
        claims.append(Claim(
            claim_id=_claim_id("relation", relation.relation_type, ordinal),
            text=f"{relation.subject_ref} {relation.relation_type} {relation.object_ref}",
            kind="relation",
            name=relation.relation_type,
            value=(relation.subject_ref, relation.object_ref),
            required=relation.required,
            importance=relation.confidence or 1.0,
            anchors=(relation.subject_ref, relation.relation_type, relation.object_ref, *anchors),
        ))
        ordinal += 1

    for qualifier in query.qualifiers:
        claims.append(Claim(
            claim_id=_claim_id("qualifier", qualifier.name, ordinal),
            text=f"{qualifier.name} = {qualifier.value}",
            kind="qualifier",
            name=qualifier.name,
            value=qualifier.value,
            required=qualifier.required,
            importance=qualifier.importance,
            anchors=(qualifier.name, qualifier.value, *anchors),
        ))
        ordinal += 1

    for constraint in query.constraints:
        claims.append(Claim(
            claim_id=_claim_id("constraint", constraint.name, ordinal),
            text=f"{constraint.name} {constraint.operator} {constraint.value}",
            kind="constraint",
            name=constraint.name,
            value=constraint.value,
            operator=constraint.operator,
            required=constraint.required,
            importance=constraint.importance,
            anchors=(constraint.name, str(constraint.value), *anchors),
        ))
        ordinal += 1

    for temporal in query.temporal_constraints:
        claims.append(_claim_from_temporal(temporal, ordinal, anchors))
        ordinal += 1

    for numeric in query.numeric_requirements:
        claims.append(_claim_from_numeric(numeric, ordinal, anchors))
        ordinal += 1

    if query.list_intent.is_list:
        item_type = query.list_intent.item_type or "item"
        list_anchors = (item_type, query.request_type or "", *anchors)
        claims.append(Claim(
            claim_id=_claim_id("list", item_type, ordinal),
            text=f"provide relevant {item_type} items",
            kind="list",
            name=item_type,
            required=True,
            importance=1.0,
            anchors=list_anchors,
        ))
        ordinal += 1

    # A plain question still has a structured request-level claim. This avoids
    # treating a query with only request_type/target information as empty.
    if not claims:
        # Keep the structured request type authoritative. The raw question is
        # retained as claim text for traceability, but is intentionally not
        # added as an all-or-nothing anchor: doing so makes a valid evidence
        # sentence fail merely because it does not repeat every query word.
        request_anchors = (query.request_type or "", *anchors)
        claims.append(Claim(
            claim_id="request_1",
            text=query.original_query,
            kind="request",
            name=query.request_type,
            required=True,
            importance=1.0,
            anchors=request_anchors,
        ))

    return tuple(claims)


# ---------------------------------------------------------------------------
# Evidence segmentation
# ---------------------------------------------------------------------------

def _looks_like_heading(line: str) -> bool:
    value = " ".join(str(line or "").split())
    if not value or len(value) > 100:
        return False
    # Numeric/parenthetical labels are often table qualifiers or value rows,
    # not structural headings. Preserve them as evidence text so temporal and
    # numeric facts are not detached from their local context.
    if value.startswith("(") and any(char.isdigit() for char in value):
        return False
    if _HASH_HEADING_RE.fullmatch(value) or _COLON_HEADING_RE.fullmatch(value):
        return True
    # Generic title-like line heuristic. It deliberately uses structure,
    # not a domain vocabulary. Avoid classifying sentence-like prose as a
    # heading when punctuation/verb-like length suggests content.
    if any(mark in value for mark in (". ", "? ", "! ")):
        return False
    words = value.split()
    if len(words) > 10:
        return False
    titleish = sum(1 for word in words if word[:1].isupper() or word.isupper())
    return len(words) <= 8 and titleish >= max(1, len(words) // 2)


def _split_text_lines(text: str) -> list[tuple[str, str | None]]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    output: list[tuple[str, str | None]] = []
    heading: str | None = None
    for line in lines:
        bullet = _BULLET_RE.match(line)
        if bullet:
            output.append((bullet.group(1).strip(), heading))
            continue
        hash_match = _HASH_HEADING_RE.fullmatch(line)
        colon_match = _COLON_HEADING_RE.fullmatch(line)
        if hash_match or colon_match or _looks_like_heading(line):
            heading = (hash_match.group(1) if hash_match else colon_match.group(1) if colon_match else line).strip().rstrip(":")
            continue
        output.append((line, heading))
    return output


def _split_paragraph(paragraph: str, heading: str | None) -> list[tuple[str, str | None]]:
    lines = [part for part in _split_text_lines(paragraph)]
    if len(lines) > 1:
        return lines
    text = " ".join(paragraph.split())
    if not text:
        return []
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if len(sentences) <= 1:
        return [(text, heading)]
    return [(sentence, heading) for sentence in sentences]


def _looks_like_value_label(text: str) -> bool:
    """Return True for short table/cell labels that can locally bind a value."""
    normalized = " ".join(str(text or "").split())
    if not normalized or len(normalized) > 100:
        return False
    marker_groups = factual_markers(normalized)
    if any(marker_groups.values()):
        return False
    tokens = _tokens(normalized)
    if not tokens or len(tokens) > 12:
        return False
    return bool(tokens & _EVIDENCE_FIELD_CUES)


def _merge_adjacent_value_lines(
    raw_units: Sequence[tuple[str, str | None]],
) -> list[tuple[str, str | None]]:
    """Merge generic table labels with immediately following value cells.

    DOCX-to-text loaders commonly serialize a table as one cell per line, for
    example ``Tuition Fee*`` followed by ``₹50,000/-``. Treating each cell as a
    separate evidence unit loses the local field/value relationship needed by
    high-risk numeric matching. This merge is deliberately generic: it only
    joins a short field-like label to an adjacent line containing an explicit
    factual marker.
    """
    merged: list[tuple[str, str | None]] = []
    index = 0
    while index < len(raw_units):
        current_text, current_heading = raw_units[index]
        if index + 1 < len(raw_units):
            next_text, next_heading = raw_units[index + 1]
            next_markers = factual_markers(next_text)
            if _looks_like_value_label(current_text) and next_markers:
                combined_heading = current_heading or next_heading
                merged.append((
                    f"{str(current_text).strip()} {str(next_text).strip()}".strip(),
                    combined_heading,
                ))
                index += 2
                continue
        merged.append((current_text, current_heading))
        index += 1
    return merged


def split_evidence_units(candidate: RetrievalCandidate) -> tuple[EvidenceUnit, ...]:
    """Split one retrieval candidate into deterministic evidence units."""
    if not isinstance(candidate, RetrievalCandidate):
        raise TypeError("candidate must be a RetrievalCandidate")

    content = str(getattr(candidate.document, "page_content", "") or "").strip()
    if not content:
        return ()

    metadata = getattr(candidate.document, "metadata", {}) or {}
    metadata_heading = metadata.get("heading") or metadata.get("section") or metadata.get("title")
    metadata_heading = str(metadata_heading).strip() if metadata_heading else None

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", content) if part.strip()]
    raw_units: list[tuple[str, str | None]] = []
    if paragraphs:
        for paragraph in paragraphs:
            units = _split_paragraph(paragraph, metadata_heading)
            raw_units.extend(units)
    else:
        raw_units = [(content, metadata_heading)]

    # Preserve local label/value attachment after line-level segmentation.
    raw_units = _merge_adjacent_value_lines(raw_units)

    units: list[EvidenceUnit] = []
    seen: set[str] = set()
    for position, (text, heading) in enumerate(raw_units):
        cleaned = " ".join(str(text).split())
        if len(cleaned) < 3:
            continue
        key = normalize_text(cleaned)
        if key in seen:
            continue
        seen.add(key)
        units.append(EvidenceUnit(
            evidence_id=f"{candidate.document_id}:{position}",
            text=cleaned,
            source=candidate.source,
            position=position,
            heading=heading,
            factual_markers=extract_factual_markers(cleaned),
            token_set=frozenset(_tokens(cleaned)),
        ))
    return tuple(units)


def extract_evidence_units(candidates: Sequence[RetrievalCandidate]) -> tuple[EvidenceUnit, ...]:
    """Flatten candidates into exact-content-deduplicated evidence units."""
    result: list[EvidenceUnit] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        for unit in split_evidence_units(candidate):
            key = (unit.source.casefold(), normalize_text(unit.text))
            if key in seen:
                continue
            seen.add(key)
            result.append(unit)
    return tuple(result)


# ---------------------------------------------------------------------------
# Claim/evidence matching
# ---------------------------------------------------------------------------

def _numeric_kind(unit: str | None) -> str:
    value = normalize_text(unit)
    if value in {"currency", "money", "monetary", "inr", "rs", "usd", "eur", "gbp"}:
        return "currency"
    if value in {"percent", "percentage", "%"}:
        return "percent"
    if value in {"score", "cgpa", "cpi", "ratio"}:
        return "ratio"
    if value in {"duration", "time"}:
        return "duration"
    if value in {"count", "number", "quantity"}:
        return "count"
    return "unknown"


def _local_windows(pattern: re.Pattern[str], text: str, radius: int = 95) -> list[str]:
    windows: list[str] = []
    for match in pattern.finditer(text):
        start = max(0, match.start() - radius)
        end = min(len(text), match.end() + radius)
        windows.append(normalize_text(text[start:end]))
    return windows


def _claim_context_tokens(claim: Claim) -> set[str]:
    name = normalize_text(claim.name or "")
    unit = normalize_text(claim.unit or "")
    ignored = {name, unit}
    return {
        token
        for anchor in claim.anchors
        if normalize_text(anchor) not in ignored
        for token in _tokens(anchor)
    }


def _field_tokens(claim: Claim) -> set[str]:
    return _tokens(claim.name or "")


_NUMERIC_FIELD_ALIASES = {
    "currency": {"fee", "fees", "charge", "charges", "cost", "costs", "rent", "amount", "price", "rate"},
    "percent": {"percent", "percentage", "marks", "score", "aggregate"},
    "ratio": {"score", "cgpa", "cpi", "ratio", "scale"},
    "duration": {"duration", "days", "weeks", "months", "years", "semesters", "stay", "period"},
    "count": {"count", "number", "quantity", "intake", "students", "items"},
}


_EVIDENCE_FIELD_CUES = frozenset(
    token
    for aliases in _NUMERIC_FIELD_ALIASES.values()
    for alias in aliases
    for token in _tokens(alias)
) | {
    "particular", "particulars", "amount", "value", "rate", "rates",
}


def _field_match_tokens(claim: Claim, kind: str) -> set[str]:
    """Build generic field aliases from the requested numeric kind."""
    tokens = _field_tokens(claim)
    tokens.update(
        alias
        for alias in _NUMERIC_FIELD_ALIASES.get(kind, set())
        if alias in tokens
    )
    # The alias vocabulary is only used to interpret the requested field's
    # generic measurement type; it never contains institution terminology.
    return tokens | {
        token
        for alias in _NUMERIC_FIELD_ALIASES.get(kind, set())
        for token in _tokens(alias)
    }


def _numeric_match(claim: Claim, evidence: EvidenceUnit) -> tuple[bool, tuple[str, ...]]:
    kind = _numeric_kind(claim.unit)
    raw = str(evidence.text)
    context_tokens = _claim_context_tokens(claim)
    field_tokens = _field_match_tokens(claim, kind)

    patterns = {
        "currency": _MONEY_RE,
        "percent": _PERCENT_RE,
        "ratio": _RATIO_RE,
        "duration": _DURATION_RE,
    }

    if kind == "count":
        patterns["count"] = re.compile(r"\b\d+(?:\.\d+)?\b")

    selected = patterns.get(kind) if kind != "unknown" else None
    candidates = ([(kind, _local_windows(selected, raw))] if selected else [
        (candidate_kind, _local_windows(pattern, raw))
        for candidate_kind, pattern in patterns.items()
    ])

    for candidate_kind, windows in candidates:
        if not windows:
            continue
        for window in windows:
            window_tokens = _tokens(window)
            field_ok = not field_tokens or bool(field_tokens & window_tokens)
            context_ok = not context_tokens or _overlap(context_tokens, window_tokens) >= 0.34
            if field_ok and context_ok:
                return True, (
                    f"{candidate_kind} value is locally attached to the requested field/context",
                )

    if kind == "currency" and not _MONEY_RE.search(raw):
        return False, ("no monetary evidence marker",)
    if kind == "percent" and not _PERCENT_RE.search(raw):
        return False, ("no percentage evidence marker",)
    if kind == "ratio" and not _RATIO_RE.search(raw):
        return False, ("no ratio/score evidence marker",)
    if kind == "duration" and not _DURATION_RE.search(raw):
        return False, ("no duration evidence marker",)
    return False, ("numeric evidence exists but is not locally attached to the requested field/context",)


def _temporal_match(claim: Claim, evidence: EvidenceUnit) -> tuple[bool, tuple[str, ...]]:
    value = normalize_text(str(claim.value or ""))
    if not value:
        return False, ("temporal claim has no value",)

    combined_text = " ".join(part for part in (evidence.heading or "", evidence.text) if part)
    markers = factual_markers(combined_text)
    if value in markers["academic_years"] or value in markers["years"]:
        return True, ("requested temporal value is explicitly present",)

    years = _YEAR_RE.findall(value)
    content = normalize_text(combined_text)
    if years and all(year in content for year in years):
        return True, ("requested temporal endpoint(s) are explicitly present",)

    return False, ("requested temporal value is absent",)


def _relation_match(claim: Claim, evidence: EvidenceUnit) -> tuple[bool, tuple[str, ...]]:
    if not isinstance(claim.value, tuple) or len(claim.value) != 2:
        return False, ("relation claim is structurally incomplete",)

    subject = normalize_text(str(claim.value[0]))
    relation = normalize_text(str(claim.name or ""))
    object_ = normalize_text(str(claim.value[1]))
    if not subject or not relation or not object_:
        return False, ("relation claim is structurally incomplete",)

    # Evaluate sentence-like windows independently. A subject in one sentence
    # and an object in another does not establish the relationship.
    segments = [part.strip() for part in _SENTENCE_SPLIT_RE.split(evidence.text) if part.strip()]
    if not segments:
        segments = [evidence.text]

    subject_tokens = _tokens(subject)
    object_tokens = _tokens(object_)
    relation_tokens = _tokens(relation)

    for segment in segments:
        tokens = _tokens(segment)
        subject_ok = not subject_tokens or bool(subject_tokens & tokens)
        object_ok = not object_tokens or bool(object_tokens & tokens)
        relation_ok = not relation_tokens or bool(relation_tokens & tokens)
        if subject_ok and object_ok and relation_ok:
            return True, ("relation subject, relation, and object are co-located",)

    overlap = _overlap(subject_tokens | relation_tokens | object_tokens, _tokens(evidence.text))
    if overlap >= 0.34:
        return False, ("relation is only partially represented",)
    return False, ("relation is not represented in evidence",)


def _value_signature(claim: Claim, evidence: EvidenceUnit) -> tuple[str, ...]:
    raw = evidence.text
    kind = _numeric_kind(claim.unit)
    if kind == "currency":
        return tuple(sorted(factual_markers(raw)["money"]))
    if kind == "percent":
        return tuple(sorted(factual_markers(raw)["percentages"]))
    if kind == "ratio":
        return tuple(sorted(factual_markers(raw)["ratios"]))
    if kind == "duration":
        return tuple(sorted(factual_markers(raw)["durations"]))
    if claim.kind == "temporal":
        markers = factual_markers(raw)
        return tuple(sorted(markers["academic_years"] + markers["years"]))
    return ()


def _token_family_overlap(left: Iterable[str], right: Iterable[str], prefix_size: int = 5) -> float:
    """Return overlap while allowing simple morphological variants.

    This is intentionally language-light and vocabulary-free: ``eligibility``
    and ``eligible`` share a stable prefix, as do singular/plural variants such
    as ``fee``/``fees``. It is only used for structured request labels, not for
    institution entities or program names.
    """
    a = set(left)
    b = set(right)
    if not a:
        return 0.0
    matched = 0
    for token in a:
        if token in b:
            matched += 1
            continue
        if len(token) >= prefix_size and any(
            len(other) >= prefix_size and token[:prefix_size] == other[:prefix_size]
            for other in b
        ):
            matched += 1
    return matched / len(a)


_REQUEST_FAMILY_CUES = {
    # Generic language families only; no institution/program vocabulary.
    "eligibility": {"eligible", "eligibility", "qualification", "qualifications", "criteria"},
    "fee": {"fee", "fees", "charge", "charges", "cost", "costs", "amount", "price", "rent"},
    "admission": {"admission", "admissions", "admit", "apply", "application", "enroll", "enrollment"},
    "registration": {"registration", "register", "registering"},
    "deadline": {"deadline", "closing", "due"},
    "process": {"process", "procedure", "steps"},
}

_REQUIREMENT_CUES = {
    "must", "required", "requirement", "requirements", "minimum",
    "qualifying", "qualification", "qualifications", "criteria",
}
_APPLICANT_CUES = {"applicant", "applicants", "candidate", "candidates"}


_FINANCIAL_REQUEST_NAMES = frozenset({
    "cost", "fee", "fees", "tuition", "amount", "price", "charges", "charge", "rent",
})

_FINANCIAL_GENERIC_TOKENS = frozenset({
    "cost", "costs", "fee", "fees", "amount", "amounts",
    "price", "prices", "charge", "charges", "rent", "rate", "rates",
    "rs", "inr",
})


def _financial_scope_tokens(claim: Claim) -> set[str]:
    """Return non-generic semantic anchors that define a financial scope."""
    tokens: set[str] = set()
    for anchor in claim.anchors:
        for token in _scope_tokens(anchor):
            if token in _FINANCIAL_GENERIC_TOKENS:
                continue
            tokens.add(token)
    return tokens


_FINANCIAL_LABEL_TOKENS = frozenset({
    "fee", "fees", "cost", "costs", "charge", "charges",
    "amount", "amounts", "price", "prices", "rate", "rates",
})


def _money_values_for_claim(claim: Claim, evidence: EvidenceUnit) -> tuple[str, ...]:
    """Return money values locally attached to the claim's requested scope."""
    combined = " ".join(
        part for part in (evidence.heading or "", evidence.text) if part
    )
    normalized = normalize_text(combined)
    matches = list(_MONEY_RE.finditer(normalized))
    if not matches:
        return ()

    scope_tokens = _financial_scope_tokens(claim)
    if not scope_tokens:
        return _dedupe_preserve(_normalize_money(match.group(0)) for match in matches)

    values: list[str] = []
    for match in matches:
        radius = 100
        left = max(0, match.start() - radius)
        right = min(len(normalized), match.end() + 60)
        window = normalized[left:right]
        value_start = match.start() - left

        # Find the generic financial field label nearest this value (for
        # example ``fee`` in ``tuition fee ₹50,000``). Then inspect only the
        # local label neighborhood. This prevents a nearby but different field
        # such as ``admission fee`` from borrowing the query's ``tuition``
        # scope merely because both occur in the same sentence.
        label_matches = list(
            re.finditer(
                r"\b(?:fee|fees|cost|costs|charge|charges|amount|amounts|price|prices|rate|rates)\b",
                window,
            )
        )
        if label_matches:
            label_match = min(
                label_matches,
                key=lambda item: abs(item.end() - value_start),
            )
            previous_labels = [
                item
                for item in label_matches
                if item.start() < label_match.start()
            ]
            previous_end = previous_labels[-1].end() if previous_labels else max(0, label_match.start() - 45)
            label_context = (
                window[previous_end: label_match.start()]
                + " "
                + window[label_match.end(): max(label_match.end(), value_start)]
            )
            local_scope = _scope_tokens(label_context)
            if scope_tokens & local_scope:
                values.append(_normalize_money(match.group(0)))
            # A conventional field label was found but does not match the
            # requested scope. Do not fall back to a wide window, because that
            # would allow a neighboring fee field in the same sentence/table
            # to be mistaken for the requested field.
            continue

        # Conservative fallback for prose without a conventional field label.
        nearest_scope = min(
            (
                abs(token_match.start() - value_start)
                for token in scope_tokens
                for token_match in re.finditer(rf"\b{re.escape(token)}\b", window)
            ),
            default=None,
        )
        if nearest_scope is not None and nearest_scope <= 45:
            values.append(_normalize_money(match.group(0)))

    return _dedupe_preserve(values)


def _financial_scope_signature(claim: Claim, evidence: EvidenceUnit) -> tuple[str, ...]:
    """Capture contextual scope around a money fact without encoding domain data."""
    combined = " ".join(
        part for part in (evidence.heading or "", evidence.text) if part
    )
    tokens = _tokens(combined)
    claim_tokens = _financial_scope_tokens(claim)

    # Remove generic financial language and obvious factual-marker tokens.
    filtered = {
        token
        for token in tokens
        if token not in _FINANCIAL_GENERIC_TOKENS
        and not token.isdigit()
        and not any(token in value.casefold().replace(",", "").split() for value in factual_markers(combined)["money"])
    }

    # Preserve explicit scope anchors such as a program, academic year,
    # student category, or other query-linked qualifier.
    if claim_tokens:
        filtered.update(tokens & claim_tokens)

    return tuple(sorted(filtered))


def _financial_scope_signature_for_value(claim: Claim, evidence: EvidenceUnit, value: str) -> tuple[str, ...]:
    """Derive a local context signature centered on one monetary marker."""
    combined = " ".join(part for part in (evidence.heading or "", evidence.text) if part)
    normalized = normalize_text(combined)

    money_match = None
    for match in _MONEY_RE.finditer(normalized):
        if _normalize_money(match.group(0)) == _normalize_money(value):
            money_match = match
            break
    if money_match is None:
        return _financial_scope_signature(claim, evidence)

    radius = 120
    left = max(0, money_match.start() - radius)
    right = min(len(normalized), money_match.end() + radius)
    window = normalized[left:right]
    value_start = money_match.start() - left
    value_end = money_match.end() - left

    label_matches = list(
        re.finditer(
            r"\b(?:fee|fees|cost|costs|charge|charges|amount|amounts|price|prices|rate|rates)\b",
            window,
        )
    )

    context = window
    if label_matches:
        label_match = min(
            label_matches,
            key=lambda item: abs(item.end() - value_start),
        )
        ordered_labels = sorted(label_matches, key=lambda item: item.start())
        current_index = ordered_labels.index(label_match)
        previous_end = (
            ordered_labels[current_index - 1].end()
            if current_index > 0
            else max(0, label_match.start() - 45)
        )

        # Keep the field label's immediate prefix and the text attached to its
        # own value, but stop before a new logical conjunction/field begins.
        before = window[previous_end: label_match.start()]
        between = window[label_match.end(): value_start]
        next_label_start = (
            ordered_labels[current_index + 1].start()
            if current_index + 1 < len(ordered_labels)
            else len(window)
        )
        after = window[value_end: next_label_start]
        after = re.split(r"\s+(?:and|or)\s+|[;|]", after, maxsplit=1)[0]
        context = before + " " + between + " " + after

    context = _MONEY_RE.sub(" ", context)
    tokens = _scope_tokens(context)
    return tuple(sorted(
        token
        for token in tokens
        if token not in _FINANCIAL_GENERIC_TOKENS
        and not token.isdigit()
    ))


def _has_prefix_or_exact(tokens: Iterable[str], cues: Iterable[str]) -> bool:
    token_set = set(tokens)
    cue_set = set(cues)
    if token_set & cue_set:
        return True
    return any(
        len(token) >= 5 and len(cue) >= 5 and token[:5] == cue[:5]
        for token in token_set
        for cue in cue_set
    )


def _request_match(claim: Claim, evidence: EvidenceUnit) -> tuple[float, tuple[str, ...]]:
    """Match a plain request claim using structured request semantics first.

    Real documents often express a request indirectly: an eligibility query may
    be answered by text such as ``the applicant must have...`` without the word
    ``eligibility``. The matcher therefore supports small, generic language
    families while keeping the raw question out of the primary decision.
    """
    combined_text = " ".join(part for part in (evidence.heading or "", evidence.text) if part)
    evidence_tokens = _tokens(combined_text)
    heading_tokens = _tokens(evidence.heading or "")

    request_tokens = _tokens(claim.name or "")

    if request_tokens:
        request_overlap = _token_family_overlap(request_tokens, evidence_tokens)
        if request_overlap > 0.0:
            heading_overlap = _token_family_overlap(request_tokens, heading_tokens)
            score = 0.84 + (0.08 if heading_overlap > 0 else 0.0)
            return _clamp01(score), tuple(
                part for part in (
                    "structured request type aligns with evidence",
                    "request type appears in evidence heading" if heading_overlap > 0 else "",
                ) if part
            )

        request_name = normalize_text(claim.name or "")

        # Financial requests can be scoped by a target/anchor such as
        # "tuition", "hostel", "admission", or a program/category. A generic
        # "fee" mention alone must not satisfy a specifically scoped fee ask.
        if request_name in _FINANCIAL_REQUEST_NAMES:
            financial_scope = _financial_scope_tokens(claim)
            if financial_scope and not (financial_scope & evidence_tokens):
                return 0.30, ("generic financial wording matches but requested financial scope is absent",)

        family = _REQUEST_FAMILY_CUES.get(request_name, set())
        if family and _has_prefix_or_exact(evidence_tokens, family):
            return 0.74, ("generic request-language family aligns with evidence",)

        # Eligibility is often written only as applicant/requirement language
        # in formal policy text. Require both cues so a random ``must`` sentence
        # does not become eligibility evidence by itself.
        if request_name == "eligibility":
            has_requirement = _has_prefix_or_exact(evidence_tokens, _REQUIREMENT_CUES)
            has_applicant = bool(evidence_tokens & _APPLICANT_CUES)
            if has_requirement and has_applicant:
                return 0.78, ("eligibility is expressed through applicant/requirement language",)

    # Fallback for a request with no structured request type: use the raw query
    # as a partial lexical signal, rather than inventing a domain interpretation.
    query_tokens = _tokens(claim.text)
    lexical = _overlap(query_tokens, evidence_tokens)
    heading_bonus = 0.10 if evidence.heading and _overlap(query_tokens, heading_tokens) > 0 else 0.0
    score = _clamp01(0.78 * lexical + heading_bonus)
    if score >= 0.62:
        return score, ("request text has strong lexical alignment",)
    if score >= 0.35:
        return score, ("request text has partial lexical alignment",)
    return score, ("weak evidence alignment",)


def _generic_match(claim: Claim, evidence: EvidenceUnit) -> tuple[float, tuple[str, ...]]:
    claim_tokens = _tokens(claim.text) | {token for anchor in claim.anchors for token in _tokens(anchor)}
    evidence_tokens = evidence.token_set
    lexical = _overlap(claim_tokens, evidence_tokens)
    heading_bonus = 0.10 if evidence.heading and _overlap(claim_tokens, _tokens(evidence.heading)) > 0 else 0.0
    score = _clamp01(0.78 * lexical + heading_bonus)
    if score >= 0.62:
        return score, ("strong structured/lexical alignment",)
    if score >= 0.35:
        return score, ("partial structured/lexical alignment",)
    return score, ("weak evidence alignment",)


def match_claim_to_evidence(claim: Claim, evidence: EvidenceUnit) -> ClaimEvidenceMatch:
    if not isinstance(claim, Claim) or not isinstance(evidence, EvidenceUnit):
        raise TypeError("claim and evidence must use E5 contracts")

    if claim.kind == "numeric":
        matched, reasons = _numeric_match(claim, evidence)
        return ClaimEvidenceMatch(claim.claim_id, evidence.evidence_id, 0.92 if matched else 0.0,
                                  "supported" if matched else "unsupported", reasons)

    if claim.kind == "temporal":
        matched, reasons = _temporal_match(claim, evidence)
        return ClaimEvidenceMatch(claim.claim_id, evidence.evidence_id, 0.94 if matched else 0.0,
                                  "supported" if matched else "unsupported", reasons)

    if claim.kind == "relation":
        matched, reasons = _relation_match(claim, evidence)
        status = "supported" if matched else "partial" if "partially" in " ".join(reasons) else "unsupported"
        return ClaimEvidenceMatch(claim.claim_id, evidence.evidence_id, 0.86 if matched else 0.30 if status == "partial" else 0.0,
                                  status, reasons)

    if claim.kind == "request":
        score, reasons = _request_match(claim, evidence)
        status = "supported" if score >= 0.62 else "partial" if score >= 0.35 else "unsupported"
        return ClaimEvidenceMatch(claim.claim_id, evidence.evidence_id, score, status, reasons)

    score, reasons = _generic_match(claim, evidence)
    status = "supported" if score >= 0.62 else "partial" if score >= 0.35 else "unsupported"
    return ClaimEvidenceMatch(claim.claim_id, evidence.evidence_id, score, status, reasons)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def _claims_conflict(claim: Claim, evidence: Sequence[EvidenceUnit], supported_ids: set[str]) -> bool:
    """Detect incompatible factual values while respecting explicit value scope.

    Multiple monetary values are not automatically contradictory: they may be
    scoped to different programs, years, categories, or service contexts. A
    conflict exists only when different values are supported under the same
    local semantic scope by multiple sources.
    """
    if claim.kind == "request":
        request_name = normalize_text(claim.name or "")
        if request_name not in _FINANCIAL_REQUEST_NAMES:
            return False

        by_scope: dict[tuple[str, ...], dict[str, set[str]]] = {}
        for unit in evidence:
            if unit.evidence_id not in supported_ids:
                continue
            for value in _money_values_for_claim(claim, unit):
                scope = _financial_scope_signature_for_value(claim, unit, value)
                bucket = by_scope.setdefault(scope, {})
                bucket.setdefault(value, set()).add(unit.source.casefold())

        for values in by_scope.values():
            distinct = {value for value in values if value}
            if len(distinct) <= 1:
                continue
            source_union = set().union(*(sources for sources in values.values()))
            if len(source_union) >= 2:
                return True
        return False

    if claim.kind not in {"numeric", "temporal"}:
        return False

    signatures: set[tuple[str, ...]] = set()
    for unit in evidence:
        match = match_claim_to_evidence(claim, unit)
        if match.status != "supported":
            continue
        signature = _value_signature(claim, unit)
        if signature:
            signatures.add(signature)

    # Do not classify a multi-value table/paragraph as a contradiction.
    # Such a unit may legitimately enumerate several categories for the same
    # requested field. Conflict detection is limited to unambiguous single-
    # value evidence units.
    if len(signatures) <= 1 or any(len(signature) != 1 for signature in signatures):
        return False

    supporting_sources = {
        unit.source.casefold()
        for unit in evidence
        if match_claim_to_evidence(claim, unit).status == "supported"
    }
    return len(signatures) > 1 and len(supporting_sources) >= 2


def audit_claims(claims: Sequence[Claim], evidence: Sequence[EvidenceUnit]) -> ClaimAudit:
    """Audit every claim against all supplied evidence units."""
    claims = tuple(claims)
    evidence = tuple(evidence)
    matches: list[ClaimEvidenceMatch] = []
    supported: list[str] = []
    partial: list[str] = []
    unsupported: list[str] = []
    conflicting: list[str] = []

    for claim in claims:
        claim_matches = [match_claim_to_evidence(claim, unit) for unit in evidence]
        matches.extend(claim_matches)

        supported_matches = [m for m in claim_matches if m.status == "supported"]
        partial_matches = [m for m in claim_matches if m.status == "partial"]

        if _claims_conflict(claim, evidence, {m.evidence_id for m in supported_matches}):
            conflicting.append(claim.claim_id)
            continue

        if supported_matches:
            supported.append(claim.claim_id)
        elif partial_matches:
            partial.append(claim.claim_id)
        else:
            unsupported.append(claim.claim_id)

    return ClaimAudit(
        claims=claims,
        matches=tuple(matches),
        supported_claim_ids=_dedupe_preserve(supported),
        partial_claim_ids=_dedupe_preserve(partial),
        unsupported_claim_ids=_dedupe_preserve(unsupported),
        conflicting_claim_ids=_dedupe_preserve(conflicting),
    )


def audit_query_against_candidates(query: Query, candidates: Sequence[RetrievalCandidate]) -> ClaimAudit:
    """Extract claims and audit them against retrieved candidates."""
    return audit_claims(extract_claims(query), extract_evidence_units(candidates))


__all__ = [
    "Claim",
    "EvidenceUnit",
    "ClaimEvidenceMatch",
    "ClaimAudit",
    "factual_markers",
    "extract_factual_markers",
    "extract_claims",
    "split_evidence_units",
    "extract_evidence_units",
    "match_claim_to_evidence",
    "audit_claims",
    "audit_query_against_candidates",
]