"""
Generic candidate verification layer.

Retrieval is responsible for recall.

This module is responsible for determining whether a retrieved candidate
actually supports the user's information need.

Core principle:

    understand
        ->
    retrieve broadly
        ->
    fuse candidates
        ->
    verify evidence compatibility
        ->
    select evidence
        ->
    answer

Important design rule:

For exact requests, target and requested attribute must be connected
within the same local evidence unit.

For non-exact capability, process/navigation, and multi-facet requests,
controlled facet-level aggregation is allowed only inside one candidate
document and only when every substantive facet is grounded and the
corresponding evidence units are locally connected.

It is not enough for a document to contain:

    target somewhere
    attribute somewhere else

Example:

    M.Tech students follow the academic curriculum.

    Summer registration is available for B.Tech students.

This document contains both "M.Tech" and "registration", but it does
NOT support the information need:

    "How does summer M.Tech registration work?"

Likewise:

    Hostel accommodation is available.

    Guest house room charges are ₹1,200 per day.

must NOT satisfy:

    "What is the hostel fee?"

The implementation is institution-agnostic.
No IITJ-specific paths, entities, programs, or rules are encoded here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

try:
    from backend.core.query.models import SemanticQueryFrame
except ImportError:
    from backend.core.query_frame import SemanticQueryFrame
from backend.core.retrieval_contracts import RetrievalCandidate


VerificationStatus = Literal[
    "verified",
    "rejected",
    "uncertain",
]


# ---------------------------------------------------------------------------
# Generic request-type vocabulary.
#
# These are information-need signals, not institution-specific facts.
# ---------------------------------------------------------------------------

_REQUEST_TYPE_TERMS: dict[str, tuple[str, ...]] = {
    "cost": (
        "price",
        "prices",
        "cost",
        "costs",
        "fee",
        "fees",
        "amount",
        "charge",
        "charges",
        "rate",
        "rates",
        "tuition",
        "payment",
        "payments",
    ),
    "registration": (
        "registration",
        "register",
        "registered",
        "registration",
        "enrollment",
        "enrolment",
        "enroll",
        "enrol",
        "apply",
        "application",
    ),
    "admission": (
        "admission",
        "admissions",
        "application",
        "apply",
        "entrance",
        "selection",
    ),
    "definition": (
        "definition",
        "meaning",
        "means",
        "defined",
        "what is",
    ),
    "eligibility": (
        "eligibility",
        "eligible",
        "criteria",
        "requirements",
        "qualify",
        "qualification",
    ),
    "comparison": (
        "compare",
        "comparison",
        "difference",
        "versus",
        "vs",
    ),
    "directions": (
        "direction",
        "directions",
        "where",
        "location",
        "located",
        "route",
        "reach",
    ),
    "procedure": (
        "procedure",
        "process",
        "steps",
        "step",
        "how",
        "method",
    ),
    "information": (),
    "other": (),
}


# ---------------------------------------------------------------------------
# Verification result contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationDecision:
    candidate: RetrievalCandidate
    status: VerificationStatus
    accepted: bool

    target_grounded: bool
    attribute_grounded: bool
    scope_compatible: bool

    semantic_compatible: bool
    conflict_detected: bool

    coverage: float
    score: float

    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate.document_id,
            "status": self.status,
            "accepted": self.accepted,
            "target_grounded": self.target_grounded,
            "attribute_grounded": self.attribute_grounded,
            "scope_compatible": self.scope_compatible,
            "semantic_compatible": self.semantic_compatible,
            "conflict_detected": self.conflict_detected,
            "coverage": self.coverage,
            "score": self.score,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class VerificationBatch:
    verified: tuple[RetrievalCandidate, ...]
    uncertain: tuple[RetrievalCandidate, ...]
    rejected: tuple[RetrievalCandidate, ...]

    decisions: tuple[VerificationDecision, ...]

    def to_dict(self) -> dict:
        return {
            "verified_ids": [
                candidate.document_id
                for candidate in self.verified
            ],
            "uncertain_ids": [
                candidate.document_id
                for candidate in self.uncertain
            ],
            "rejected_ids": [
                candidate.document_id
                for candidate in self.rejected
            ],
            "decisions": [
                decision.to_dict()
                for decision in self.decisions
            ],
        }


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """
    Normalize text conservatively.

    This deliberately preserves alphanumeric identity while making
    punctuation/format differences less important.

    Examples:

        M.Tech   -> m tech
        M Tech   -> m tech
        hostel-fee -> hostel fee
    """

    value = str(text or "").lower()

    value = value.replace("_", " ")
    value = value.replace("-", " ")

    value = re.sub(
        r"[^a-z0-9\u0900-\u097f\s]",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def _tokens(text: str) -> list[str]:
    normalized = _normalize(text)

    if not normalized:
        return []

    return normalized.split()


def _meaningful_tokens(text: str) -> set[str]:
    """
    Generic content tokens.

    One-letter identifiers are handled separately by _target_tokens().
    """

    stop_words = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "for",
        "in",
        "on",
        "at",
        "and",
        "or",
        "can",
        "could",
        "would",
        "should",
        "what",
        "which",
        "how",
        "why",
        "when",
        "where",
        "who",
        "me",
        "my",
        "please",
        "tell",
        "give",
        "show",
        "get",
        "does",
        "do",
        "i",
        "we",
        "you",
        "it",
        "this",
        "that",
        "hai",
        "ka",
        "ki",
        "ke",
        "mein",
        "mujhe",
        "kya",
        "kitna",
        "kitne",
        "kaise",
        "wala",
        "wali",
        "wale",
    }

    return {
        token
        for token in _tokens(text)
        if len(token) >= 2
        and token not in stop_words
    }


def _target_tokens(text: str) -> set[str]:
    """
    Target-aware token extraction.

    This preserves identifiers such as:

        Item A
        Model X
        Section B
        M.Tech

    without treating the identifier as generic stop-word noise.
    """

    raw_tokens = str(text or "").split()

    normalized_tokens: set[str] = set()

    for raw_token in raw_tokens:
        cleaned = re.sub(
            r"[^A-Za-z0-9\u0900-\u097f]",
            "",
            raw_token,
        )

        if not cleaned:
            continue

        # Preserve single uppercase identifiers.
        if len(cleaned) == 1 and cleaned.isalpha():
            if cleaned.isupper():
                normalized_tokens.add(
                    cleaned.lower()
                )
            continue

        normalized = _normalize(cleaned)

        if normalized:
            normalized_tokens.add(normalized)

    return normalized_tokens


def _contains_phrase(
    document_text: str,
    phrase: str,
) -> bool:
    normalized_document = _normalize(
        document_text
    )

    normalized_phrase = _normalize(
        phrase
    )

    if not normalized_document:
        return False

    if not normalized_phrase:
        return False

    return normalized_phrase in normalized_document


def _token_grounding_ratio(
    document_text: str,
    query_text: str,
) -> float:
    query_tokens = _meaningful_tokens(
        query_text
    )

    if not query_tokens:
        return 0.0

    document_tokens = set(
        _tokens(document_text)
    )

    matched = sum(
        1
        for token in query_tokens
        if token in document_tokens
    )

    return matched / len(query_tokens)


# ---------------------------------------------------------------------------
# Candidate document extraction
# ---------------------------------------------------------------------------


def _document_text(candidate: RetrievalCandidate) -> str:
    """Return only factual page content; provenance metadata is not evidence."""
    document = candidate.document
    return str(getattr(document, "page_content", "") or "").strip()



# ---------------------------------------------------------------------------
# Local evidence units
# ---------------------------------------------------------------------------


def _split_evidence_units(text: str) -> tuple[str, ...]:
    """Split factual content into conservative sentence/line evidence units."""
    raw = str(text or "").strip()
    if not raw:
        return ()
    raw = re.sub(r"\r\n?", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    units=[]; sentence_pattern=re.compile(r"(?<=[.!?।])\s+")
    for paragraph in paragraphs:
        for line in [x.strip() for x in paragraph.split("\n") if x.strip()]:
            units.extend(s.strip() for s in sentence_pattern.split(line) if s.strip())
    return tuple(units)



# ---------------------------------------------------------------------------
# Target / attribute grounding
# ---------------------------------------------------------------------------


def _singular_variants(token: str) -> set[str]:
    """Return conservative singular/plural-normalized forms."""
    value = _normalize(token)
    if not value:
        return set()
    out = {value}
    if value.endswith("ies") and len(value) > 4:
        out.add(value[:-3] + "y")
    if value.endswith(("ches", "shes", "xes", "zes", "ses")) and len(value) > 5:
        out.add(value[:-2])
    if value.endswith("es") and len(value) > 4:
        out.add(value[:-2])
    if value.endswith("s") and len(value) > 3:
        out.add(value[:-1])
    if not value.endswith("s") and len(value) > 2:
        out.add(value + "s")
    return out

def _compact_alphanumeric(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _normalize(text))

def _ground_target_in_text(target: str, text: str) -> bool:
    """Ground a target with phrase, inflection-tolerant tokens, or compact identity."""
    target=str(target or "").strip()
    if not target: return False
    if _contains_phrase(text,target): return True
    tt=_target_tokens(target)
    if not tt: return False
    dt=set(_tokens(text)); matched=0
    for t in tt:
        if _singular_variants(t)&dt: matched+=1; continue
        c=_compact_alphanumeric(t)
        if c and any(c==_compact_alphanumeric(x) for x in dt): matched+=1
    return matched/len(tt)>=0.80



def _facet_values(
    frame: SemanticQueryFrame,
) -> tuple[str, ...]:
    facets = getattr(
        frame,
        "facets",
        (),
    ) or ()

    values: list[str] = []

    for facet in facets:
        value = getattr(
            facet,
            "value",
            "",
        )

        if value:
            values.append(
                str(value)
            )

    return tuple(values)


def _attribute_terms_for_frame(
    frame: SemanticQueryFrame,
) -> tuple[str, ...]:
    """
    Return generic attribute phrases relevant to the request.

    Query facets are preferred.

    Request-type vocabulary is used as a fallback.

    This does NOT encode institution-specific attributes.
    """

    values: list[str] = []

    for facet_value in _facet_values(frame):
        normalized = _normalize(
            facet_value
        )

        if normalized:
            values.append(normalized)

    request_type = str(
        getattr(
            frame,
            "request_type",
            "",
        )
        or ""
    ).lower()

    values.extend(
        _REQUEST_TYPE_TERMS.get(
            request_type,
            (),
        )
    )

    # Preserve order while deduplicating.
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        normalized = _normalize(value)

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(normalized)

    return tuple(result)


def _ground_attribute_in_text(
    frame: SemanticQueryFrame,
    text: str,
) -> bool:
    """
    Determine whether the requested attribute appears in the given
    local evidence unit.
    """

    normalized_text = _normalize(
        text
    )

    if not normalized_text:
        return False

    attribute_terms = (
        _attribute_terms_for_frame(frame)
    )

    for term in attribute_terms:
        if term in normalized_text:
            return True

    return False


def _semantic_target_support(frame: SemanticQueryFrame, candidate: RetrievalCandidate) -> bool:
    """Semantic target support is diagnostic-only; it never establishes factual grounding."""
    return False


def _ground_target(frame: SemanticQueryFrame, candidate: RetrievalCandidate) -> bool:
    """Ground the requested target using factual content or exact candidate meaning."""
    document_text = _document_text(candidate)
    target = str(getattr(frame, "target", "") or "").strip()
    if not target:
        return False
    if _ground_target_in_text(target, document_text):
        return True
    meaning = candidate.meaning
    meaning_values = []
    for field_name in ("programs", "entities", "topics", "attributes"):
        meaning_values.extend(str(value) for value in (getattr(meaning, field_name, ()) or ()))
    target_compact = _compact_alphanumeric(target)
    if target_compact:
        for value in meaning_values:
            if _compact_alphanumeric(value) == target_compact:
                return True
    target_tokens = _meaningful_tokens(target)
    for value in meaning_values:
        value_tokens = _meaningful_tokens(value)
        if not target_tokens or not value_tokens:
            continue
        matched = 0
        for token in target_tokens:
            compact_token = _compact_alphanumeric(token)
            if not compact_token:
                continue
            if any(compact_token == _compact_alphanumeric(value_token) or (len(compact_token) >= 4 and compact_token in _compact_alphanumeric(value_token)) for value_token in value_tokens):
                matched += 1
        if matched / max(len(target_tokens), 1) >= 0.80:
            return True
    return False




def _ground_attribute(
    frame: SemanticQueryFrame,
    candidate: RetrievalCandidate,
) -> bool:
    """
    Determine whether the requested attribute exists anywhere
    in the candidate.

    Exact verification additionally requires local target+attribute
    relation through _ground_target_attribute_relation().
    """

    document_text = _document_text(
        candidate
    )

    if _ground_attribute_in_text(
        frame,
        document_text,
    ):
        return True

    alignment = candidate.alignment

    if alignment is not None:
        attribute_match = float(
            getattr(
                alignment,
                "attribute_match",
                0.0,
            )
            or 0.0
        )

        if attribute_match >= 0.90:
            return True

    return False


# ---------------------------------------------------------------------------
# Local target-attribute relation
# ---------------------------------------------------------------------------


_RELATION_CONTINUITY_MARKERS={"it","its","they","their","these","those","this","that","also","additionally","further","furthermore","here","there","such"}

def _has_relation_continuity(first_unit: str, second_unit: str, target: str) -> bool:
    """Allow adjacent-unit joins only with explicit target continuity."""
    second_tokens=set(_tokens(second_unit))
    if second_tokens & _RELATION_CONTINUITY_MARKERS:
        return True
    if _ground_target_in_text(target,second_unit):
        return True
    target_tokens=_target_tokens(target)
    if target_tokens:
        matched=sum(bool(_singular_variants(t)&second_tokens) for t in target_tokens)
        if matched/len(target_tokens)>=0.80:
            return True
    return bool(second_tokens & {"because","therefore","thus","while","whereas"})



def _relation_evidence_units(frame: SemanticQueryFrame, candidate: RetrievalCandidate) -> tuple[str,...]:
    units=list(_split_evidence_units(_document_text(candidate)))
    if len(units)<2: return tuple(units)
    target=str(getattr(frame,"target","") or "").strip()
    if not target: return tuple(units)
    out=list(units)
    for i in range(len(units)-1):
        a,b=units[i],units[i+1]
        if not (_ground_target_in_text(target,a) or _ground_target_in_text(target,b)): continue
        if _has_relation_continuity(a,b,target): out.append(f"{a} {b}")
    return tuple(out)

def _facet_values_for_frame(frame: SemanticQueryFrame) -> tuple[str,...]:
    out=[]; seen=set()
    for facet in (getattr(frame,"facets",()) or ()):
        v=str(getattr(facet,"value","") or ""); n=_normalize(v)
        if n and n not in seen: seen.add(n); out.append(n)
    return tuple(out)

def _ground_facet_in_candidate(facet: str, candidate: RetrievalCandidate) -> bool:
    text=_document_text(candidate); f=_normalize(facet)
    if not f: return False
    if _contains_phrase(text,f): return True
    ft=_meaningful_tokens(f); dt=set(_tokens(text))
    if not ft: return False
    return sum(bool(_singular_variants(t)&dt) for t in ft)/len(ft)>=0.80

def _ground_target_attribute_relation(frame: SemanticQueryFrame, candidate: RetrievalCandidate) -> bool:
    """Verify target/attribute relation without joining unrelated facts."""
    target=str(getattr(frame,"target","") or "").strip()
    if not target: return False
    for unit in _relation_evidence_units(frame,candidate):
        if _ground_target_in_text(target,unit) and _ground_attribute_in_text(frame,unit): return True
    facets=_facet_values_for_frame(frame)
    if len(facets)>1:
        if all(_ground_facet_in_candidate(f,candidate) for f in facets) and (_ground_target(frame,candidate) or _semantic_target_support(frame,candidate)): return True
    return _semantic_target_support(frame,candidate) and _ground_attribute_in_text(frame,_document_text(candidate))


# ---------------------------------------------------------------------------
# Change 21: controlled facet-level evidence aggregation
# ---------------------------------------------------------------------------

# These are generic language signals for information-need shape. They do not
# encode institution-specific entities or facts.
_CAPABILITY_PATTERNS = (
    "does ",
    "do ",
    "can ",
    "could ",
    "is there",
    "are there",
    "is available",
    "are available",
    "exists",
    "exist",
    "offers ",
    "offer ",
    "supports ",
    "support ",
    "allows ",
    "allow ",
    "whether ",
    "milta hai kya",
    "milti hai kya",
    "milte hain kya",
    "hai kya",
    "hain kya",
    "hota hai kya",
    "hoti hai kya",
)

_PROCESS_NAV_REQUEST_TYPES = {
    "procedure",
    "directions",
    "registration",
}

_PROCESS_NAV_START_PATTERNS = (
    "how do i",
    "how can i",
    "how to",
    "how does",
    "how is",
    "how are",
    "how should",
    "where do i",
    "where can i",
    "where is",
    "where are",
    "who approves",
    "who helps",
    "who handles",
    "who manages",
    "who is",
    "whom",
)

_PROCESS_NAV_CONTAINS_MARKERS = (
    " reach ",
    " route ",
    " directions ",
    " process ",
    " steps ",
    " comes after ",
    " come after ",
    " happens after ",
    " happens next ",
    " next after ",
    " following ",
)

_PLACEHOLDER_TARGETS = {
    "true",
    "false",
    "yes",
    "no",
    "unknown",
    "none",
    "null",
}

_GENERIC_BRIDGE_WORDS = {
    "and",
    "also",
    "with",
    "during",
    "after",
    "before",
    "from",
    "to",
    "for",
    "by",
    "via",
    "through",
    "under",
    "using",
    "which",
    "where",
    "while",
    "because",
    "therefore",
    "thus",
    "then",
    "first",
    "next",
    "further",
    "additionally",
    "here",
    "there",
    "such",
}

_GENERIC_CONTEXT_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "can",
    "could",
    "would",
    "should",
    "may",
    "might",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "will",
    "shall",
    "not",
    "yes",
    "no",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "what",
    "which",
    "how",
    "why",
    "when",
    "where",
    "who",
    "whom",
    "whose",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "their",
    "them",
    "there",
    "here",
    "me",
    "my",
    "you",
    "your",
    "we",
    "our",
    "i",
}


def _original_query(frame: SemanticQueryFrame) -> str:
    for field_name in (
        "original_query",
        "query",
        "user_query",
        "raw_query",
    ):
        value = getattr(frame, field_name, "")
        if value:
            return str(value)
    return ""


def _is_placeholder_target(frame: SemanticQueryFrame) -> bool:
    target = _normalize(
        str(getattr(frame, "target", "") or "")
    )
    return target in _PLACEHOLDER_TARGETS


def _substantive_facets(
    frame: SemanticQueryFrame,
) -> tuple[str, ...]:
    """Return query facets that can carry factual evidence."""
    values = _facet_values_for_frame(frame)
    out: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized = _normalize(value)
        if not normalized:
            continue
        if normalized in _PLACEHOLDER_TARGETS:
            continue
        meaningful = _meaningful_tokens(normalized)
        if not meaningful:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)

    # When the frame has no explicit facets, a real target is a safe single
    # facet fallback. Placeholder targets such as True/False are not.
    if not out and not _is_placeholder_target(frame):
        target = _normalize(
            str(getattr(frame, "target", "") or "")
        )
        if target and _meaningful_tokens(target):
            out.append(target)

    return tuple(out)


def _ground_facet_in_text(
    facet: str,
    text: str,
) -> bool:
    normalized_facet = _normalize(facet)
    normalized_text = _normalize(text)

    if not normalized_facet or not normalized_text:
        return False

    if normalized_facet in normalized_text:
        return True

    facet_tokens = _meaningful_tokens(normalized_facet)
    if not facet_tokens:
        return False

    document_tokens = set(_tokens(normalized_text))
    matched = 0

    for token in facet_tokens:
        if _singular_variants(token) & document_tokens:
            matched += 1

    return matched / len(facet_tokens) >= 0.80


def _facet_evidence_indexes(
    facet: str,
    units: Sequence[str],
) -> tuple[int, ...]:
    return tuple(
        index
        for index, unit in enumerate(units)
        if _ground_facet_in_text(facet, unit)
    )


_STRONG_BRIDGE_WORDS = {
    "also",
    "additionally",
    "further",
    "furthermore",
    "therefore",
    "thus",
    "then",
    "next",
    "while",
    "whereas",
    "because",
    "here",
    "there",
    "such",
}


def _unit_has_bridge_marker(unit: str) -> bool:
    tokens = set(_tokens(unit))
    return bool(tokens & _STRONG_BRIDGE_WORDS)


def _unit_context_tokens(
    unit: str,
    facets: Sequence[str],
    target: str = "",
) -> set[str]:
    """Extract conservative non-facet context words for aggregation links."""
    tokens = _meaningful_tokens(unit)

    excluded = set(_GENERIC_CONTEXT_WORDS)
    excluded.update(_GENERIC_BRIDGE_WORDS)
    excluded.update(_REQUEST_TYPE_TERMS.get("information", ()))

    for facet in facets:
        excluded.update(_meaningful_tokens(facet))

    if target:
        excluded.update(_meaningful_tokens(target))

    return {
        token
        for token in tokens
        if token not in excluded
        and len(token) >= 3
        and not token.isdigit()
    }


def _facet_units_connected(
    first_unit: str,
    second_unit: str,
    anchor: str,
    facets: Sequence[str],
    target: str,
) -> bool:
    """
    Check whether two adjacent evidence units can safely belong to the same
    information need. Mere adjacency or repeated facet presence is not enough
    when a concrete target exists.
    """
    real_target = bool(target and not _normalize(target) in _PLACEHOLDER_TARGETS)

    if real_target:
        # A concrete target must continue into the next unit, or the prose must
        # explicitly refer back to it. This blocks M.Tech evidence in one
        # sentence from being joined to B.Tech registration in the next.
        if _ground_target_in_text(target, second_unit):
            return True

        second_tokens = set(_tokens(second_unit))
        if second_tokens & _RELATION_CONTINUITY_MARKERS:
            return True

        if _unit_has_bridge_marker(first_unit) or _unit_has_bridge_marker(second_unit):
            first_context = _unit_context_tokens(
                first_unit,
                facets,
                target,
            )
            second_context = _unit_context_tokens(
                second_unit,
                facets,
                target,
            )
            if first_context & second_context:
                return True

        return False

    # With no concrete target, an explicit textual bridge or shared local
    # context is required. This permits legitimate statements such as
    # "It also ..." without turning every adjacent sentence into evidence.
    if _unit_has_bridge_marker(first_unit) or _unit_has_bridge_marker(second_unit):
        first_context = _unit_context_tokens(
            first_unit,
            facets,
            target,
        )
        second_context = _unit_context_tokens(
            second_unit,
            facets,
            target,
        )
        if first_context & second_context:
            return True

    return bool(
        second_unit
        and _ground_facet_in_text(anchor, first_unit)
        and _ground_facet_in_text(anchor, second_unit)
    )


def _is_capability_question(query: str) -> bool:
    """Return True only for explicit yes/no or availability-style requests."""
    normalized = _normalize(query)
    if not normalized:
        return False

    # "what ... available" is informational/list wording, not a yes/no
    # capability query. Do not classify it from the presence of "available"
    # alone.
    if normalized.startswith((
        "what ",
        "which ",
        "when ",
        "where ",
        "why ",
        "who ",
        "how many ",
        "how much ",
    )):
        return False

    return normalized.startswith(_CAPABILITY_PATTERNS) or normalized.endswith((
        "milta hai kya",
        "milti hai kya",
        "milte hain kya",
        "hai kya",
        "hain kya",
        "hota hai kya",
        "hoti hai kya",
    ))


def _is_process_navigation_question(
    frame: SemanticQueryFrame,
) -> bool:
    """Return True for procedural, navigation, or workflow information needs."""
    query = _normalize(_original_query(frame))
    request_type = str(getattr(frame, "request_type", "") or "").lower()

    if query.startswith(("how many ", "how much ")):
        return False

    if request_type in _PROCESS_NAV_REQUEST_TYPES:
        return True

    if query.startswith(_PROCESS_NAV_START_PATTERNS):
        return True

    padded = f" {query} "
    return any(marker in padded for marker in _PROCESS_NAV_CONTAINS_MARKERS)


def _query_requires_strict_exact_grounding(
    frame: SemanticQueryFrame,
) -> bool:
    """
    Decide whether an ``exact`` requirement must keep strict relation
    grounding.

    The query frame may conservatively classify several information-needs as
    ``exact`` because they require target/attribute alignment.  That flag is
    not, by itself, enough to forbid controlled aggregation.  The actual user
    question can still be a capability, process/navigation, or multi-facet
    request.

    Strict exact grounding wins for explicit factual-relation/lookup
    language such as prices, fees, amounts, rates, comparisons, and direct
    inclusion/separation/waiver relations.
    """
    mode = _requirement_mode(frame)
    if mode != "exact":
        return False

    query = _normalize(_original_query(frame))
    request_type = str(
        getattr(frame, "request_type", "") or ""
    ).lower()

    facets = _substantive_facets(frame)
    multi_facet = len(facets) >= 2

    capability = _is_capability_question(query)
    process_navigation = _is_process_navigation_question(frame)

    # Direct factual relation/lookup questions stay strict even when they
    # contain multiple substantive facets.  This protects cases such as:
    #   "Is hostel fee included in semester fee?"
    #   "What is the admission fee?"
    #   "What is the scholarship amount?"
    strict_relation_markers = (
        "included in",
        "separate",
        "exempt from",
        "exemption",
        "waiver",
        "how much",
        "what is the fee",
        "what is the amount",
        "what is the cost",
        "what is the price",
        "what is the rate",
        "what are the fees",
        "how many",
        "compare",
        "comparison",
        "difference between",
        " versus ",
        " vs ",
    )

    strict_lookup_type = request_type in {
        "cost",
        "comparison",
    }

    explicit_strict_relation = any(
        marker in query
        for marker in strict_relation_markers
    )

    if strict_lookup_type and not (capability or process_navigation):
        return True

    if explicit_strict_relation and not process_navigation:
        return True

    # Capability/process questions are allowed to aggregate even if the frame
    # conservatively arrived with mode="exact".
    if capability or process_navigation:
        return False

    # Multi-facet descriptive questions are the third controlled family. They
    # are non-strict unless the explicit factual-relation rules above apply.
    if multi_facet:
        return False

    return True


def _controlled_aggregation_shape(
    frame: SemanticQueryFrame,
) -> tuple[bool, bool, bool]:
    """Return capability, process/navigation, and multi-facet shape flags."""
    query = _normalize(_original_query(frame))
    request_type = str(
        getattr(frame, "request_type", "") or ""
    ).lower()
    facets = _substantive_facets(frame)

    capability = _is_capability_question(query)
    process_navigation = _is_process_navigation_question(frame)

    multi_facet = len(facets) >= 2
    return capability, process_navigation, multi_facet


def _process_signal_terms(frame: SemanticQueryFrame) -> tuple[str, ...]:
    """Return generic operational terms that the requested process needs."""
    query = _normalize(_original_query(frame))
    padded = f" {query} "
    groups = (
        ((" register ", " registration ", " enroll ", " enrollment ", " enrol ", " enrolment "), (
            "registration", "register", "enrollment", "enroll", "enrol",
        )),
        ((" approve ", " approves ", " approved ", " approval "), (
            "approve", "approves", "approval", "approved",
        )),
        ((" prepare ", " preparation "), (
            "prepare", "preparation",
        )),
        ((" booking ", " book "), (
            "booking", "book",
        )),
        ((" helps ", " help ", " advisor ", " guide "), (
            "help", "helps", "advisor", "guide",
        )),
        ((" manages ", " manages? ", " handles ", " handle "), (
            "manage", "manages", "handle", "handles",
        )),
        ((" reach ", " route ", " directions ", " transport "), (
            "reach", "route", "directions", "transport",
        )),
        ((" location ", " located ", " address ", " where is "), (
            "location", "located", "address",
        )),
        ((" change ", " changed ", " changing "), (
            "change", "changed", "changing",
        )),
        ((" after ", " before ", " next ", " comes after ", " happens after "), (
            "after", "before", "next", "comes", "happens",
        )),
        ((" counselling ", " counseling ", " seat allocation ", " selection "), (
            "counselling", "counseling", "allocation", "selection",
        )),
    )

    out: list[str] = []
    for triggers, terms in groups:
        if any(trigger in padded for trigger in triggers):
            out.extend(terms)

    return tuple(dict.fromkeys(out))


def _ground_any_term_in_text(terms: Sequence[str], text: str) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    return any(_contains_phrase(normalized, term) or term in _tokens(normalized) for term in terms)


def _connected_component_cover(
    units: Sequence[str],
    required_phrases: Sequence[str],
    target: str = "",
) -> bool:
    """Require all requested evidence facets in one locally connected component."""
    if not units or not required_phrases:
        return False

    indexes = {
        phrase: tuple(
            i for i, unit in enumerate(units)
            if _ground_facet_in_text(phrase, unit)
        )
        for phrase in required_phrases
    }

    if any(not locations for locations in indexes.values()):
        return False

    target_indexes = tuple(
        i for i, unit in enumerate(units)
        if target and _ground_target_in_text(target, unit)
    )

    adjacency = {i: set() for i in range(len(units))}
    for left in range(len(units) - 1):
        right = left + 1
        connected = False

        if target and (left in target_indexes or right in target_indexes):
            if _has_relation_continuity(units[left], units[right], target):
                connected = True

        if not connected:
            for phrase in required_phrases:
                if not (left in indexes[phrase] or right in indexes[phrase]):
                    continue
                if _facet_units_connected(
                    units[left],
                    units[right],
                    phrase,
                    required_phrases,
                    target,
                ):
                    connected = True
                    break

        if connected:
            adjacency[left].add(right)
            adjacency[right].add(left)

    seen: set[int] = set()
    for start in range(len(units)):
        if start in seen:
            continue
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            seen.add(node)
            stack.extend(adjacency[node] - component)

        if all(any(index in component for index in locations) for locations in indexes.values()):
            if target and not any(index in component for index in target_indexes):
                continue
            return True

    return False


def _exact_placeholder_facet_support(
    frame: SemanticQueryFrame,
    candidate: RetrievalCandidate,
) -> bool:
    """Support exact count/lookup questions whose target is intentionally abstract."""
    facets = _substantive_facets(frame)
    if not facets:
        return False
    units = _split_evidence_units(_document_text(candidate))
    return _connected_component_cover(units, facets, target="")


def _controlled_facet_aggregation(
    frame: SemanticQueryFrame,
    candidate: RetrievalCandidate,
) -> tuple[bool, float, str]:
    """
    Verify capability/process/multi-facet evidence without cross-context mixing.

    Rules:
      * never aggregate across candidates;
      * every substantive facet must be factually grounded;
      * a real target must be grounded in the same connected evidence component;
      * process/navigation requests must contain their operational evidence;
      * exact factual lookups are excluded and keep strict relation grounding.
    """
    if _query_requires_strict_exact_grounding(frame):
        return False, 0.0, "none"

    facets = _substantive_facets(frame)
    if not facets:
        return False, 0.0, "none"

    capability, process_navigation, multi_facet = _controlled_aggregation_shape(frame)
    if not (capability or process_navigation or multi_facet):
        return False, 0.0, "none"

    units = _split_evidence_units(_document_text(candidate))
    if not units:
        return False, 0.0, "none"

    facet_indexes = {facet: _facet_evidence_indexes(facet, units) for facet in facets}
    grounded_count = sum(bool(indexes) for indexes in facet_indexes.values())
    facet_coverage = grounded_count / len(facets)
    if grounded_count != len(facets):
        return False, facet_coverage, "capability" if capability else "process_navigation" if process_navigation else "multi_facet"

    target = str(getattr(frame, "target", "") or "").strip()
    target_is_real = bool(target and not _is_placeholder_target(frame))

    # Process/navigation requires the evidence for the actual operation, not
    # merely the subject noun. This blocks, for example, a curriculum page from
    # satisfying a registration question just because both mention "summer".
    if process_navigation:
        signal_terms = _process_signal_terms(frame)
        if signal_terms and not any(_ground_any_term_in_text(signal_terms, unit) for unit in units):
            return False, facet_coverage, "process_navigation"

    # One real target must be connected to the requested facets. Presence in a
    # different part of the document is not enough.
    if target_is_real:
        if not _ground_target(frame, candidate):
            return False, facet_coverage, "capability" if capability else "process_navigation" if process_navigation else "multi_facet"

        if not _connected_component_cover(units, facets, target=target):
            return False, facet_coverage, "capability" if capability else "process_navigation" if process_navigation else "multi_facet"

    else:
        # With an abstract target (True/False/unknown), facets themselves are
        # the information need. They still have to belong to one local evidence
        # component; otherwise unrelated facts are being stitched together.
        if not _connected_component_cover(units, facets, target=""):
            return False, facet_coverage, "capability" if capability else "process_navigation" if process_navigation else "multi_facet"

    kind = (
        "capability"
        if capability
        else "process_navigation"
        if process_navigation
        else "multi_facet"
    )
    return True, 1.0, kind


# ---------------------------------------------------------------------------
# Scope / semantic compatibility
# ---------------------------------------------------------------------------


def _scope_is_compatible(
    candidate: RetrievalCandidate,
) -> bool:
    alignment = candidate.alignment

    if alignment is None:
        return True

    conflicts = getattr(
        alignment,
        "conflicts",
        (),
    ) or ()

    if conflicts:
        return False

    scope_match = float(
        getattr(
            alignment,
            "scope_match",
            0.0,
        )
        or 0.0
    )

    # Zero means neutral / not established.
    if scope_match <= 0.0:
        return True

    return scope_match >= 0.50


def _semantic_compatibility(
    candidate: RetrievalCandidate,
) -> tuple[bool, float]:
    alignment = candidate.alignment

    if alignment is None:
        return False, 0.0

    conflicts = getattr(
        alignment,
        "conflicts",
        (),
    ) or ()

    if conflicts:
        return False, 0.0

    semantic_match = float(
        getattr(
            alignment,
            "semantic_match",
            0.0,
        )
        or 0.0
    )

    coverage = float(
        getattr(
            alignment,
            "coverage",
            0.0,
        )
        or 0.0
    )

    score = max(
        0.0,
        min(
            1.0,
            max(
                semantic_match,
                coverage,
            ),
        ),
    )

    compatible = (
        semantic_match >= 0.50
        or coverage >= 0.50
    )

    return compatible, score


# ---------------------------------------------------------------------------
# Requirement helpers
# ---------------------------------------------------------------------------


def _requirement_mode(
    frame: SemanticQueryFrame,
) -> str:
    requirement = getattr(
        frame,
        "requirement",
        None,
    )

    if requirement is None:
        return "standard"

    return str(
        getattr(
            requirement,
            "mode",
            "standard",
        )
        or "standard"
    ).lower()


def _requirement_flag(
    frame: SemanticQueryFrame,
    field_name: str,
    default: bool,
) -> bool:
    requirement = getattr(
        frame,
        "requirement",
        None,
    )

    if requirement is None:
        return default

    return bool(
        getattr(
            requirement,
            field_name,
            default,
        )
    )


def _explicit_conflict(
    candidate: RetrievalCandidate,
) -> bool:
    alignment = candidate.alignment

    if alignment is None:
        return False

    conflicts = getattr(
        alignment,
        "conflicts",
        (),
    ) or ()

    return bool(conflicts)



# ---------------------------------------------------------------------------
# Deterministic lexical recall rescue
# ---------------------------------------------------------------------------

# These groups are intentionally generic normalization aids. They are not
# institution-specific facts. They preserve the useful behavior of a classic
# keyword retriever when a stricter semantic frame is incomplete or overly
# narrow.
_LEXICAL_EQUIVALENCE_GROUPS = (
    frozenset({"program", "programs", "programme", "programmes"}),
    frozenset({"application", "applications", "apply", "applying", "applied", "applies"}),
    frozenset({"admission", "admissions"}),
    frozenset({"requirement", "requirements", "criteria", "criterion"}),
    frozenset({"location", "located", "address"}),
    frozenset({"direction", "directions", "route", "routes", "reach", "reached", "reaches", "reaching"}),
    frozenset({"fee", "fees", "tuition", "cost", "costs", "charge", "charges", "amount"}),
)

_LEXICAL_QUERY_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of",
    "for", "in", "on", "at", "and", "or", "can", "could", "would", "should",
    "what", "which", "how", "why", "when", "where", "who", "whom", "whose",
    "do", "does", "did", "i", "me", "my", "you", "your", "we", "our", "it",
    "this", "that", "these", "those", "tell", "give", "show", "please", "about",
    "available", "following", "stated", "there", "here", "for", "students",
}


def _expanded_token_forms(token: str) -> set[str]:
    normalized = _normalize(token)
    if not normalized:
        return set()
    forms = _singular_variants(normalized)
    forms.add(normalized)
    for group in _LEXICAL_EQUIVALENCE_GROUPS:
        if normalized in group:
            forms.update(group)
            break
    # Common joined degree/program spellings.
    compact = _compact_alphanumeric(normalized)
    if compact:
        forms.add(compact)
    return {item for item in forms if item}


def _lexical_token_hit(token: str, document_tokens: set[str]) -> bool:
    forms = _expanded_token_forms(token)
    if not forms:
        return False
    normalized_document = set(document_tokens)
    compact_document = {_compact_alphanumeric(item) for item in normalized_document}
    return any(
        form in normalized_document or form in compact_document
        for form in forms
    )


def _lexical_token_overlap(query_text: str, document_text: str) -> float:
    query_tokens = [
        token
        for token in _tokens(query_text)
        if token not in _LEXICAL_QUERY_STOPWORDS and len(token) >= 2
    ]
    if not query_tokens:
        return 0.0
    document_tokens = set(_tokens(document_text))
    matched = sum(_lexical_token_hit(token, document_tokens) for token in query_tokens)
    return matched / len(query_tokens)


def _lexical_phrase_or_token_support(phrase: str, document_text: str) -> float:
    phrase = _normalize(phrase)
    if not phrase:
        return 0.0
    if _contains_phrase(document_text, phrase):
        return 1.0
    phrase_tokens = _meaningful_tokens(phrase)
    if not phrase_tokens:
        return 0.0
    document_tokens = set(_tokens(document_text))
    matched = sum(_lexical_token_hit(token, document_tokens) for token in phrase_tokens)
    return matched / len(phrase_tokens)


def _lexical_recall_signal(
    frame: SemanticQueryFrame,
    candidate: RetrievalCandidate,
) -> tuple[float, float, float, float, float]:
    """Return target, facet, query, operational and overall lexical support."""
    text = _document_text(candidate)
    query = _original_query(frame)
    target = str(getattr(frame, "target", "") or "").strip()
    facets = _substantive_facets(frame)

    target_score = _lexical_phrase_or_token_support(target, text) if target else 0.0
    if facets:
        facet_scores = [_lexical_phrase_or_token_support(facet, text) for facet in facets]
        facet_score = sum(facet_scores) / len(facet_scores)
    else:
        facet_score = 0.0

    query_score = _lexical_token_overlap(query, text)
    signal_terms = _process_signal_terms(frame)
    operational_score = (
        1.0 if _ground_any_term_in_text(signal_terms, text) else 0.0
    ) if signal_terms else 0.0

    # Target/facet evidence is more valuable than generic query overlap.
    if target_score and facet_score:
        overall = 0.45 * target_score + 0.35 * facet_score + 0.20 * query_score
    elif target_score:
        overall = 0.60 * target_score + 0.40 * query_score
    elif facet_score:
        overall = 0.55 * facet_score + 0.45 * query_score
    else:
        overall = query_score

    if operational_score:
        overall = min(1.0, overall + 0.15)

    return target_score, facet_score, query_score, operational_score, min(1.0, overall)


_RELATION_SENSITIVE_MARKERS = (
    "included in",
    "included within",
    "separate",
    "exempt from",
    "exempted from",
    "exemption",
    "waiver",
    "whether",
    "comes after",
    "happens after",
    "after ",
    "before ",
    "difference between",
    "versus",
    " vs ",
    "compare",
    "comparison",
)


def _relation_sensitive_question(frame: SemanticQueryFrame) -> bool:
    """Return True when the user asks for a concrete factual relationship."""
    query = f" {_normalize(_original_query(frame))} "
    request_type = str(getattr(frame, "request_type", "") or "").casefold()
    return request_type == "comparison" or any(
        marker in query for marker in _RELATION_SENSITIVE_MARKERS
    )


def _lexical_rescue_allowed(
    frame: SemanticQueryFrame,
    candidate: RetrievalCandidate,
    *,
    exact_request: bool,
) -> tuple[bool, float, tuple[str, ...]]:
    """Bounded keyword fallback for candidates whose semantic metadata is weak."""
    if _explicit_conflict(candidate):
        return False, 0.0, ("explicit_conflict",)
    if not candidate.quality.usable or not _scope_is_compatible(candidate):
        return False, 0.0, ("scope_or_quality_not_usable",)

    if not _all_required_entities_grounded(frame,candidate):
        return False, 0.0, ("protected_entity_not_grounded",)

    target_score, facet_score, query_score, operational_score, overall = _lexical_recall_signal(
        frame,
        candidate,
    )
    placeholder = _is_placeholder_target(frame)
    relation_sensitive = _relation_sensitive_question(frame)

    # Relationships stay strict. We may ignore a bad semantic label, but we
    # never infer a relationship merely because two words occur somewhere.
    if relation_sensitive:
        if placeholder:
            if _exact_placeholder_facet_support(frame, candidate):
                return True, max(overall, 0.85), (
                    "lexical_recall_rescue",
                    "relation_locally_grounded",
                )

            # Process/navigation questions often mention a contextual target
            # in the question that does not repeat verbatim in the procedure
            # sentence. Require strong query overlap plus an explicit process
            # signal and at least one substantive facet anchor.
            if (
                _is_process_navigation_question(frame)
                and operational_score > 0.0
                and query_score >= 0.45
                and facet_score >= 0.45
            ):
                return True, max(overall, 0.78), (
                    "lexical_recall_rescue",
                    "process_relation_supported",
                )
            return False, overall, ("relation_not_locally_grounded",)

        if _ground_target_attribute_relation(frame, candidate):
            return True, max(overall, 0.90), (
                "lexical_recall_rescue",
                "target_attribute_relation_grounded",
            )

        # Some process questions are expressed as factual relations, e.g.
        # "What comes after JEE Advanced?". They are relation-sensitive, but
        # the relationship itself is explicitly present in the candidate text.
        # A strong lexical match plus an operational/process signal is enough
        # to recover them without opening generic relation inference.
        if (
            _is_process_navigation_question(frame)
            and operational_score > 0.0
            and overall >= 0.75
            and query_score >= 0.70
        ):
            return True, max(overall, 0.78), (
                "lexical_recall_rescue",
                "process_relation_text_supported",
            )

        # Direct lookups such as "M.Tech eligibility requirements" have one
        # conceptual target rather than two independent entities.
        if target_score >= 0.85 and facet_score >= 0.55 and query_score >= 0.55:
            return True, max(overall, 0.80), (
                "lexical_recall_rescue",
                "direct_lookup_support",
            )
        return False, overall, ("strict_relation_not_grounded",)

    capability, process_navigation, multi_facet = _controlled_aggregation_shape(frame)
    request_type = str(getattr(frame, "request_type", "") or "").lower()
    target_text = str(getattr(frame, "target", "") or "").strip()
    facets = _substantive_facets(frame)

    # When the query does not carry a stable structured target/facet, a strong
    # direct lexical match is the safest generic recall path. This covers
    # natural questions such as "Can parents visit the campus?" where the
    # underlying answer text can match the user's substantive words closely
    # even when semantic target extraction is intentionally conservative.
    if (
        not target_text
        and not facets
        and overall >= 0.72
        and query_score >= 0.72
        and _scope_is_compatible(candidate)
        and not _explicit_conflict(candidate)
    ):
        return True, max(overall, 0.72), (
            "lexical_recall_rescue",
            "strong_direct_query_overlap",
        )

    if process_navigation or capability or multi_facet or request_type in {
        "information",
        "research",
        "directions",
        "procedure",
        "admission",
        "eligibility",
    }:
        anchor = max(target_score, facet_score)
        operational_ok = operational_score > 0.0 if process_navigation else True
        if overall >= 0.58 and query_score >= 0.50 and anchor >= 0.50 and operational_ok:
            kind = (
                "process"
                if process_navigation
                else "capability"
                if capability
                else "facet"
                if multi_facet
                else "query"
            )
            return True, max(overall, 0.60), (
                "lexical_recall_rescue",
                f"lexical_{kind}_support",
            )

        # Role/identity/navigation questions such as "Who is a faculty
        # advisor?" can legitimately lack a process verb even though the
        # verifier classifies the question as navigation. Strong direct text
        # overlap is enough to recover such candidates, while the earlier
        # relation-sensitive branch still protects concrete relationships.
        if process_navigation and operational_score > 0.0 and overall >= 0.55 and query_score >= 0.45:
            return True, max(overall, 0.70), (
                "lexical_recall_rescue",
                "process_text_and_query_overlap",
            )

        if process_navigation and overall >= 0.72 and query_score >= 0.72:
            return True, max(overall, 0.72), (
                "lexical_recall_rescue",
                "strong_direct_query_overlap",
            )

        return False, overall, ("lexical_rescue_threshold_not_met",)

    # Final generic lexical safety net. This is intentionally stricter than
    # ordinary BM25 recall: a candidate must contain a large fraction of the
    # user's substantive words. It exists so missing/weak semantic metadata
    # cannot turn a plainly matching corpus sentence into a false negative.
    if overall >= 0.72 and query_score >= 0.72:
        return True, max(overall, 0.72), (
            "lexical_recall_rescue",
            "strong_direct_query_overlap",
        )

    if overall >= 0.68 and query_score >= 0.58:
        return True, max(overall, 0.65), ("lexical_recall_rescue",)

    return False, overall, ("lexical_rescue_threshold_not_met",)


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------


def _resolved_entities(frame: SemanticQueryFrame) -> tuple[object, ...]:
    result=[]
    for entity in (getattr(frame,"entities",()) or ()):
        state=str(getattr(entity,"resolution_state","") or "").casefold()
        confidence=float(getattr(entity,"confidence",0.0) or 0.0)
        if state in {"resolved","approved","trusted"} or confidence>=0.95:
            result.append(entity)
    return tuple(result)

def _entity_grounded(entity: object, candidate: RetrievalCandidate) -> bool:
    name=_normalize(str(getattr(entity,"name","") or "")); entity_id=_normalize(str(getattr(entity,"entity_id","") or ""))
    if not name and not entity_id: return False
    text=_document_text(candidate)
    if name and _contains_phrase(text,name): return True
    if entity_id and _contains_phrase(text,entity_id): return True
    meaning=candidate.meaning; kind=_normalize(str(getattr(entity,"entity_type","entity") or "entity"))
    field="programs" if kind=="program" else "topics" if kind=="topic" else "attributes" if kind=="attribute" else "entities"
    wanted=_meaningful_tokens(name or entity_id)
    for value in (getattr(meaning,field,()) or ()):
        v=_normalize(value)
        if v==(name or entity_id): return True
        tokens=_meaningful_tokens(v)
        if wanted and tokens and len(wanted & tokens)/len(wanted)>=0.90: return True
    return False

def _all_required_entities_grounded(frame: SemanticQueryFrame, candidate: RetrievalCandidate) -> bool:
    return all(_entity_grounded(e,candidate) for e in _resolved_entities(frame))

def verify_candidate(
    frame: SemanticQueryFrame,
    candidate: RetrievalCandidate,
) -> VerificationDecision:
    """
    Verify one retrieved candidate against the query frame.

    Exact mode:

        conflict
            -> reject

        required target missing
            -> reject

        required attribute missing
            -> reject

        target+attribute local relation missing
            -> reject

        required scope mismatch
            -> reject

        otherwise
            -> verified

    Standard/focused mode remains more permissive.
    """

    reasons: list[str] = []

    mode = _requirement_mode(
        frame
    )

    strict_exact_request = _query_requires_strict_exact_grounding(frame)
    exact_request = strict_exact_request

    # A boolean/relation frame may intentionally use True/False as the target.
    # That placeholder is not a factual entity and must never force target
    # grounding by itself, even when an upstream requirement flag says true.
    requires_target = (
        _requirement_flag(
            frame,
            "require_target_alignment",
            default=(exact_request and not _is_placeholder_target(frame)),
        )
        and not _is_placeholder_target(frame)
    )

    requires_attribute = _requirement_flag(
        frame,
        "require_attribute_alignment",
        default=exact_request,
    )

    requires_scope = _requirement_flag(
        frame,
        "require_scope_alignment",
        default=False,
    )

    reject_conflict = _requirement_flag(
        frame,
        "reject_explicit_conflict",
        default=True,
    )

    target_grounded = _ground_target(
        frame,
        candidate,
    )

    resolved_entities = _resolved_entities(frame)
    entity_grounded = _all_required_entities_grounded(frame,candidate)

    attribute_grounded = _ground_attribute(
        frame,
        candidate,
    )

    scope_compatible = _scope_is_compatible(
        candidate
    )

    semantic_compatible, semantic_score = (
        _semantic_compatibility(
            candidate
        )
    )

    conflict_detected = _explicit_conflict(
        candidate
    )

    target_attribute_relation = (
        _ground_target_attribute_relation(
            frame,
            candidate,
        )
    )

    controlled_shape = _controlled_aggregation_shape(frame)
    controlled_family_requested = any(controlled_shape)

    controlled_support, facet_coverage, aggregation_kind = (
        _controlled_facet_aggregation(
            frame,
            candidate,
        )
    )

    placeholder_exact_support = (
        exact_request
        and _is_placeholder_target(frame)
        and _exact_placeholder_facet_support(frame, candidate)
    )

    # For a concrete target, a standard/focused answer must not mix a target
    # fact with an unrelated attribute fact merely because both are present in
    # the same document. Controlled aggregation is the only escape hatch, and
    # it has already applied its stronger local-connectivity checks above.
    anti_mixing_failure = (
        not exact_request
        and not controlled_support
        and not _is_placeholder_target(frame)
        and target_grounded
        and attribute_grounded
        and bool(_substantive_facets(frame))
        and not target_attribute_relation
    )

    lexical_rescue, lexical_score, lexical_reasons = _lexical_rescue_allowed(
        frame,
        candidate,
        exact_request=exact_request,
    )

    if lexical_rescue:
        reasons.extend(lexical_reasons)

    if anti_mixing_failure and not lexical_rescue:
        reasons.append(
            "target_attribute_relation_not_grounded"
        )

    if conflict_detected:
        reasons.append(
            "explicit_conflict"
        )

    if (
        requires_target
        and not target_grounded
        and not lexical_rescue
    ):
        reasons.append(
            "required_target_not_grounded"
        )

    if (
        requires_attribute
        and not attribute_grounded
        and not lexical_rescue
    ):
        reasons.append(
            "required_attribute_not_grounded"
        )

    if (
        exact_request
        and requires_target
        and requires_attribute
        and not target_attribute_relation
        and not lexical_rescue
    ):
        reasons.append(
            "required_target_attribute_relation_not_grounded"
        )

    if (
        requires_scope
        and not scope_compatible
    ):
        reasons.append(
            "scope_mismatch"
        )

    if semantic_compatible:
        reasons.append(
            "semantic_compatibility_present"
        )
    else:
        reasons.append(
            "semantic_compatibility_weak"
        )

    if controlled_support:
        reasons.append(
            f"controlled_{aggregation_kind}_evidence"
        )

    # -----------------------------------------------------------------------
    # Coverage calculation
    # -----------------------------------------------------------------------

    coverage_parts: list[float] = []

    if requires_target:
        coverage_parts.append(
            1.0
            if target_grounded
            else 0.0
        )

    if resolved_entities:
        coverage_parts.append(1.0 if entity_grounded else 0.0)

    if requires_attribute:
        coverage_parts.append(
            1.0
            if attribute_grounded
            else 0.0
        )

    if (
        exact_request
        and requires_target
        and requires_attribute
    ):
        coverage_parts.append(
            1.0
            if target_attribute_relation
            else 0.0
        )

    if requires_scope:
        coverage_parts.append(
            1.0
            if scope_compatible
            else 0.0
        )

    if coverage_parts:
        coverage = (
            sum(coverage_parts)
            / len(coverage_parts)
        )
    else:
        coverage = semantic_score

    if controlled_support:
        coverage = max(
            coverage,
            facet_coverage,
        )

    # -----------------------------------------------------------------------
    # EXACT REQUESTS
    # -----------------------------------------------------------------------

    if exact_request:
        if resolved_entities and not entity_grounded:
            return VerificationDecision(
                candidate=candidate, status="rejected", accepted=False,
                target_grounded=target_grounded, attribute_grounded=attribute_grounded,
                scope_compatible=scope_compatible, semantic_compatible=semantic_compatible,
                conflict_detected=conflict_detected, coverage=coverage, score=coverage,
                reasons=tuple(reasons),
            )
        if (
            reject_conflict
            and conflict_detected
        ):
            return VerificationDecision(
                candidate=candidate,
                status="rejected",
                accepted=False,
                target_grounded=target_grounded,
                attribute_grounded=attribute_grounded,
                scope_compatible=scope_compatible,
                semantic_compatible=semantic_compatible,
                conflict_detected=conflict_detected,
                coverage=coverage,
                score=0.0,
                reasons=tuple(reasons),
            )

        if (
            requires_target
            and not target_grounded
            and not lexical_rescue
        ):
            return VerificationDecision(
                candidate=candidate,
                status="rejected",
                accepted=False,
                target_grounded=target_grounded,
                attribute_grounded=attribute_grounded,
                scope_compatible=scope_compatible,
                semantic_compatible=semantic_compatible,
                conflict_detected=conflict_detected,
                coverage=coverage,
                score=coverage,
                reasons=tuple(reasons),
            )

        if (
            requires_attribute
            and not attribute_grounded
            and not lexical_rescue
        ):
            return VerificationDecision(
                candidate=candidate,
                status="rejected",
                accepted=False,
                target_grounded=target_grounded,
                attribute_grounded=attribute_grounded,
                scope_compatible=scope_compatible,
                semantic_compatible=semantic_compatible,
                conflict_detected=conflict_detected,
                coverage=coverage,
                score=coverage,
                reasons=tuple(reasons),
            )

        if (
            requires_target
            and requires_attribute
            and not target_attribute_relation
            and not lexical_rescue
        ):
            return VerificationDecision(
                candidate=candidate,
                status="rejected",
                accepted=False,
                target_grounded=target_grounded,
                attribute_grounded=attribute_grounded,
                scope_compatible=scope_compatible,
                semantic_compatible=semantic_compatible,
                conflict_detected=conflict_detected,
                coverage=coverage,
                score=coverage,
                reasons=tuple(reasons),
            )

        if (
            _is_placeholder_target(frame)
            and not placeholder_exact_support
            and not lexical_rescue
        ):
            reasons.append(
                "required_facet_evidence_not_grounded"
            )
            return VerificationDecision(
                candidate=candidate,
                status="rejected",
                accepted=False,
                target_grounded=target_grounded,
                attribute_grounded=attribute_grounded,
                scope_compatible=scope_compatible,
                semantic_compatible=semantic_compatible,
                conflict_detected=conflict_detected,
                coverage=coverage,
                score=coverage,
                reasons=tuple(reasons),
            )

        if (
            requires_scope
            and not scope_compatible
        ):
            return VerificationDecision(
                candidate=candidate,
                status="rejected",
                accepted=False,
                target_grounded=target_grounded,
                attribute_grounded=attribute_grounded,
                scope_compatible=scope_compatible,
                semantic_compatible=semantic_compatible,
                conflict_detected=conflict_detected,
                coverage=coverage,
                score=coverage,
                reasons=tuple(reasons),
            )

        final_score = (
            0.60 * coverage
            + 0.40 * semantic_score
        )
        if lexical_rescue:
            final_score = max(final_score, 0.55 + 0.45 * lexical_score)

        return VerificationDecision(
            candidate=candidate,
            status="verified",
            accepted=True,
            target_grounded=target_grounded,
            attribute_grounded=attribute_grounded,
            scope_compatible=scope_compatible,
            semantic_compatible=semantic_compatible,
            conflict_detected=conflict_detected,
            coverage=coverage,
            score=final_score,
            reasons=tuple(reasons),
        )

    # -----------------------------------------------------------------------
    # STANDARD / FOCUSED REQUESTS
    # -----------------------------------------------------------------------

    if (
        reject_conflict
        and conflict_detected
    ):
        return VerificationDecision(
            candidate=candidate,
            status="rejected",
            accepted=False,
            target_grounded=target_grounded,
            attribute_grounded=attribute_grounded,
            scope_compatible=scope_compatible,
            semantic_compatible=semantic_compatible,
            conflict_detected=conflict_detected,
            coverage=coverage,
            score=0.0,
            reasons=tuple(reasons),
        )

    if anti_mixing_failure and not lexical_rescue:
        return VerificationDecision(
            candidate=candidate,
            status="rejected",
            accepted=False,
            target_grounded=target_grounded,
            attribute_grounded=attribute_grounded,
            scope_compatible=scope_compatible,
            semantic_compatible=semantic_compatible,
            conflict_detected=conflict_detected,
            coverage=coverage,
            score=coverage,
            reasons=tuple(reasons),
        )

    if (
        requires_target
        and not target_grounded
        and not controlled_support
        and not lexical_rescue
    ):
        return VerificationDecision(
            candidate=candidate,
            status="rejected",
            accepted=False,
            target_grounded=target_grounded,
            attribute_grounded=attribute_grounded,
            scope_compatible=scope_compatible,
            semantic_compatible=semantic_compatible,
            conflict_detected=conflict_detected,
            coverage=coverage,
            score=coverage,
            reasons=tuple(reasons),
        )

    if (
        controlled_support
        and scope_compatible
        and (
            _is_placeholder_target(frame)
            or target_grounded
        )
    ):
        final_score = (
            0.70 * coverage
            + 0.30 * semantic_score
        )

        return VerificationDecision(
            candidate=candidate,
            status="verified",
            accepted=True,
            target_grounded=target_grounded,
            attribute_grounded=attribute_grounded,
            scope_compatible=scope_compatible,
            semantic_compatible=semantic_compatible,
            conflict_detected=conflict_detected,
            coverage=coverage,
            score=final_score,
            reasons=tuple(reasons),
        )

    # Semantic compatibility is never a substitute for factual evidence when
    # the question itself belongs to a controlled capability/process/multi-
    # facet family. In those families, failure to establish the required
    # facets must not fall through to a generic semantic acceptance.
    if (
        (semantic_compatible or lexical_rescue)
        and scope_compatible
        and (not controlled_family_requested or lexical_rescue)
        and (not resolved_entities or entity_grounded)
    ):
        final_score = (
            0.50 * coverage
            + 0.50 * semantic_score
        )

        return VerificationDecision(
            candidate=candidate,
            status="verified",
            accepted=True,
            target_grounded=target_grounded,
            attribute_grounded=attribute_grounded,
            scope_compatible=scope_compatible,
            semantic_compatible=semantic_compatible,
            conflict_detected=conflict_detected,
            coverage=coverage,
            score=final_score,
            reasons=tuple(reasons),
        )

    if lexical_rescue and scope_compatible and not conflict_detected and (not resolved_entities or entity_grounded):
        return VerificationDecision(
            candidate=candidate,
            status="verified",
            accepted=True,
            target_grounded=target_grounded,
            attribute_grounded=attribute_grounded,
            scope_compatible=scope_compatible,
            semantic_compatible=semantic_compatible,
            conflict_detected=conflict_detected,
            coverage=max(coverage, lexical_score),
            score=max(0.55 + 0.45 * lexical_score, lexical_score),
            reasons=tuple(reasons),
        )

    return VerificationDecision(
        candidate=candidate,
        status="uncertain",
        accepted=False,
        target_grounded=target_grounded,
        attribute_grounded=attribute_grounded,
        scope_compatible=scope_compatible,
        semantic_compatible=semantic_compatible,
        conflict_detected=conflict_detected,
        coverage=coverage,
        score=coverage,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Batch verification
# ---------------------------------------------------------------------------


def verify_candidates(
    frame: SemanticQueryFrame,
    candidates: Sequence[RetrievalCandidate],
) -> VerificationBatch:
    verified: list[RetrievalCandidate] = []
    uncertain: list[RetrievalCandidate] = []
    rejected: list[RetrievalCandidate] = []
    decisions: list[VerificationDecision] = []

    for candidate in candidates:
        decision = verify_candidate(
            frame,
            candidate,
        )

        decisions.append(
            decision
        )

        if decision.status == "verified":
            verified.append(
                candidate
            )

        elif decision.status == "uncertain":
            uncertain.append(
                candidate
            )

        else:
            rejected.append(
                candidate
            )

    return VerificationBatch(
        verified=tuple(verified),
        uncertain=tuple(uncertain),
        rejected=tuple(rejected),
        decisions=tuple(decisions),
    )


def select_verified_candidates(
    frame: SemanticQueryFrame,
    candidates: Sequence[RetrievalCandidate],
    top_k: int | None = None,
) -> tuple[RetrievalCandidate, ...]:
    """
    Return only verified candidates.

    Verification score is the primary ordering signal.
    Retrieval final_score remains a secondary tie-breaker.
    """

    batch = verify_candidates(
        frame,
        candidates,
    )

    decision_by_id = {
        decision.candidate.document_id: decision
        for decision in batch.decisions
    }

    ordered = sorted(
        batch.verified,
        key=lambda candidate: (
            decision_by_id[
                candidate.document_id
            ].score,
            float(
                getattr(
                    candidate,
                    "final_score",
                    0.0,
                )
                or 0.0
            ),
        ),
        reverse=True,
    )

    if top_k is None:
        return tuple(ordered)

    return tuple(
        ordered[
            :max(
                0,
                top_k,
            )
        ]
    )


def explain_verification(
    frame: SemanticQueryFrame,
    candidates: Iterable[RetrievalCandidate],
) -> list[dict]:
    """
    Debug/evaluation helper.
    """

    return [
        verify_candidate(
            frame,
            candidate,
        ).to_dict()
        for candidate in candidates
    ]


__all__ = [
    "VerificationDecision",
    "VerificationBatch",
    "verify_candidate",
    "verify_candidates",
    "select_verified_candidates",
    "explain_verification",
]