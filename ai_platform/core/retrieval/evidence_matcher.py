"""Generic evidence matching for retrieval verification.

This module contains no institution-specific vocabulary or source paths.
It separates factual evidence from soft retrieval/semantic signals so that
semantic similarity cannot manufacture a required fact that is absent from
the candidate evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from ai_platform.core.query.models import SemanticQueryFrame
from ai_platform.core.retrieval.contracts import RetrievalCandidate


# Generic request-type vocabulary only. It describes information needs,
# not any institution, program, department, or source.
REQUEST_TERMS: dict[str, tuple[str, ...]] = {
    "cost": ("fee", "fees", "cost", "amount", "charge", "charges", "price", "rate", "rates"),
    "registration": ("registration", "register", "registered", "enrollment", "enrolment", "enroll", "enrol"),
    "admission": ("admission", "admissions", "application", "apply", "entrance", "selection"),
    "definition": ("definition", "meaning", "means", "defined"),
    "eligibility": ("eligibility", "eligible", "criteria", "requirements", "qualify", "qualification"),
    "comparison": ("compare", "comparison", "difference", "versus", "vs"),
    "directions": ("direction", "directions", "where", "location", "located", "route", "reach"),
    "procedure": ("procedure", "process", "steps", "step", "how", "method"),
    "research": ("research", "areas", "opportunities", "work"),
    "information": (),
    "other": (),
}

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "for",
    "in", "on", "at", "and", "or", "can", "could", "would", "should", "what",
    "which", "how", "why", "when", "where", "who", "whom", "does", "do", "did",
    "i", "we", "you", "me", "my", "your", "please", "tell", "give", "show", "get",
    "this", "that", "it", "they", "their", "there", "here", "ke", "ka", "ki", "mein",
    "mujhe", "kya", "kitna", "kitne", "kaise", "hai", "hain", "karo", "karna", "liye",
}

PLACEHOLDER_TARGETS = {"", "true", "false", "yes", "no", "answer", "boolean"}


@dataclass(frozen=True, slots=True)
class EvidenceAnalysis:
    """Evidence features computed for one candidate."""

    target_score: float
    attribute_score: float
    relation_score: float
    entity_score: float
    query_overlap: float
    semantic_score: float
    scope_score: float
    conflict: bool
    hard_evidence: bool
    evidence_units: tuple[str, ...]
    supporting_units: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def factual_score(self) -> float:
        return (
            0.38 * self.target_score
            + 0.27 * self.attribute_score
            + 0.25 * self.relation_score
            + 0.10 * self.entity_score
        )

    @property
    def ranking_score(self) -> float:
        return (
            0.72 * self.factual_score
            + 0.12 * self.query_overlap
            + 0.12 * self.semantic_score
            + 0.04 * self.scope_score
        )

    def to_dict(self) -> dict:
        return {
            "target_score": self.target_score,
            "attribute_score": self.attribute_score,
            "relation_score": self.relation_score,
            "entity_score": self.entity_score,
            "query_overlap": self.query_overlap,
            "semantic_score": self.semantic_score,
            "scope_score": self.scope_score,
            "conflict": self.conflict,
            "hard_evidence": self.hard_evidence,
            "factual_score": self.factual_score,
            "ranking_score": self.ranking_score,
            "evidence_units": list(self.evidence_units),
            "supporting_units": list(self.supporting_units),
            "reasons": list(self.reasons),
        }


def normalize(text: str) -> str:
    value = str(text or "").casefold().replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9\u0900-\u097f\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokens(text: str) -> tuple[str, ...]:
    value = normalize(text)
    return tuple(value.split()) if value else ()


def meaningful_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if len(t) >= 2 and t not in STOP_WORDS}


def compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize(text))


def _singular_forms(token: str) -> set[str]:
    value = normalize(token)
    if not value:
        return set()
    forms = {value}
    if value.endswith("ies") and len(value) > 4:
        forms.add(value[:-3] + "y")
    if value.endswith("es") and len(value) > 4:
        forms.add(value[:-2])
    if value.endswith("s") and len(value) > 3:
        forms.add(value[:-1])
    if not value.endswith("s") and len(value) > 2:
        forms.add(value + "s")
    return forms


def _token_ratio(query: str, text: str) -> float:
    q = meaningful_tokens(query)
    if not q:
        return 0.0
    d = set(tokens(text))
    matched = sum(bool(_singular_forms(t) & d) for t in q)
    return matched / len(q)


def _phrase_or_ratio(phrase: str, text: str) -> float:
    phrase_n = normalize(phrase)
    text_n = normalize(text)
    if not phrase_n or not text_n:
        return 0.0
    if phrase_n in text_n:
        return 1.0
    # Handle canonical forms such as ``M.Tech`` <-> ``mtech`` and
    # punctuation/spacing variants without introducing institution-specific
    # aliases.
    phrase_compact = compact(phrase_n)
    text_compact = compact(text_n)
    if phrase_compact and phrase_compact in text_compact:
        return 1.0
    q = meaningful_tokens(phrase_n)
    if not q:
        return 0.0
    d = set(tokens(text_n))
    matched = sum(bool(_singular_forms(t) & d) for t in q)
    return matched / len(q)


def split_units(text: str) -> tuple[str, ...]:
    raw = str(text or "").strip()
    if not raw:
        return ()
    raw = re.sub(r"\r\n?", "\n", raw)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        for line in [x.strip() for x in paragraph.split("\n") if x.strip()]:
            pieces = re.split(r"(?<=[.!?।])\s+", line)
            units.extend(p.strip() for p in pieces if p.strip())
    return tuple(units)


def _meaning_values(candidate: RetrievalCandidate) -> tuple[str, ...]:
    meaning = getattr(candidate, "meaning", None)
    if meaning is None:
        return ()
    values: list[str] = []
    for field in (
        "programs", "entities", "topics", "attributes", "scope", "qualifiers", "constraints"
    ):
        values.extend(str(v) for v in (getattr(meaning, field, ()) or ()) if str(v).strip())
    intent = getattr(meaning, "intent", None)
    if intent:
        values.append(str(intent))
    return tuple(values)


def _candidate_text(candidate: RetrievalCandidate) -> str:
    return str(getattr(getattr(candidate, "document", None), "page_content", "") or "").strip()


def _target_aliases(frame: SemanticQueryFrame) -> tuple[str, ...]:
    values: list[str] = []
    target = str(getattr(frame, "target", "") or "").strip()
    if target:
        values.append(target)

    entities = getattr(frame, "entities", ()) or ()
    for entity in entities:
        state = str(getattr(entity, "resolution_state", "") or "").casefold()
        confidence = float(getattr(entity, "confidence", 0.0) or 0.0)
        if state not in {"resolved", "approved", "trusted"} and confidence < 0.90:
            continue
        for field in ("name", "entity_id", "aliases"):
            value = getattr(entity, field, None)
            if isinstance(value, (tuple, list)):
                values.extend(str(v) for v in value)
            elif value:
                values.append(str(value))

    # preserve order and remove normalized duplicates
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        n = normalize(value)
        if n and n not in seen:
            seen.add(n)
            out.append(value)
    return tuple(out)


def _attribute_phrases(frame: SemanticQueryFrame) -> tuple[str, ...]:
    values: list[str] = []
    for facet in getattr(frame, "facets", ()) or ():
        value = str(getattr(facet, "value", "") or "").strip()
        if value:
            values.append(value)
    for value in getattr(frame, "qualifiers", ()) or ():
        if str(value).strip():
            values.append(str(value))
    request_type = normalize(str(getattr(frame, "request_type", "") or ""))
    values.extend(REQUEST_TERMS.get(request_type, ()))

    target_tokens = meaningful_tokens(str(getattr(frame, "target", "") or ""))
    query_words = meaningful_tokens(
        " ".join(
            [
                str(getattr(frame, "original_query", "") or ""),
                str(getattr(frame, "semantic_query", "") or ""),
                " ".join(str(x) for x in (getattr(frame, "preserved_terms", ()) or ())),
            ]
        )
    )
    # Generic lexical recovery for attributes not explicitly named as facets.
    for word in sorted(query_words - target_tokens):
        if word not in STOP_WORDS and len(word) >= 3:
            values.append(word)

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        n = normalize(value)
        if not n or n in seen:
            continue
        if len(n) < 2:
            continue
        seen.add(n)
        out.append(n)
    return tuple(out)


def _target_match(frame: SemanticQueryFrame, candidate: RetrievalCandidate, units: Sequence[str]) -> float:
    aliases = _target_aliases(frame)
    if not aliases:
        return 0.0

    text_score = max((_phrase_or_ratio(alias, _candidate_text(candidate)) for alias in aliases), default=0.0)
    meaning_values = _meaning_values(candidate)
    meaning_score = max(
        (_phrase_or_ratio(alias, " ".join(meaning_values)) for alias in aliases),
        default=0.0,
    )
    unit_score = max(
        (_phrase_or_ratio(alias, unit) for alias in aliases for unit in units),
        default=0.0,
    )
    return max(text_score, unit_score, 0.95 * meaning_score)


def _attribute_match(frame: SemanticQueryFrame, candidate: RetrievalCandidate, units: Sequence[str]) -> float:
    phrases = _attribute_phrases(frame)
    if not phrases:
        return 0.0
    text = _candidate_text(candidate)
    text_score = max((_phrase_or_ratio(p, text) for p in phrases), default=0.0)
    unit_score = max((_phrase_or_ratio(p, u) for p in phrases for u in units), default=0.0)
    meaning = getattr(candidate, "meaning", None)
    meaning_attrs = " ".join(str(v) for v in (getattr(meaning, "attributes", ()) or ()))
    meaning_score = max((_phrase_or_ratio(p, meaning_attrs) for p in phrases), default=0.0)
    return max(text_score, unit_score, 0.90 * meaning_score)


def _entity_match(frame: SemanticQueryFrame, candidate: RetrievalCandidate) -> float:
    entities = getattr(frame, "entities", ()) or ()
    if not entities:
        return 0.0
    text = _candidate_text(candidate)
    meaning = " ".join(_meaning_values(candidate))
    scores: list[float] = []
    for entity in entities:
        state = str(getattr(entity, "resolution_state", "") or "").casefold()
        confidence = float(getattr(entity, "confidence", 0.0) or 0.0)
        if state not in {"resolved", "approved", "trusted"} and confidence < 0.90:
            continue
        aliases = [str(getattr(entity, "name", "") or "")]
        aliases.extend(str(x) for x in (getattr(entity, "aliases", ()) or ()))
        entity_id = getattr(entity, "entity_id", None)
        if entity_id:
            aliases.append(str(entity_id))
        scores.append(max((_phrase_or_ratio(a, text) for a in aliases if a), default=0.0))
        scores.append(max((_phrase_or_ratio(a, meaning) for a in aliases if a), default=0.0))
    return max(scores, default=0.0)


def _local_relation(frame: SemanticQueryFrame, candidate: RetrievalCandidate, units: Sequence[str]) -> tuple[float, tuple[str, ...]]:
    aliases = _target_aliases(frame)
    attributes = _attribute_phrases(frame)
    if not aliases or not attributes:
        return 0.0, ()

    supporting: list[str] = []

    # Strongest proof: the target and at least one requested facet occur in the
    # same local evidence unit.
    for unit in units:
        target_here = max((_phrase_or_ratio(a, unit) for a in aliases), default=0.0)
        attr_here = max((_phrase_or_ratio(p, unit) for p in attributes), default=0.0)
        if target_here >= 0.80 and attr_here >= 0.70:
            supporting.append(unit)
    if supporting:
        return 1.0, tuple(supporting[:4])

    # Second proof: two adjacent sentences in the same paragraph jointly carry
    # the request. No arbitrary distant-document join is permitted.
    raw = _candidate_text(candidate)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    query = str(getattr(frame, "original_query", "") or "")
    for paragraph in paragraphs:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", paragraph) if s.strip()]
        for left, right in zip(sentences, sentences[1:]):
            pair = f"{left} {right}"
            target_here = max((_phrase_or_ratio(a, pair) for a in aliases), default=0.0)
            attr_here = max((_phrase_or_ratio(p, pair) for p in attributes), default=0.0)
            overlap = _token_ratio(query, pair)
            if target_here >= 0.75 and attr_here >= 0.65 and overlap >= 0.45:
                supporting.append(pair)
    if supporting:
        return 0.82, tuple(supporting[:4])

    # Third proof: explicit structured meaning relation. Structured meaning is
    # useful when the document text is sparse or uses a different surface form,
    # but it must NOT be allowed to bridge two contradictory/distant textual
    # facts. Therefore metadata can supply a missing side only when that side
    # is not independently present somewhere in the text.
    raw_text_normalized = normalize(_candidate_text(candidate))
    text_target_present = any(alias and normalize(alias) in raw_text_normalized for alias in aliases)
    text_attribute_present = any(phrase and normalize(phrase) in raw_text_normalized for phrase in attributes)
    if text_target_present and text_attribute_present:
        return 0.0, ()

    meaning = getattr(candidate, "meaning", None)
    target_values = []
    for field in ("programs", "entities", "topics"):
        target_values.extend(str(v) for v in (getattr(meaning, field, ()) or ()))
    attribute_values = [str(v) for v in (getattr(meaning, "attributes", ()) or ())]
    target_ok = max((_phrase_or_ratio(a, " ".join(target_values)) for a in aliases), default=0.0) >= 0.90
    attr_ok = max((_phrase_or_ratio(p, " ".join(attribute_values)) for p in attributes), default=0.0) >= 0.90
    semantic_match = float(getattr(getattr(candidate, "alignment", None), "semantic_match", 0.0) or 0.0)
    if target_ok and attr_ok and semantic_match >= 0.80:
        return 0.75, ("structured meaning: target+attribute",)

    return 0.0, ()


def _query_overlap(frame: SemanticQueryFrame, candidate: RetrievalCandidate) -> float:
    query = " ".join(
        [
            str(getattr(frame, "original_query", "") or ""),
            str(getattr(frame, "semantic_query", "") or ""),
            " ".join(str(x) for x in (getattr(frame, "preserved_terms", ()) or ())),
        ]
    )
    return _token_ratio(query, _candidate_text(candidate))


def _scope_score(candidate: RetrievalCandidate) -> float:
    alignment = getattr(candidate, "alignment", None)
    if alignment is None:
        return 1.0
    value = float(getattr(alignment, "scope_match", 0.0) or 0.0)
    return 1.0 if value <= 0.0 else max(0.0, min(1.0, value))


def _conflict(candidate: RetrievalCandidate) -> bool:
    alignment = getattr(candidate, "alignment", None)
    return bool(getattr(alignment, "conflicts", ()) or ()) if alignment is not None else False


def analyze_candidate(frame: SemanticQueryFrame, candidate: RetrievalCandidate) -> EvidenceAnalysis:
    units = split_units(_candidate_text(candidate))
    target = _target_match(frame, candidate, units)
    attribute = _attribute_match(frame, candidate, units)
    relation, supporting = _local_relation(frame, candidate, units)
    entity = _entity_match(frame, candidate)
    overlap = _query_overlap(frame, candidate)
    alignment = getattr(candidate, "alignment", None)
    semantic = float(getattr(alignment, "semantic_match", 0.0) or 0.0) if alignment is not None else 0.0
    scope = _scope_score(candidate)
    conflict = _conflict(candidate)

    reasons: list[str] = []
    if target >= 0.80:
        reasons.append("target_supported")
    else:
        reasons.append("target_weak_or_missing")
    if attribute >= 0.70:
        reasons.append("attribute_supported")
    else:
        reasons.append("attribute_weak_or_missing")
    if relation >= 0.70:
        reasons.append("target_attribute_relation_supported")
    else:
        reasons.append("target_attribute_relation_weak_or_missing")
    if semantic >= 0.80:
        reasons.append("semantic_compatibility_present")
    else:
        reasons.append("semantic_compatibility_weak")
    if conflict:
        reasons.append("explicit_conflict")

    hard_evidence = target >= 0.80 and (attribute >= 0.70 or not _attribute_phrases(frame))
    return EvidenceAnalysis(
        target_score=max(0.0, min(1.0, target)),
        attribute_score=max(0.0, min(1.0, attribute)),
        relation_score=max(0.0, min(1.0, relation)),
        entity_score=max(0.0, min(1.0, entity)),
        query_overlap=max(0.0, min(1.0, overlap)),
        semantic_score=max(0.0, min(1.0, semantic)),
        scope_score=max(0.0, min(1.0, scope)),
        conflict=conflict,
        hard_evidence=hard_evidence,
        evidence_units=tuple(units),
        supporting_units=supporting,
        reasons=tuple(reasons),
    )


def is_placeholder_target(frame: SemanticQueryFrame) -> bool:
    return normalize(str(getattr(frame, "target", "") or "")) in PLACEHOLDER_TARGETS


def requires_target(frame: SemanticQueryFrame) -> bool:
    requirement = getattr(frame, "requirement", None)
    default = not is_placeholder_target(frame)
    return bool(getattr(requirement, "require_target_alignment", default)) and not is_placeholder_target(frame)


def requires_attribute(frame: SemanticQueryFrame) -> bool:
    requirement = getattr(frame, "requirement", None)
    default = bool(_attribute_phrases(frame))
    return bool(getattr(requirement, "require_attribute_alignment", default))


def requires_scope(frame: SemanticQueryFrame) -> bool:
    requirement = getattr(frame, "requirement", None)
    return bool(getattr(requirement, "require_scope_alignment", False))


def is_exact(frame: SemanticQueryFrame) -> bool:
    requirement = getattr(frame, "requirement", None)
    return bool(getattr(requirement, "exact", False) or str(getattr(requirement, "mode", "") or "").casefold() == "exact")


def accepts_partial(frame: SemanticQueryFrame) -> bool:
    requirement = getattr(frame, "requirement", None)
    return bool(getattr(requirement, "allow_partial_evidence", True))
