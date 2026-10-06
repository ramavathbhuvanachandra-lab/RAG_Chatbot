"""Deterministic final-context construction for the reusable RAG core.

Pipeline position
-----------------
    verified candidates -> ranking -> final context -> answer LLM

This module is NOT an evidence gate. Every input candidate is assumed to have
already passed candidate verification and ranking.

Its only responsibilities are:
    - remove exact / near-duplicate units;
    - keep the most query-relevant sentences/paragraphs;
    - preserve useful diversity across sources/candidates;
    - obey a hard context budget;
    - return an answer-model-ready package without re-verifying or inventing
      new candidates.

No new document may enter the context from outside `ranked_candidates`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Sequence


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "for",
    "in", "on", "at", "and", "or", "can", "could", "would", "should", "what",
    "which", "how", "why", "when", "where", "who", "does", "do", "did", "i",
    "we", "you", "me", "my", "your", "please", "tell", "give", "show", "get",
    "this", "that", "it", "they", "their", "there", "here", "about", "with", "from",
    "into", "than", "then", "have", "has", "had", "will", "would", "may", "might",
    "mujhe", "kya", "kaise", "hai", "hain", "ke", "ka", "ki", "mein", "me",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")
_UNIT_SPLIT = re.compile(r"(?<=[;:])\s+|(?<=\.)\s+(?=[A-Z])")
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u0900-\u097f]+", re.IGNORECASE)

# Generic information-need synonyms. These are language-level retrieval aids,
# not institution-specific vocabulary. They help the deterministic compressor
# keep useful text when the KB uses a nearby wording such as
# "certificates/proof" for a query asking for "documents".
_FOCUS_WEIGHTS = {
    "documents": 3.0,
    "eligibility": 2.6,
    "requirements": 2.6,
    "percentage": 2.6,
    "marks": 2.4,
    "cgpa": 2.4,
    "gate": 2.6,
    "jam": 2.6,
    "written": 2.6,
    "experience": 2.6,
    "provisional": 2.6,
    "steps": 2.6,
    "process": 2.4,
    "fee": 2.6,
    "fees": 2.6,
    "facilities": 2.4,
    "research": 2.4,
    "faculty": 2.4,
}

_GENERIC_PROGRAM_MARKERS = (
    "m.tech", "m tech", "mtech", "m.sc", "m sc", "msc",
    "m.b.a", "mba", "ph.d", "phd", "b.tech", "b tech", "btech",
    "b.sc", "b sc", "bsc", "b.e", "b e",
)

_QUERY_SYNONYMS = {
    "documents": {"document", "documents", "certificate", "certificates", "proof", "scorecard", "scorecards", "photograph", "photographs", "originals", "copies"},
    "eligibility": {"eligibility", "eligible", "criteria", "criterion", "requirement", "requirements", "qualify", "qualification", "degree", "bachelor", "master", "marks", "cgpa", "percentage"},
    "qualification": {"qualification", "qualifications", "degree", "bachelor", "master", "eligibility", "eligible", "requirement", "requirements"},
    "degree": {"degree", "bachelor", "master", "qualification", "qualifications", "eligibility"},
    "percentage": {"percentage", "percent", "marks", "score", "cgpa", "minimum"},
    "marks": {"marks", "percentage", "percent", "score", "cgpa", "minimum"},
    "cgpa": {"cgpa", "marks", "percentage", "percent", "score"},
    "gate": {"gate", "score", "examination", "exam", "national"},
    "jam": {"jam", "joint", "admission", "master"},
    "fee": {"fee", "fees", "cost", "amount", "charge", "charges", "price", "rate", "processing"},
    "fees": {"fee", "fees", "cost", "amount", "charge", "charges", "price", "rate", "processing"},
    "registration": {"registration", "register", "registered", "enrollment", "enrolment", "enroll", "enrol"},
    "written": {"written", "test", "examination", "exam"},
    "experience": {"experience", "work", "industry", "laboratory", "laboratories", "rd", "research"},
    "provisional": {"provisional", "temporary", "certificate", "documents"},
    "steps": {"steps", "step", "process", "procedure", "stage", "initial", "final", "shortlisting"},
    "shortlisting": {"shortlisting", "shortlist", "initial", "final", "process", "steps"},
    "process": {"process", "procedure", "step", "steps", "stage", "initial", "final"},
}

_PROGRAM_FAMILY_PATTERNS = (
    ("mtech", re.compile(r"\b(?:m\.?\s*t\.?e\.?c\.?h|mtech)\b", re.IGNORECASE)),
    ("msc", re.compile(r"\b(?:m\.?\s*s\.?c|msc)\b", re.IGNORECASE)),
    ("mba", re.compile(r"\b(?:m\.?\s*b\.?a|mba)\b", re.IGNORECASE)),
    ("phd", re.compile(r"\b(?:ph\.?\s*d|phd)\b", re.IGNORECASE)),
    ("btech", re.compile(r"\b(?:b\.?\s*t\.?e\.?c\.?h|btech)\b", re.IGNORECASE)),
)

_FRAGMENT_PREFIXES = (
    "with ", "or ", "and ", "which ", "that ", "this ", "these ",
    "those ", "however ", "therefore ", "accordingly ", "for ",
)


@dataclass(frozen=True, slots=True)
class ContextItem:
    candidate_id: str
    source: str
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class FinalContextResult:
    query: str
    context: str
    items: tuple[ContextItem, ...]
    input_candidates: int
    selected_items: int
    omitted_items: int
    context_chars: int
    context_tokens_estimate: int
    reasons: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return bool(self.context.strip() and self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "context": self.context,
            "items": [
                {
                    "candidate_id": x.candidate_id,
                    "source": x.source,
                    "text": x.text,
                    "score": x.score,
                }
                for x in self.items
            ],
            "input_candidates": self.input_candidates,
            "selected_items": self.selected_items,
            "omitted_items": self.omitted_items,
            "context_chars": self.context_chars,
            "context_tokens_estimate": self.context_tokens_estimate,
            "reasons": list(self.reasons),
            "ready": self.ready,
        }


def _text(candidate: Any) -> str:
    document = getattr(candidate, "document", candidate)
    if isinstance(document, dict):
        return str(document.get("page_content") or "").strip()
    return str(getattr(document, "page_content", "") or "").strip()


def _source(candidate: Any) -> str:
    source = str(getattr(candidate, "source", "") or "").strip()
    if source:
        return source
    document = getattr(candidate, "document", candidate)
    metadata = getattr(document, "metadata", {}) or {}
    if isinstance(document, dict):
        metadata = document.get("metadata") or {}
    if isinstance(metadata, dict):
        for key in ("source", "source_path", "path", "url"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
    return "unknown"


def _candidate_id(candidate: Any, index: int) -> str:
    value = str(getattr(candidate, "document_id", "") or "").strip()
    return value or f"candidate_{index + 1}"


def _program_family(text: str) -> str | None:
    value = str(text or "")
    for family, pattern in _PROGRAM_FAMILY_PATTERNS:
        if pattern.search(value):
            return family
    return None


def _query_phrases(query: str) -> tuple[str, ...]:
    tokens = [t for t in re.findall(r"[a-z0-9]+", query.casefold()) if len(t) > 2 and t not in _STOPWORDS]
    phrases: list[str] = []
    for width in (3, 2):
        for i in range(max(0, len(tokens) - width + 1)):
            phrase = " ".join(tokens[i:i + width])
            if phrase not in phrases:
                phrases.append(phrase)
    return tuple(phrases)


def _query_focus_flags(query: str) -> set[str]:
    q = query.casefold()
    flags: set[str] = set()
    phrase_map = {
        "percentage": ("percentage", "percent", "%"),
        "marks": ("marks", "score", "cgpa"),
        "experience": ("work experience", "years of work", "industry", "r&d", "research laboratories"),
        "documents": ("document", "documents", "certificate", "proof"),
        "gate": ("gate",),
        "jam": ("jam",),
        "written": ("written test", "written", "exam", "examination"),
        "fee": ("fee", "fees", "cost", "amount", "price"),
        "provisional": ("provisional",),
        "steps": ("steps", "step", "process", "procedure", "shortlisting"),
    }
    for flag, markers in phrase_map.items():
        if any(marker in q for marker in markers):
            flags.add(flag)
    return flags


def _focus_bonus(query: str, unit: str) -> float:
    q = query.casefold()
    u = unit.casefold()
    flags = _query_focus_flags(query)
    bonus = 0.0

    # Generic query phrase preservation. This is not domain-specific: a phrase
    # explicitly present in the question is stronger evidence than isolated
    # query tokens.
    for phrase in _query_phrases(query):
        if len(phrase.split()) >= 2 and phrase in u:
            bonus += 0.10
            break

    if "percentage" in flags and ("%" in u or "percent" in u or "percentage" in u or "minimum" in u and "marks" in u):
        bonus += 0.30
    if "marks" in flags and ("marks" in u or "cgpa" in u or "%" in u):
        bonus += 0.24
    if "experience" in flags and ("work experience" in u or "years of work" in u or "industry" in u or "r&d" in u or "research laboratories" in u):
        bonus += 0.32
    if "documents" in flags and any(x in u for x in ("certificate", "certificates", "proof", "scorecard", "score cards", "photograph", "identity proof", "address")):
        bonus += 0.28
    if "gate" in flags and "gate" in u:
        bonus += 0.28
    if "jam" in flags and ("jam" in u or "joint admission test" in u):
        bonus += 0.28
    if "written" in flags and "written" in u and ("test" in u or "examination" in u or "exam" in u):
        bonus += 0.28
    if "fee" in flags and ("fee" in u or "fees" in u or "processing" in u or "rs." in u or "₹" in u):
        bonus += 0.26
    if "provisional" in flags and "provisional" in u:
        bonus += 0.30
    if "steps" in flags and any(x in u for x in ("step", "shortlisting", "process", "procedure", "initial", "final")):
        bonus += 0.24

    if any(marker in q for marker in ("qualification", "qualifications", "degree", "bachelor", "master")) and any(
        marker in u for marker in ("bachelor's degree", "bachelor’s degree", "master's degree", "master’s degree", "qualifying degree", "degree in")
    ):
        bonus += 0.32

    # Numeric facts matter when the query asks for a quantitative constraint.
    if any(marker in q for marker in ("percentage", "marks", "cgpa", "fee", "amount", "cost", "how many", "years")):
        if re.search(r"\d", u) or "₹" in u or "rs." in u:
            bonus += 0.12

    return min(0.65, bonus)


def _is_fragment(unit: str) -> bool:
    value = " ".join(unit.strip().split()).casefold()
    return any(value.startswith(prefix) for prefix in _FRAGMENT_PREFIXES)


def _normalize_unit(text: str) -> str:
    return " ".join(text.casefold().split())


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if len(token) >= 2 and token not in _STOPWORDS
    }


def _idf_weights(query: str, units: Sequence[str]) -> dict[str, float]:
    query_tokens = _tokens(query)
    if not query_tokens:
        return {}
    n = max(1, len(units))
    df = {token: 0 for token in query_tokens}
    for unit in units:
        unit_tokens = _tokens(unit)
        for token in query_tokens & unit_tokens:
            df[token] += 1
    return {
        token: 1.0 + math.log((n + 1.0) / (count + 1.0))
        for token, count in df.items()
    }


def _unit_relevance(query: str, unit: str, weights: dict[str, float]) -> float:
    q_tokens = _tokens(query)
    u_tokens = _tokens(unit)
    if not q_tokens or not u_tokens:
        return 0.0

    total = sum(
        weights.get(token, 1.0) * _FOCUS_WEIGHTS.get(token, 1.0)
        for token in q_tokens
    )

    # Score each query concept once. A synonym can satisfy the concept without
    # being penalized as a weak secondary token: "documents" can be supported
    # by "certificates", "proof", or "score cards" in the KB.
    weighted_hits = 0.0
    for token in q_tokens:
        candidates = {token} | _QUERY_SYNONYMS.get(token, set())
        if u_tokens & candidates:
            weighted_hits += weights.get(token, 1.0) * _FOCUS_WEIGHTS.get(token, 1.0)

    score = weighted_hits / max(total, 1e-9)
    score = min(1.0, score)
    score = min(1.0, score + _focus_bonus(query, unit))

    q_norm = _normalize_unit(query)
    u_norm = _normalize_unit(unit)
    q_compact = re.sub(r"[^a-z0-9]", "", q_norm)
    u_compact = re.sub(r"[^a-z0-9]", "", u_norm)
    if q_norm and q_norm in u_norm:
        score = min(1.0, score + 0.30)
    elif q_compact and q_compact in u_compact:
        score = min(1.0, score + 0.20)
    elif len(u_tokens & q_tokens) >= 2:
        score = min(1.0, score + 0.05)

    # Query-aware noise suppression. These are generic admission-language
    # cues: do not let a regular-program question fill its context with an
    # alternate-mode paragraph, and do not prefer dual-degree text unless the
    # user explicitly asked for it.
    q_cf = q_norm
    unit_cf = u_norm

    # Suppress generic competitor-program noise when the query clearly targets
    # one program family. This remains deployment-agnostic: it operates on
    # standard academic program forms rather than institution names.
    q_program = _program_family(q_cf)
    unit_program = _program_family(unit_cf)
    if q_program and unit_program and q_program != unit_program:
        score *= 0.30

    if "regular" in q_cf and any(
        marker in unit_cf
        for marker in ("executive", "part time", "external", "sponsored")
    ):
        score *= 0.45
    if "dual degree" not in q_cf and "dual degree" in unit_cf:
        score *= 0.02
    if "jam" not in q_cf and "jam" in unit_cf:
        score *= 0.35
    if "experience" not in q_cf and "work experience" in unit_cf:
        score *= 0.30
    if ("document" in q_cf or "documents" in q_cf):
        document_evidence_terms = {
            "document", "documents", "certificate", "certificates", "proof",
            "scorecard", "scorecards", "photograph", "photographs", "identity",
            "address", "caste", "domicile", "originals", "copies", "bring",
        }
        if not (u_tokens & document_evidence_terms):
            score *= 0.25

    # Short navigation headings can be useful labels, but do not let a heading
    # displace a more factual sentence when the query asks for the underlying
    # value/detail (experience, percentage, fee, documents, etc.).
    word_count = len(u_norm.split())
    if word_count <= 8 and score < 0.45:
        score *= 0.25
    if word_count <= 10 and _query_focus_flags(query) & {"percentage", "marks", "experience", "fee", "documents", "provisional"} and not re.search(r"\d|%|₹|rs\.", unit_cf):
        if not any(marker in unit_cf for marker in ("work experience", "certificate", "proof", "fee", "provisional", "marks", "cgpa")):
            score *= 0.45

    return min(1.0, max(0.0, score))


def _jaccard(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _split_units(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", raw) if p.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        pieces = [x.strip() for x in _SENTENCE_SPLIT.split(paragraph) if x.strip()]
        if not pieces:
            pieces = [paragraph]
        for piece in pieces:
            if len(piece) <= 420:
                units.append(piece)
                continue
            smaller = [x.strip() for x in _UNIT_SPLIT.split(piece) if x.strip()]
            if not smaller:
                smaller = [piece]
            for item in smaller:
                if len(item) <= 600:
                    units.append(item)
                else:
                    # Final deterministic fallback for exceptionally long KB
                    # lines. Split only at whitespace; never invent text.
                    words = item.split()
                    buf: list[str] = []
                    used = 0
                    for word in words:
                        extra = len(word) + (1 if buf else 0)
                        if buf and used + extra > 600:
                            units.append(" ".join(buf))
                            buf = []
                            used = 0
                        buf.append(word)
                        used += extra
                    if buf:
                        units.append(" ".join(buf))
    return units


def _heading_like(unit: str) -> bool:
    value = " ".join(str(unit or "").split())
    lowered = value.casefold()
    if re.match(r"^\d+(?:\.\d+)+\b", value):
        return True
    if any(marker in lowered for marker in (
        "dual degree", "executive/part-time", "external/sponsored",
    )):
        return True
    return len(value.split()) <= 14 and any(
        marker in lowered
        for marker in (
            "program (regular)", "program (executive", "program (part-time",
            "admission process", "application process", "registration process",
            "fulfillment of admission requirements",
        )
    )


def _section_context(units: Sequence[str], index: int) -> str:
    """Return the nearest local heading, never crossing candidate boundaries."""
    for position in range(index - 1, max(-1, index - 6), -1):
        if _heading_like(units[position]):
            return units[position]
    return ""



def _contextualize_selected_unit(unit: str, candidate_units: Sequence[str], index: int) -> str:
    """Attach one immediate predecessor when the selected unit is a fragment."""
    clean = " ".join(unit.split())
    if not _is_fragment(clean) or index <= 0:
        return clean
    previous = " ".join(str(candidate_units[index - 1]).split())
    if not previous:
        return clean
    # Keep the operation local and bounded; this is context preservation, not
    # retrieval. It never crosses candidate boundaries.
    if len(previous) + len(clean) + 2 <= 850:
        return f"{previous} {clean}"
    return clean


def build_final_context(
    query: str,
    ranked_candidates: Sequence[Any],
    *,
    max_candidates: int = 8,
    max_units: int = 18,
    max_context_chars: int = 12_000,
    min_unit_score: float = 0.25,
) -> FinalContextResult:
    """Build compact context strictly from already-ranked candidates."""
    normalized_query = " ".join(str(query or "").split())
    candidates = tuple(ranked_candidates[: max(1, int(max_candidates))])

    if not normalized_query or not candidates:
        return FinalContextResult(
            query=normalized_query,
            context="",
            items=(),
            input_candidates=len(candidates),
            selected_items=0,
            omitted_items=0,
            context_chars=0,
            context_tokens_estimate=0,
            reasons=("no_query_or_ranked_candidates",),
        )

    all_units: list[tuple[int, str, str, str]] = []
    raw_units: list[str] = []
    for candidate_index, candidate in enumerate(candidates):
        candidate_units = _split_units(_text(candidate))
        for local_index, unit in enumerate(candidate_units):
            raw_units.append(unit)
            all_units.append(
                (
                    candidate_index,
                    _candidate_id(candidate, candidate_index),
                    _source(candidate),
                    unit,
                    local_index,
                    tuple(candidate_units),
                )
            )

    weights = _idf_weights(normalized_query, raw_units)

    scored: list[tuple[float, int, str, str, str]] = []
    for candidate_index, candidate_id, source, unit, local_index, candidate_units in all_units:
        section = _section_context(candidate_units, local_index)
        scoring_text = f"{section} {unit}" if section else unit
        score = _unit_relevance(normalized_query, scoring_text, weights)
        # Preserve a small amount of the upstream candidate ordering. A top-
        # ranked approved candidate is preferred when relevance is tied.
        score_with_rank = min(1.0, score + 0.02 * (1.0 - candidate_index / max(1, len(candidates))))
        if score_with_rank >= min_unit_score:
            scored.append((score_with_rank, candidate_index, candidate_id, source, unit, local_index, candidate_units))

    # Deterministic MMR-like selection: relevance first, redundancy second.
    selected: list[tuple[float, int, str, str, str]] = []
    seen_units: set[str] = set()
    source_counts: dict[str, int] = {}

    while scored and len(selected) < max(1, int(max_units)):
        best = None
        best_mmr = -1.0
        for row in scored:
            score, candidate_index, candidate_id, source, unit, local_index, candidate_units = row
            key = _normalize_unit(unit)
            if not key or key in seen_units:
                continue

            redundancy = max(
                (_jaccard(unit, chosen[4]) for chosen in selected),
                default=0.0,
            )
            if redundancy >= 0.68:
                continue

            # Do not let one document monopolize the whole context. Two units
            # from the same source are generally enough unless the query has
            # very strong lexical support.
            source_penalty = 0.10 if source_counts.get(source, 0) >= 2 else 0.0
            mmr = 0.82 * score - 0.18 * redundancy - source_penalty

            if mmr > best_mmr:
                best_mmr = mmr
                best = row

        if best is None:
            break

        scored.remove(best)
        selected.append(best)
        seen_units.add(_normalize_unit(best[4]))
        source_counts[best[3]] = source_counts.get(best[3], 0) + 1

    # Do not force one sentence from every ranked candidate. Ranking has already
    # selected the approved candidate set; compression is allowed to remove a
    # low-value candidate entirely when none of its local units is relevant
    # enough to justify prompt space. If everything was filtered out, retain
    # exactly one best unit from the highest-ranked approved candidate so a
    # verified answer path cannot collapse to an empty prompt.
    if not selected and candidates:
        top_candidate = candidates[0]
        top_units = _split_units(_text(top_candidate))
        if top_units:
            best_unit = max(
                top_units,
                key=lambda u: _unit_relevance(normalized_query, u, weights),
            )
            best_score = _unit_relevance(normalized_query, best_unit, weights)
            if best_score > 0.0:
                selected.append(
                    (best_score, 0, _candidate_id(top_candidate, 0), _source(top_candidate), best_unit, 0, tuple(top_units))
                )


    # Final deterministic ordering follows relevance while preserving the
    # upstream rank as a tie-breaker.
    selected.sort(key=lambda row: (-row[0], row[1], row[3], row[4]))

    rendered: list[str] = []
    items: list[ContextItem] = []
    total_chars = 0

    for score, candidate_index, candidate_id, source, unit, local_index, candidate_units in selected:
        # The source is retained in machine-readable metadata, not inserted
        # into the model-facing context. This prevents internal paths/IDs from
        # leaking into the answer prompt.
        clean_unit = _contextualize_selected_unit(unit, candidate_units, local_index)
        if not clean_unit:
            continue

        separator = "\n\n" if rendered else ""
        if total_chars + len(separator) + len(clean_unit) > max_context_chars:
            continue

        rendered.append(clean_unit)
        items.append(
            ContextItem(
                candidate_id=candidate_id,
                source=source,
                text=clean_unit,
                score=max(0.0, min(1.0, score)),
            )
        )
        total_chars += len(separator) + len(clean_unit)

    context = "\n\n".join(rendered).strip()
    reasons = [
        "ranking_is_authoritative",
        "compression_does_not_add_candidates",
        "duplicate_units_removed",
        "query_relevance_filter_applied",
        "diversity_selection_applied",
        "hard_context_budget_applied",
    ]
    if len(candidates) > len({item.candidate_id for item in items}):
        reasons.append("some_ranked_candidates_were_compressed_or_omitted")

    return FinalContextResult(
        query=normalized_query,
        context=context,
        items=tuple(items),
        input_candidates=len(candidates),
        selected_items=len(items),
        omitted_items=max(0, len(all_units) - len(items)),
        context_chars=len(context),
        context_tokens_estimate=max(0, len(context) // 4),
        reasons=tuple(reasons),
    )


def build_final_context_package(
    query: str,
    ranked_candidates: Sequence[Any],
    *,
    max_candidates: int = 8,
    max_units: int = 18,
    max_context_chars: int = 12_000,
) -> Any:
    """Create the existing EvidencePackage *only as a transport object*.

    The old evidence/claim/coverage machinery is deliberately not invoked.
    ``EvidencePackage`` remains useful because the answer generator already
    consumes that stable transport contract.
    """
    result = build_final_context(
        query,
        ranked_candidates,
        max_candidates=max_candidates,
        max_units=max_units,
        max_context_chars=max_context_chars,
    )

    from ai_platform.core.evidence.packaging import (
        EvidencePackage,
        PackagedEvidenceItem,
    )

    package_items = tuple(
        PackagedEvidenceItem(
            evidence_id=item.candidate_id,
            text=item.text,
            source=item.source,
        )
        for item in result.items
    )

    return EvidencePackage(
        status="ready" if result.ready else "empty",
        context=result.context,
        items=package_items,
        source_count=len({item.source for item in result.items}),
        selected_count=len(package_items),
        omitted_count=result.omitted_items,
        reasons=result.reasons,
    )


__all__ = [
    "ContextItem",
    "FinalContextResult",
    "build_final_context",
    "build_final_context_package",
]
