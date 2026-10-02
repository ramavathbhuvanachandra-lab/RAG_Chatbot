"""Institution-aware deterministic lexical recall for the reusable RAG core.

The generic engine owns retrieval mechanics. The active institution adapter owns
institution-specific vocabulary. The adapter is discovered dynamically through
INSTITUTION_ID so the core contains no IITJ-specific words or rules.

Contract:
    dense retrieval + BM25 + this lexical lane -> fusion -> verification

This module is recall-only. It must never verify a candidate or generate an
answer.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import re
from typing import Any, Iterable, Mapping, Sequence


_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "for", "in", "on", "at", "by", "with", "and", "or", "but",
        "if", "then", "than", "as", "from", "into", "about", "this", "that",
        "these", "those", "it", "its", "they", "their", "them", "there", "here",
        "what", "which", "how", "why", "when", "where", "who", "whom", "whose",
        "do", "does", "did", "can", "could", "would", "should", "may", "might",
        "will", "shall", "i", "me", "my", "you", "your", "we", "our", "us",
        "please", "tell", "give", "show", "get", "available", "students",
    }
)

_EQUIVALENCE_GROUPS = (
    frozenset({"program", "programs", "programme", "programmes"}),
    frozenset({"application", "applications", "apply", "applying", "applied", "applies"}),
    frozenset({"admission", "admissions", "admit"}),
    frozenset({"requirement", "requirements", "criterion", "criteria"}),
    frozenset({"department", "departments"}),
    frozenset({"fee", "fees", "tuition", "cost", "costs", "charge", "charges", "amount", "price", "prices"}),
    frozenset({"location", "located", "address"}),
    frozenset({"direction", "directions", "route", "routes", "reach", "reached", "reaching"}),
    frozenset({"research", "researches"}),
    frozenset({"facility", "facilities"}),
    frozenset({"hostel", "hostels"}),
    frozenset({"club", "clubs"}),
    frozenset({"eligibility", "eligible"}),
    frozenset({"phd", "doctoral", "doctorate"}),
    frozenset({"mtech", "m.tech"}),
    frozenset({"msc", "m.sc"}),
    frozenset({"btech", "b.tech"}),
)


@dataclass(frozen=True, slots=True)
class LexicalHit:
    document: Any
    score: float
    query_overlap: float
    target_overlap: float
    facet_overlap: float
    phrase_hit: bool
    matched_terms: tuple[str, ...] = ()
    institution_id: str = ""
    institution_score: float = 0.0
    institution_terms: tuple[str, ...] = ()
    institution_phrases: tuple[str, ...] = ()
    retrieval_queries: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(float(self.score), 6),
            "query_overlap": round(float(self.query_overlap), 6),
            "target_overlap": round(float(self.target_overlap), 6),
            "facet_overlap": round(float(self.facet_overlap), 6),
            "phrase_hit": self.phrase_hit,
            "matched_terms": list(self.matched_terms),
            "institution_id": self.institution_id,
            "institution_score": round(float(self.institution_score), 6),
            "institution_terms": list(self.institution_terms),
            "institution_phrases": list(self.institution_phrases),
            "retrieval_queries": list(self.retrieval_queries),
        }


def _normalize(value: Any) -> str:
    text = str(value or "").casefold()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\u0900-\u097f\s]", " ", text)
    return " ".join(text.split())


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", _normalize(value))


def _variants(token: str) -> set[str]:
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
    for group in _EQUIVALENCE_GROUPS:
        if value in group:
            out.update(group)
            break
    compact = _compact(value)
    if compact:
        out.add(compact)
    return out


def _tokens(value: Any, *, stopwords: bool = False) -> tuple[str, ...]:
    normalized = _normalize(value)
    if not normalized:
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for token in normalized.split():
        if stopwords and (token in _STOPWORDS or len(token) < 2):
            continue
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return tuple(result)


def _term_hit(term: str, document_tokens: set[str]) -> bool:
    forms = _variants(term)
    if any(form in document_tokens for form in forms):
        return True
    compact_doc = {_compact(token) for token in document_tokens}
    return any(form in compact_doc for form in forms)


def _overlap(terms: Sequence[str], document_tokens: set[str]) -> tuple[float, tuple[str, ...]]:
    terms = tuple(dict.fromkeys(terms))
    if not terms:
        return 0.0, ()
    matched = [term for term in terms if _term_hit(term, document_tokens)]
    return len(matched) / len(terms), tuple(matched)


def _text(document: Any) -> str:
    if isinstance(document, dict):
        content = str(document.get("page_content") or "")
        metadata = document.get("metadata") or {}
    else:
        content = str(getattr(document, "page_content", "") or "")
        metadata = getattr(document, "metadata", {}) or {}

    if not isinstance(metadata, dict):
        metadata = {}

    structural = [
        metadata.get("title"),
        metadata.get("heading"),
        metadata.get("section"),
        metadata.get("name"),
    ]
    labels = [str(value).strip() for value in structural if str(value or "").strip()]
    return " ".join((*labels, content)).strip()


def _target(frame: Any) -> str:
    target = getattr(frame, "target", "")
    if hasattr(target, "text"):
        target = getattr(target, "text", "")
    return str(target or "")


def _facets(frame: Any) -> tuple[str, ...]:
    values: list[str] = []
    for facet in getattr(frame, "facets", ()) or ():
        value = getattr(facet, "value", facet)
        text = str(value or "").strip()
        if text:
            values.append(text)
    return tuple(dict.fromkeys(values))


def _interesting_query_tokens(query: str) -> tuple[str, ...]:
    return _tokens(query, stopwords=True)


def _active_institution_id() -> str:
    configured = os.getenv("INSTITUTION_ID", "").strip().casefold()
    if configured:
        return configured
    try:
        module = importlib.import_module("backend.config")
        configured = str(getattr(module, "INSTITUTION_ID", "") or "").strip().casefold()
    except Exception:
        configured = ""
    return configured


def _load_institution_adapter() -> tuple[str, Any | None]:
    """Load active institution lexical adapter without naming any institution."""
    institution_id = _active_institution_id()
    if not institution_id:
        return "", None
    try:
        module = importlib.import_module(
            f"ai_platform.institutions.{institution_id}.lexical"
        )
    except Exception:
        return institution_id, None
    return institution_id, module


def _adapter_score(adapter: Any | None, query: str, document_text: str) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    if adapter is None:
        return 0.0, (), ()
    scorer = getattr(adapter, "score_pair", None)
    if not callable(scorer):
        return 0.0, (), ()
    try:
        result = scorer(query, document_text)
    except Exception:
        return 0.0, (), ()
    score = float(getattr(result, "query_overlap", 0.0) or 0.0)
    terms = tuple(str(value) for value in (getattr(result, "canonical_terms", ()) or ()) if str(value).strip())
    phrases = tuple(str(value) for value in (getattr(result, "matched_phrases", ()) or ()) if str(value).strip())
    return max(0.0, min(1.0, score)), terms, phrases


def _adapter_queries(adapter: Any | None, query: str) -> tuple[str, ...]:
    if adapter is None:
        return (query,)
    expander = getattr(adapter, "expand_query", None)
    if not callable(expander):
        return (query,)
    try:
        values = tuple(str(value).strip() for value in expander(query) if str(value).strip())
    except Exception:
        return (query,)
    values = tuple(dict.fromkeys(values))
    if not values or values[0] != query:
        values = (query, *values)
    return values[:3]


def _score_generic(
    query: str,
    document: Any,
    *,
    query_frame: Any | None = None,
) -> tuple[float, float, float, float, bool, tuple[str, ...]]:
    text = _text(document)
    normalized_query = _normalize(query)
    normalized_text = _normalize(text)
    document_tokens = set(_tokens(normalized_text))

    query_tokens = _interesting_query_tokens(query)
    query_overlap, query_matches = _overlap(query_tokens, document_tokens)

    target = _target(query_frame) if query_frame is not None else ""
    target_tokens = _tokens(target, stopwords=True)
    target_overlap, target_matches = _overlap(target_tokens, document_tokens)

    facet_scores: list[float] = []
    facet_matches: list[str] = []
    for facet in _facets(query_frame) if query_frame is not None else ():
        tokens = _tokens(facet, stopwords=True)
        score, matches = _overlap(tokens, document_tokens)
        facet_text = _normalize(facet)
        facet_pattern = rf"(?<![a-z0-9]){re.escape(facet_text)}(?![a-z0-9])" if facet_text else ""
        if facet_text and re.search(facet_pattern, normalized_text):
            score = 1.0
        facet_scores.append(score)
        facet_matches.extend(matches)
    facet_overlap = sum(facet_scores) / len(facet_scores) if facet_scores else 0.0

    phrase_pattern = (
        rf"(?<![a-z0-9]){re.escape(normalized_query)}(?![a-z0-9])"
        if normalized_query
        else ""
    )
    phrase_hit = bool(
        phrase_pattern
        and re.search(phrase_pattern, normalized_text)
    )

    if target_overlap and facet_overlap:
        score = 0.45 * target_overlap + 0.35 * facet_overlap + 0.20 * query_overlap
    elif target_overlap:
        score = 0.60 * target_overlap + 0.40 * query_overlap
    elif facet_overlap:
        score = 0.55 * facet_overlap + 0.45 * query_overlap
    else:
        score = query_overlap

    if phrase_hit:
        score = min(1.0, score + 0.25)

    if target and _compact(target) and _compact(target) in _compact(normalized_text):
        score = min(1.0, score + 0.15)

    matched_terms = tuple(dict.fromkeys((*query_matches, *target_matches, *facet_matches)))
    return min(1.0, max(0.0, score)), query_overlap, target_overlap, facet_overlap, phrase_hit, matched_terms


def score_document(
    query: str,
    document: Any,
    *,
    query_frame: Any | None = None,
) -> LexicalHit:
    """Score a document with generic lexical recall plus optional institution vocabulary."""
    institution_id, adapter = _load_institution_adapter()
    retrieval_queries = _adapter_queries(adapter, query)

    best = None
    best_key = None
    for variant in retrieval_queries:
        result = _score_generic(variant, document, query_frame=query_frame)
        key = result[0], result[4], result[2], result[3], result[1]
        if best is None or key > best_key:
            best = result
            best_key = key

    assert best is not None
    base_score, query_overlap, target_overlap, facet_overlap, phrase_hit, generic_matches = best

    institution_score, institution_terms, institution_phrases = _adapter_score(
        adapter,
        query,
        _text(document),
    )

    # Institution knowledge enriches recall; it does not replace the generic
    # lexical signal. A bounded 25% contribution is enough to rescue aliases
    # without letting institution metadata dominate retrieval.
    enriched_score = min(
        1.0,
        0.70 * base_score
        + 0.25 * institution_score
        + 0.05 * (1.0 if institution_phrases else 0.0),
    )

    matched_terms = tuple(
        dict.fromkeys(
            (
                *generic_matches,
                *institution_terms,
                *institution_phrases,
            )
        )
    )

    return LexicalHit(
        document=document,
        score=enriched_score,
        query_overlap=query_overlap,
        target_overlap=target_overlap,
        facet_overlap=facet_overlap,
        phrase_hit=phrase_hit,
        matched_terms=matched_terms,
        institution_id=institution_id,
        institution_score=institution_score,
        institution_terms=institution_terms,
        institution_phrases=institution_phrases,
        retrieval_queries=retrieval_queries,
    )


def retrieve_lexical(
    query: str,
    documents: Iterable[Any],
    *,
    query_frame: Any | None = None,
    limit: int = 40,
    min_score: float = 0.30,
) -> tuple[LexicalHit, ...]:
    """Return strongest deterministic lexical hits in stable order."""
    query = str(query or "").strip()
    if not query or limit < 1:
        return ()

    hits: list[tuple[LexicalHit, int]] = []
    seen: set[str] = set()

    for index, document in enumerate(documents or ()):
        hit = score_document(query, document, query_frame=query_frame)
        if hit.score < float(min_score):
            continue

        metadata = getattr(document, "metadata", {}) or {}
        if isinstance(document, dict):
            metadata = document.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        source = str(
            metadata.get("source")
            or metadata.get("source_path")
            or metadata.get("path")
            or ""
        )
        identity = f"{source.casefold()}\n{_normalize(_text(document))}"
        if identity in seen:
            continue
        seen.add(identity)
        hits.append((hit, index))

    hits.sort(
        key=lambda item: (
            item[0].score,
            item[0].institution_phrases,
            item[0].phrase_hit,
            item[0].target_overlap,
            item[0].facet_overlap,
            item[0].query_overlap,
            -item[1],
        ),
        reverse=True,
    )
    return tuple(hit for hit, _ in hits[:limit])


__all__ = ["LexicalHit", "score_document", "retrieve_lexical"]