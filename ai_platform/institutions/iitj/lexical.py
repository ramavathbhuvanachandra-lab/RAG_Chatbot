"""IIT Jodhpur lexical adapter.

This module supplies institution-specific lexical recall signals. It does not
decide whether evidence is true and it never bypasses generic verification.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .profile import DATA_ROOT, INSTITUTION_ID

ALIASES_PATH = DATA_ROOT / "aliases.json"
PHRASES_PATH = DATA_ROOT / "phrases.json"
ACRONYMS_PATH = DATA_ROOT / "acronyms.json"

CANONICALIZATION_SECTIONS = ("institution", "programs")

_PROTECTED_ABBREVIATIONS = (
    "M.S. by Research", "M.S. by research",
    "B.Tech.", "B.Tech", "M.Tech.", "M.Tech",
    "M.Sc.", "M.Sc", "Ph.D.", "Ph.D",
    "M.B.A.", "M.B.A", "B.S.", "B.S",
    "B.Sc.", "B.Sc",
)


@dataclass(frozen=True)
class LexicalProfile:
    institution_id: str
    aliases: Mapping[str, tuple[str, ...]]
    phrases: Mapping[str, tuple[str, ...]]
    acronyms: Mapping[str, tuple[str, ...]]
    canonicalization: tuple[str, ...]


@dataclass(frozen=True)
class LexicalMatch:
    canonical_terms: tuple[str, ...]
    exact_terms: tuple[str, ...]
    matched_phrases: tuple[str, ...]
    query_tokens: tuple[str, ...]
    document_tokens: tuple[str, ...]
    query_overlap: float


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing IITJ lexical file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _norm(value: Any) -> str:
    text = _clean(value).casefold()
    text = text.replace("–", "-").replace("—", "-").replace("_", " ")
    return re.sub(r"\s+", " ", text).strip(" \t\n\r,;:")


def _coerce_mapping(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for canonical, values in value.items():
        if isinstance(values, str):
            values = (values,)
        if not isinstance(values, (list, tuple, set, frozenset)):
            continue
        cleaned = []
        seen = set()
        for value in values:
            item = _clean(value)
            key = _norm(item)
            if key and key not in seen:
                seen.add(key)
                cleaned.append(item)
        if _clean(canonical) and cleaned:
            result[_clean(canonical)] = tuple(cleaned)
    return result


def _flatten_aliases(payload: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    result = {}
    for section in payload.values():
        if isinstance(section, Mapping):
            result.update(_coerce_mapping(section))
    return result


def _normalize_phrase_mapping(payload: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    result = {}
    for section, values in payload.items():
        if not isinstance(values, (list, tuple, set, frozenset)):
            continue
        items = []
        seen = set()
        for value in values:
            item = _clean(value)
            key = _norm(item)
            if key and key not in seen:
                seen.add(key)
                items.append(item)
        if items:
            result[_clean(section) or "general"] = tuple(items)
    return result


def _canonical_variant(value: str) -> str:
    value = _norm(value)
    value = re.sub(r"[^a-z0-9&+ ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _boundary_replace(text: str, variant: str, replacement: str) -> str:
    pieces = [re.escape(piece) for piece in _clean(variant).split()]
    if not pieces:
        return text
    joined = r"\s+".join(pieces)
    pattern = rf"(?<![a-z0-9]){joined}(?![a-z0-9])"
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)


def _normalize_for_matching(text: Any) -> str:
    value = _norm(text)
    if not value:
        return ""

    placeholders = {}
    protected = value
    for i, abbreviation in enumerate(sorted(_PROTECTED_ABBREVIATIONS, key=len, reverse=True)):
        placeholder = f" zzabbrev{i}zz "
        replaced = _boundary_replace(protected, abbreviation, placeholder)
        if replaced != protected:
            placeholders[placeholder.strip()] = _norm(abbreviation)
            protected = replaced

    protected = re.sub(r"[^a-z0-9&+ ]+", " ", protected)
    protected = re.sub(r"\s+", " ", protected).strip()

    for placeholder, abbreviation in placeholders.items():
        protected = protected.replace(
            placeholder,
            re.sub(r"[^a-z0-9&+ ]+", " ", abbreviation),
        )

    return re.sub(r"\s+", " ", protected).strip()


@lru_cache(maxsize=1)
def load_lexical_profile() -> LexicalProfile:
    aliases_payload = _load_json(ALIASES_PATH)
    profile = LexicalProfile(
        institution_id=INSTITUTION_ID,
        aliases=_flatten_aliases(aliases_payload),
        phrases=_normalize_phrase_mapping(_load_json(PHRASES_PATH)),
        acronyms=_coerce_mapping(_load_json(ACRONYMS_PATH)),
        canonicalization=tuple(
            key
            for section in CANONICALIZATION_SECTIONS
            for key in (
                aliases_payload.get(section, {}).keys()
                if isinstance(aliases_payload.get(section, {}), Mapping)
                else ()
            )
            if _clean(key)
        ),
    )
    validate_lexical_profile(profile)
    return profile


def get_lexical_profile() -> LexicalProfile:
    return load_lexical_profile()


def _canonicalization_pairs(profile: LexicalProfile) -> tuple[tuple[str, str], ...]:
    pairs = []
    for canonical in profile.canonicalization:
        for variant in profile.aliases.get(canonical, ()):
            pairs.append((_canonical_variant(variant), canonical))
        for variant in profile.acronyms.get(canonical, ()):
            pairs.append((_canonical_variant(variant), canonical))
    return tuple(sorted((p for p in pairs if p[0]), key=lambda x: len(x[0]), reverse=True))


def normalize_for_institution(text: str) -> str:
    result = _normalize_for_matching(text)
    if not result:
        return ""
    profile = get_lexical_profile()
    for variant, canonical in _canonicalization_pairs(profile):
        result = _boundary_replace(result, variant, canonical)
    return re.sub(r"\s+", " ", result).strip()


def canonical_terms(text: str) -> tuple[str, ...]:
    normalized = normalize_for_institution(text)
    if not normalized:
        return ()

    profile = get_lexical_profile()
    found = []
    for canonical in profile.aliases:
        canonical_normalized = _canonical_variant(canonical)
        if canonical_normalized and re.search(
            rf"(?<![a-z0-9]){re.escape(canonical_normalized)}(?![a-z0-9])",
            normalized,
            flags=re.IGNORECASE,
        ):
            found.append(canonical)
    return tuple(dict.fromkeys(found))


def matched_phrases(query: str, document: str) -> tuple[str, ...]:
    profile = get_lexical_profile()
    q = _normalize_for_matching(query)
    d = _normalize_for_matching(document)
    if not q or not d:
        return ()

    matches = []
    for phrases in profile.phrases.values():
        for phrase in phrases:
            p = _normalize_for_matching(phrase)
            if p and re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", q, re.I) and re.search(
                rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", d, re.I
            ):
                matches.append(phrase)
    return tuple(dict.fromkeys(matches))


def _tokens(text: str) -> tuple[str, ...]:
    normalized = normalize_for_institution(text)
    return tuple(token for token in normalized.split() if len(token) > 1)


def _interesting_tokens(text: str) -> tuple[str, ...]:
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "what", "which",
        "who", "how", "when", "where", "why", "can", "could", "would",
        "do", "does", "did", "for", "of", "to", "in", "at", "on", "and",
        "or", "my", "me", "i", "please", "tell", "give", "about",
    }
    return tuple(token for token in _tokens(text) if token not in stopwords)


def score_pair(query: str, document: str) -> LexicalMatch:
    q_tokens = _interesting_tokens(query)
    d_tokens = _interesting_tokens(document)
    q_terms = set(canonical_terms(query))
    d_terms = set(canonical_terms(document))

    q_set, d_set = set(q_tokens), set(d_tokens)
    overlap = len(q_set & d_set) / len(q_set) if q_set else 0.0

    exact_terms = tuple(sorted(q_terms & d_terms, key=str.casefold))
    phrases = matched_phrases(query, document)

    if q_terms:
        overlap = max(overlap, len(exact_terms) / len(q_terms))
    if phrases:
        overlap = max(overlap, min(1.0, 0.75 + 0.25 * min(1, len(phrases))))

    return LexicalMatch(
        canonical_terms=tuple(sorted(q_terms, key=str.casefold)),
        exact_terms=exact_terms,
        matched_phrases=phrases,
        query_tokens=q_tokens,
        document_tokens=d_tokens,
        query_overlap=max(0.0, min(1.0, overlap)),
    )


def expand_query(query: str) -> tuple[str, ...]:
    original = _clean(query)
    if not original:
        return ()

    values = [original]
    normalized = normalize_for_institution(original)
    if normalized.casefold() != original.casefold():
        values.append(normalized)

    terms = canonical_terms(original)
    if terms:
        concept_query = " ".join(terms)
        if concept_query and all(concept_query.casefold() != x.casefold() for x in values):
            values.append(concept_query)

    return tuple(dict.fromkeys(values))[:3]


def validate_lexical_profile(profile: LexicalProfile | None = None) -> None:
    value = profile or load_lexical_profile()
    if value.institution_id != INSTITUTION_ID:
        raise ValueError("Lexical profile belongs to another institution")
    for name, mapping in (("aliases", value.aliases), ("acronyms", value.acronyms)):
        if not isinstance(mapping, Mapping) or not mapping:
            raise ValueError(f"{name} must be a non-empty mapping")
        for canonical, variants in mapping.items():
            if not _clean(canonical) or not variants:
                raise ValueError(f"Invalid {name} entry: {canonical!r}")
    if not value.phrases or not value.canonicalization:
        raise ValueError("Incomplete lexical profile")


__all__ = [
    "LexicalProfile", "LexicalMatch", "load_lexical_profile",
    "get_lexical_profile", "normalize_for_institution", "canonical_terms",
    "matched_phrases", "score_pair", "expand_query", "validate_lexical_profile",
]
