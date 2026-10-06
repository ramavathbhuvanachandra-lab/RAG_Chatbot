"""IIT Jodhpur semantic registry.

The registry is institution knowledge, not retrieval logic.

It provides deterministic detection/canonicalization for:
- programs
- topics
- entities
- attributes
- qualifiers
- institution identity
- scope labels

It never retrieves, ranks, verifies, cites, or generates answers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from ai_platform.core.retrieval.semantic_registry import SemanticRegistry

from .profile import DATA_ROOT, INSTITUTION_ID

ALIASES_PATH = DATA_ROOT / "aliases.json"
SEMANTIC_TERMS_PATH = DATA_ROOT / "semantic_terms.json"
SCOPE_POLICY_PATH = DATA_ROOT / "scope_policy.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing IITJ institution data file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _norm(value: Any) -> str:
    text = _clean(value).casefold()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _terms(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        cleaned = _clean(item)
        key = _norm(cleaned)
        if key and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return tuple(result)


def _mapping(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for canonical, variants in value.items():
        key = _clean(canonical)
        vals = _terms(variants)
        if key and vals:
            result[key] = vals
    return result


def _contains(text: str, term: str) -> bool:
    text_n = _norm(text)
    term_n = _norm(term)
    if not text_n or not term_n:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(term_n)}(?![a-z0-9])"
    return bool(re.search(pattern, text_n, flags=re.IGNORECASE))


class IITJSemanticRegistry(SemanticRegistry):
    """IITJ implementation of the shared SemanticRegistry contract."""

    __slots__ = ("_data",)

    def __init__(self) -> None:
        aliases = _load_json(ALIASES_PATH)
        semantic = _load_json(SEMANTIC_TERMS_PATH)
        scopes = _load_json(SCOPE_POLICY_PATH)

        object.__setattr__(
            self,
            "_data",
            {
                "institution_id": INSTITUTION_ID,
                "institution": _mapping(aliases.get("institution", {})),
                "programs": _mapping(aliases.get("programs", {})),
                "topics": _mapping(semantic.get("topics", {})),
                "entities": _mapping(semantic.get("entities", {})),
                "attributes": _mapping(semantic.get("attributes", {})),
                "qualifiers": _mapping(semantic.get("qualifiers", {})),
                "scopes": _mapping(scopes.get("dimensions", {})),
            },
        )
        self.validate()

    @property
    def institution_id(self) -> str:
        return self._data["institution_id"]

    @property
    def program_terms(self) -> Mapping[str, tuple[str, ...]]:
        return self._data["programs"]

    @property
    def topic_terms(self) -> Mapping[str, tuple[str, ...]]:
        return self._data["topics"]

    @property
    def entity_terms(self) -> Mapping[str, tuple[str, ...]]:
        return self._data["entities"]

    @property
    def attribute_terms(self) -> Mapping[str, tuple[str, ...]]:
        return self._data["attributes"]

    @property
    def qualifier_terms(self) -> Mapping[str, tuple[str, ...]]:
        return self._data["qualifiers"]

    @property
    def scope_terms(self) -> Mapping[str, tuple[str, ...]]:
        return self._data["scopes"]

    @staticmethod
    def _detect(text: str, mapping: Mapping[str, tuple[str, ...]]) -> set[str]:
        found: set[str] = set()
        for canonical, variants in mapping.items():
            if _contains(text, canonical) or any(_contains(text, v) for v in variants):
                found.add(canonical)
        return found

    def detect_programs(self, text: str) -> set[str]:
        return self._detect(text, self.program_terms)

    def detect_topics(self, text: str) -> set[str]:
        return self._detect(text, self.topic_terms)

    def detect_entities(self, text: str) -> set[str]:
        return self._detect(text, self.entity_terms)

    def detect_attributes(self, text: str) -> set[str]:
        return self._detect(text, self.attribute_terms)

    def detect_qualifiers(self, text: str) -> set[str]:
        return self._detect(text, self.qualifier_terms)

    def detect_institution(self, text: str) -> set[str]:
        return self._detect(text, self._data["institution"])

    def detect_scopes(self, text: str) -> dict[str, set[str]]:
        return {
            dimension: self._detect(text, terms)
            for dimension, terms in self.scope_terms.items()
        }

    def canonicalize(self, text: str) -> str:
        """Canonicalize only stable institution/program terminology."""
        result = _clean(text)
        if not result:
            return ""

        mappings: list[tuple[str, str]] = []
        for section in ("institution", "programs"):
            for canonical, variants in self._data[section].items():
                mappings.extend((variant, canonical) for variant in variants)

        mappings.sort(key=lambda pair: len(pair[0]), reverse=True)

        for variant, canonical in mappings:
            pattern = rf"(?<![a-z0-9]){re.escape(_clean(variant))}(?![a-z0-9])"
            result = re.sub(pattern, canonical, result, flags=re.IGNORECASE)

        return re.sub(r"\s+", " ", result).strip()

    def describe(self, text: str) -> dict[str, Any]:
        return {
            "institution": tuple(sorted(self.detect_institution(text))),
            "programs": tuple(sorted(self.detect_programs(text))),
            "topics": tuple(sorted(self.detect_topics(text))),
            "entities": tuple(sorted(self.detect_entities(text))),
            "attributes": tuple(sorted(self.detect_attributes(text))),
            "qualifiers": tuple(sorted(self.detect_qualifiers(text))),
            "scopes": {
                key: tuple(sorted(value))
                for key, value in self.detect_scopes(text).items()
                if value
            },
        }

    def validate(self) -> None:
        if self.institution_id != INSTITUTION_ID:
            raise ValueError(f"Expected institution_id={INSTITUTION_ID!r}")

        for section in (
            "institution",
            "programs",
            "topics",
            "entities",
            "attributes",
            "qualifiers",
            "scopes",
        ):
            mapping = self._data[section]
            if not isinstance(mapping, Mapping):
                raise TypeError(f"{section} must be a mapping")
            for canonical, variants in mapping.items():
                if not _clean(canonical) or not variants:
                    raise ValueError(f"Invalid {section} entry: {canonical!r}")


SEMANTIC_REGISTRY = IITJSemanticRegistry()
IITJ_SEMANTIC_REGISTRY = SEMANTIC_REGISTRY


def get_semantic_registry() -> IITJSemanticRegistry:
    return SEMANTIC_REGISTRY


__all__ = [
    "IITJSemanticRegistry",
    "SEMANTIC_REGISTRY",
    "IITJ_SEMANTIC_REGISTRY",
    "get_semantic_registry",
]
