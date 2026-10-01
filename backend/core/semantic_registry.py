"""
Institution-agnostic semantic vocabulary registry.

The reusable RAG core must never hardcode institution-specific programs,
departments, entities, or taxonomy terms.

This module defines the contract for supplying those vocabularies at runtime.

Example:

    registry = SemanticRegistry(
        programs={
            "program_a": ("alias one", "alias two"),
        },
        topics={
            "topic_a": ("term one", "term two"),
        },
        entities={
            "entity_a": ("name one", "name two"),
        },
    )

The core can then detect semantic signals without knowing which institution
created the registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


# ============================================================
# Normalization
# ============================================================


def normalize_text(text: object) -> str:
    """
    Normalize text for deterministic semantic matching.

    This is deliberately generic. It does not contain institution-specific
    abbreviations, program names, or domain vocabulary.
    """

    value = str(
        text or ""
    ).casefold()

    replacements = {
        "_": " ",
        "-": " ",
        "/": " ",
    }

    for old, new in replacements.items():
        value = value.replace(
            old,
            new,
        )

    # Collapse punctuation into spaces while preserving common
    # alphanumeric content.
    normalized_chars: list[str] = []

    for char in value:

        if char.isalnum() or char.isspace():
            normalized_chars.append(
                char
            )
        else:
            normalized_chars.append(
                " "
            )

    value = "".join(
        normalized_chars
    )

    return " ".join(
        value.split()
    )


def _clean_label(value: object) -> str:
    return " ".join(
        str(
            value or ""
        ).strip().split()
    )


def _clean_aliases(
    aliases: Iterable[object],
) -> tuple[str, ...]:
    """
    Normalize and deduplicate aliases while preserving order.
    """

    result: list[str] = []
    seen: set[str] = set()

    for alias in aliases:

        cleaned = _clean_label(
            alias
        )

        if not cleaned:
            continue

        normalized = normalize_text(
            cleaned
        )

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        result.append(
            cleaned
        )

    return tuple(
        result
    )


def _normalize_mapping(
    mapping: Mapping[object, Iterable[object]] | None,
) -> dict[str, tuple[str, ...]]:
    """
    Normalize a semantic category mapping.

    Keys are canonical semantic labels.
    Values are aliases that may appear in queries/documents.
    """

    if mapping is None:
        return {}

    if not isinstance(
        mapping,
        Mapping,
    ):
        raise TypeError(
            "Semantic registry categories must be mappings."
        )

    result: dict[str, tuple[str, ...]] = {}

    for raw_key, raw_aliases in mapping.items():

        key = _clean_label(
            raw_key
        )

        if not key:
            raise ValueError(
                "Semantic registry labels cannot be empty."
            )

        if raw_aliases is None:
            aliases: Iterable[object] = ()
        else:
            aliases = raw_aliases

        cleaned_aliases = _clean_aliases(
            (
                key,
                *aliases,
            )
        )

        if not cleaned_aliases:
            raise ValueError(
                f"Semantic registry entry '{key}' "
                "must contain at least one alias."
            )

        result[key] = cleaned_aliases

    return result


# ============================================================
# Semantic registry
# ============================================================


@dataclass(frozen=True, slots=True)
class SemanticRegistry:
    """
    Runtime semantic vocabulary for one deployment.

    Categories are intentionally generic:

        programs
        topics
        entities

    The actual vocabulary belongs to the institution/deployment layer.
    """

    programs: Mapping[str, Iterable[str]] = field(
        default_factory=dict
    )

    topics: Mapping[str, Iterable[str]] = field(
        default_factory=dict
    )

    entities: Mapping[str, Iterable[str]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        normalized_programs = _normalize_mapping(
            self.programs
        )

        normalized_topics = _normalize_mapping(
            self.topics
        )

        normalized_entities = _normalize_mapping(
            self.entities
        )

        object.__setattr__(
            self,
            "programs",
            normalized_programs,
        )

        object.__setattr__(
            self,
            "topics",
            normalized_topics,
        )

        object.__setattr__(
            self,
            "entities",
            normalized_entities,
        )

    # --------------------------------------------------------
    # Internal category lookup
    # --------------------------------------------------------

    @staticmethod
    def _detect_from_mapping(
        text: str,
        mapping: Mapping[str, tuple[str, ...]],
    ) -> set[str]:
        """
        Detect canonical labels whose aliases occur in text.

        Matching is phrase-aware rather than raw substring matching where
        possible, reducing accidental partial matches.
        """

        normalized_text = normalize_text(
            text
        )

        if not normalized_text:
            return set()

        text_tokens = set(
            normalized_text.split()
        )

        found: set[str] = set()

        for label, aliases in mapping.items():

            for alias in aliases:

                normalized_alias = normalize_text(
                    alias
                )

                if not normalized_alias:
                    continue

                alias_tokens = normalized_alias.split()

                if len(alias_tokens) == 1:

                    if alias_tokens[0] in text_tokens:
                        found.add(
                            label
                        )
                        break

                    continue

                padded_text = (
                    f" {normalized_text} "
                )

                padded_alias = (
                    f" {normalized_alias} "
                )

                if (
                    padded_alias
                    in padded_text
                ):
                    found.add(
                        label
                    )
                    break

        return found

    # --------------------------------------------------------
    # Public detection API
    # --------------------------------------------------------

    def detect_programs(
        self,
        text: str,
    ) -> set[str]:
        """Detect configured program labels."""

        return self._detect_from_mapping(
            text,
            self.programs,
        )

    def detect_topics(
        self,
        text: str,
    ) -> set[str]:
        """Detect configured topic labels."""

        return self._detect_from_mapping(
            text,
            self.topics,
        )

    def detect_entities(
        self,
        text: str,
    ) -> set[str]:
        """Detect configured entity labels."""

        return self._detect_from_mapping(
            text,
            self.entities,
        )

    # --------------------------------------------------------
    # Combined detection
    # --------------------------------------------------------

    def detect_all(
        self,
        text: str,
    ) -> dict[str, set[str]]:
        """
        Detect all configured semantic categories.
        """

        return {
            "programs": self.detect_programs(
                text
            ),
            "topics": self.detect_topics(
                text
            ),
            "entities": self.detect_entities(
                text
            ),
        }

    # --------------------------------------------------------
    # Alias inspection
    # --------------------------------------------------------

    def aliases_for(
        self,
        category: str,
        label: str,
    ) -> tuple[str, ...]:
        """
        Return aliases for one canonical label.
        """

        category_name = _clean_label(
            category
        ).casefold()

        label_name = _clean_label(
            label
        )

        categories = {
            "programs": self.programs,
            "topics": self.topics,
            "entities": self.entities,
        }

        if category_name not in categories:
            raise ValueError(
                "Unknown semantic category: "
                f"{category}"
            )

        return categories[
            category_name
        ].get(
            label_name,
            (),
        )

    # --------------------------------------------------------
    # Validation / diagnostics
    # --------------------------------------------------------

    def validate(self) -> "SemanticRegistry":
        """
        Validate the registry and return itself.

        This makes deployment startup validation explicit.
        """

        for category_name, mapping in (
            (
                "programs",
                self.programs,
            ),
            (
                "topics",
                self.topics,
            ),
            (
                "entities",
                self.entities,
            ),
        ):

            if not isinstance(
                mapping,
                Mapping,
            ):
                raise TypeError(
                    f"{category_name} must be a mapping."
                )

            for label, aliases in mapping.items():

                if not _clean_label(label):
                    raise ValueError(
                        f"{category_name} contains an empty label."
                    )

                if not aliases:
                    raise ValueError(
                        f"{category_name}.{label} "
                        "must contain at least one alias."
                    )

        return self

    @property
    def empty(self) -> bool:
        """Return True when no semantic vocabulary is configured."""

        return not (
            self.programs
            or self.topics
            or self.entities
        )


# ============================================================
# Public API
# ============================================================


__all__ = [
    "SemanticRegistry",
    "normalize_text",
]