"""IIT Jodhpur source-category policy.

This is a weak ranking prior only. It never selects evidence and never
overrides verification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .profile import DATA_ROOT, INSTITUTION_ID

POLICY_PATH = DATA_ROOT / "source_policy.json"


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).casefold().replace("\\", "/")


@dataclass(frozen=True)
class SourceCategory:
    name: str
    path_markers: tuple[str, ...]
    base_weight: float


@dataclass(frozen=True)
class IITJSourcePolicy:
    institution_id: str
    categories: tuple[SourceCategory, ...]
    preference_by_request: Mapping[str, tuple[tuple[str, float], ...]]

    def classify_source(self, source: str) -> tuple[str, ...]:
        value = _norm(source)
        return tuple(
            category.name
            for category in self.categories
            if any(marker in value for marker in category.path_markers)
        )

    def preferred_categories(self, request_type: str) -> tuple[tuple[str, float], ...]:
        return self.preference_by_request.get(_norm(request_type), ())

    def source_alignment(self, request_type: str, source: str) -> float:
        categories = self.classify_source(source)
        preferences = dict(self.preferred_categories(request_type))
        return max(
            (min(1.0, max(0.0, float(preferences.get(c, 0.0)))) for c in categories),
            default=0.0,
        )


def _load() -> IITJSourcePolicy:
    with POLICY_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    categories = []
    for name, config in payload.get("categories", {}).items():
        markers = tuple(_norm(x) for x in config.get("path_markers", ()) if _norm(x))
        categories.append(
            SourceCategory(
                name=str(name),
                path_markers=markers,
                base_weight=min(1.0, max(0.0, float(config.get("rank_weight", 0.0)))),
            )
        )

    preferences = {}
    for request, values in payload.get("preference_by_request", {}).items():
        preferences[_norm(request)] = tuple(
            (str(category), min(1.0, max(0.0, float(weight))))
            for category, weight in values.items()
        )

    policy = IITJSourcePolicy(INSTITUTION_ID, tuple(categories), preferences)
    validate_source_policy(policy)
    return policy


@lru_cache(maxsize=1)
def load_source_policy() -> IITJSourcePolicy:
    return _load()


def get_source_policy() -> IITJSourcePolicy:
    return load_source_policy()


def validate_source_policy(policy: IITJSourcePolicy | None = None) -> None:
    value = policy or load_source_policy()
    if value.institution_id != INSTITUTION_ID or not value.categories:
        raise ValueError("Invalid IITJ source policy")
    for category in value.categories:
        if not category.name or not category.path_markers:
            raise ValueError("Every source category needs a name and path marker")


SOURCE_POLICY = load_source_policy()

__all__ = [
    "SourceCategory", "IITJSourcePolicy", "POLICY_PATH",
    "SOURCE_POLICY", "load_source_policy", "get_source_policy",
    "validate_source_policy",
]
