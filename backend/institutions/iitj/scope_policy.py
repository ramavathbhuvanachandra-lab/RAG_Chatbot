"""IIT Jodhpur claim-scope knowledge.

The generic core owns scope-conflict evaluation. This module only provides
institution-specific vocabulary and mutually exclusive dimensions.

The policy is deliberately conservative: it defines explicit scopes rather
than guessing hidden user attributes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .profile import DATA_ROOT, INSTITUTION_ID

POLICY_PATH = DATA_ROOT / "scope_policy.json"


@dataclass(frozen=True)
class ScopeDimension:
    name: str
    values: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class IITJScopePolicy:
    institution_id: str
    dimensions: tuple[ScopeDimension, ...]

    def detect(self, text: str) -> dict[str, frozenset[str]]:
        normalized = " ".join(str(text or "").casefold().split())
        result = {}
        for dimension in self.dimensions:
            found = set()
            for canonical, variants in dimension.values.items():
                terms = (canonical, *variants)
                if any(_contains(normalized, term) for term in terms):
                    found.add(canonical)
            if found:
                result[dimension.name] = frozenset(found)
        return result

    def conflicting_dimensions(
        self,
        query_scopes: Mapping[str, frozenset[str]],
        evidence_scopes: Mapping[str, frozenset[str]],
    ) -> tuple[str, ...]:
        conflicts = []
        for dimension in self.dimensions:
            q = set(query_scopes.get(dimension.name, ()))
            e = set(evidence_scopes.get(dimension.name, ()))
            if q and e and q.isdisjoint(e):
                conflicts.append(dimension.name)
        return tuple(conflicts)


def _contains(text: str, term: str) -> bool:
    term = " ".join(str(term or "").casefold().split())
    if not term:
        return False
    import re
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))


@lru_cache(maxsize=1)
def load_scope_policy() -> IITJScopePolicy:
    with POLICY_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    dimensions = []
    for name, values in payload.get("dimensions", {}).items():
        cleaned = {}
        for canonical, variants in values.items():
            vals = tuple(str(x).strip() for x in variants if str(x).strip())
            if vals:
                cleaned[str(canonical)] = vals
        dimensions.append(ScopeDimension(str(name), cleaned))

    policy = IITJScopePolicy(INSTITUTION_ID, tuple(dimensions))
    validate_scope_policy(policy)
    return policy


def get_scope_policy() -> IITJScopePolicy:
    return load_scope_policy()


def validate_scope_policy(policy: IITJScopePolicy | None = None) -> None:
    value = policy or load_scope_policy()
    if value.institution_id != INSTITUTION_ID or not value.dimensions:
        raise ValueError("Invalid IITJ scope policy")
    for dimension in value.dimensions:
        if not dimension.name or not dimension.values:
            raise ValueError("Every scope dimension needs values")


SCOPE_POLICY = load_scope_policy()

__all__ = [
    "ScopeDimension", "IITJScopePolicy", "SCOPE_POLICY",
    "POLICY_PATH", "load_scope_policy", "get_scope_policy",
    "validate_scope_policy",
]
