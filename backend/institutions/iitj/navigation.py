"""IIT Jodhpur navigation concepts.

This file describes navigation vocabulary only. It intentionally contains no
hard-coded coordinates, building locations, routes, or answers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .profile import DATA_ROOT, INSTITUTION_ID

NAVIGATION_PATH = DATA_ROOT / "navigation.json"


@dataclass(frozen=True)
class NavigationModel:
    institution_id: str
    concepts: dict[str, tuple[str, ...]]

    def detect(self, text: str) -> tuple[str, ...]:
        value = " ".join(str(text or "").casefold().split())
        import re
        found = []
        for concept, terms in self.concepts.items():
            if any(re.search(rf"(?<![a-z0-9]){re.escape(t.casefold())}(?![a-z0-9])", value) for t in terms):
                found.append(concept)
        return tuple(sorted(found))


@lru_cache(maxsize=1)
def load_navigation_model() -> NavigationModel:
    with NAVIGATION_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    concepts = {
        str(k): tuple(str(v).strip() for v in values if str(v).strip())
        for k, values in payload.get("concepts", {}).items()
    }
    model = NavigationModel(INSTITUTION_ID, concepts)
    if not model.concepts:
        raise ValueError("IITJ navigation model cannot be empty")
    return model


NAVIGATION_MODEL = load_navigation_model()

__all__ = ["NavigationModel", "NAVIGATION_PATH", "NAVIGATION_MODEL", "load_navigation_model"]