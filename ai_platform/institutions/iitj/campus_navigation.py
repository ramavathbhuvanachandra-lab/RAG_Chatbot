"""
IIT Jodhpur campus navigation.

Institution-specific implementation:
- IITJ campus locations
- IITJ location aliases
- IITJ location data source
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DATA_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "campus_locations.json"
)


def load_locations() -> list[dict[str, Any]]:
    """
    Load IIT Jodhpur campus locations.

    Returns an empty list if the institution data file cannot be read.
    """
    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            return []

        return data

    except Exception as exc:
        print(
            f"[IITJ Campus Navigation] "
            f"Error loading locations: {exc}"
        )
        return []


def find_location(query: str) -> dict[str, Any] | None:
    """
    Find an IIT Jodhpur campus location by name or alias.

    Matching is case-insensitive and uses whole-word/phrase matching.

    Examples:
        Where is the library?
        How do I reach the Central Mess?
    """
    normalized_query = str(query or "").strip().lower()

    if not normalized_query:
        return None

    best_match = None
    best_match_len = 0

    for location in load_locations():
        name = str(location.get("name", "")).strip().lower()
        if name:
            pattern = r"\b" + re.escape(name) + r"\b"
            if re.search(pattern, normalized_query):
                match_len = len(name)
                if match_len > best_match_len:
                    best_match = location
                    best_match_len = match_len

        aliases = location.get("aliases", [])
        if not isinstance(aliases, list):
            continue

        for alias in aliases:
            alias_text = str(alias or "").strip().lower()
            if not alias_text:
                continue

            pattern = r"\b" + re.escape(alias_text) + r"\b"
            if re.search(pattern, normalized_query):
                match_len = len(alias_text)
                if match_len > best_match_len:
                    best_match = location
                    best_match_len = match_len

    return best_match