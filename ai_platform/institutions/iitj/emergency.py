"""
IIT Jodhpur emergency-contact lookup.

This module contains institution-specific emergency-contact behavior
and reads only IIT Jodhpur's structured emergency-contact data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATA_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "emergency_contacts.json"
)


def load_emergency_contacts() -> list[dict[str, Any]]:
    """
    Load IIT Jodhpur emergency contacts.

    Returns an empty list when the data file cannot be read or does not
    contain a JSON list.
    """
    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            return []

        return data

    except Exception as exc:
        print(
            f"[IITJ Emergency] "
            f"Error loading contacts: {exc}"
        )
        return []


def find_emergency(
    query: str,
) -> dict[str, Any] | None:
    """
    Find an IIT Jodhpur emergency contact by name or alias.

    Matching is case-insensitive.

    Returns:
        Matching contact dictionary, or None when no contact matches.
    """
    normalized_query = str(query or "").strip().lower()

    if not normalized_query:
        return None

    for contact in load_emergency_contacts():

        name = str(
            contact.get("name", "")
        ).strip().lower()

        if name and name in normalized_query:
            return contact

        aliases = contact.get("aliases", [])

        if not isinstance(aliases, list):
            continue

        for alias in aliases:

            alias_text = str(
                alias or ""
            ).strip().lower()

            if not alias_text:
                continue

            if alias_text in normalized_query:
                return contact

    return None


__all__ = [
    "DATA_FILE",
    "load_emergency_contacts",
    "find_emergency",
]
