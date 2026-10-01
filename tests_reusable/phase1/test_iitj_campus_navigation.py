from __future__ import annotations

import json
from pathlib import Path

from backend.campus_navigation import (
    find_location as compatibility_find_location,
)
from backend.institutions.iitj.campus_navigation import (
    DATA_FILE,
    find_location,
    load_locations,
)


def test_iitj_location_data_exists() -> None:
    assert DATA_FILE.exists(), (
        f"IITJ campus data missing: {DATA_FILE}"
    )


def test_iitj_locations_load() -> None:
    locations = load_locations()

    assert locations, "IITJ campus location data is empty."

    assert all(
        isinstance(location, dict)
        for location in locations
    )


def test_find_location_by_real_location_name() -> None:
    locations = load_locations()

    location = next(
        (
            item
            for item in locations
            if str(item.get("name", "")).strip()
        ),
        None,
    )

    assert location is not None

    name = str(location["name"]).strip()

    result = find_location(
        f"Where is the {name}?"
    )

    assert result is not None
    assert result.get("name") == name


def test_find_location_by_real_alias_when_available() -> None:
    locations = load_locations()

    location = next(
        (
            item
            for item in locations
            if isinstance(item.get("aliases"), list)
            and any(
                str(alias).strip()
                for alias in item.get("aliases", [])
            )
        ),
        None,
    )

    if location is None:
        return

    alias = next(
        str(alias).strip()
        for alias in location.get("aliases", [])
        if str(alias).strip()
    )

    result = find_location(
        f"Where is the {alias}?"
    )

    assert result is not None
    assert result.get("name") == location.get("name")


def test_compatibility_wrapper_matches_iitj_implementation() -> None:
    locations = load_locations()

    location = next(
        (
            item
            for item in locations
            if str(item.get("name", "")).strip()
        ),
        None,
    )

    assert location is not None

    query = f"Where is {location['name']}?"

    direct_result = find_location(query)
    wrapper_result = compatibility_find_location(query)

    assert wrapper_result == direct_result


if __name__ == "__main__":
    print("IITJ campus navigation regression tests passed.")