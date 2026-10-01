from __future__ import annotations

from backend.emergency import (
    find_emergency as compatibility_find_emergency,
)
from backend.institutions.iitj.emergency import (
    DATA_FILE,
    find_emergency,
    load_emergency_contacts,
)


def test_iitj_emergency_data_exists() -> None:
    assert DATA_FILE.exists(), (
        f"IITJ emergency data missing: {DATA_FILE}"
    )


def test_iitj_emergency_contacts_load() -> None:
    contacts = load_emergency_contacts()

    assert contacts, (
        "IITJ emergency-contact data is empty."
    )

    assert all(
        isinstance(contact, dict)
        for contact in contacts
    )


def test_find_emergency_by_real_contact_name() -> None:
    contacts = load_emergency_contacts()

    contact = next(
        (
            item
            for item in contacts
            if str(item.get("name", "")).strip()
        ),
        None,
    )

    assert contact is not None

    name = str(
        contact["name"]
    ).strip()

    result = find_emergency(
        f"Please give me the {name} emergency contact."
    )

    assert result is not None
    assert result.get("name") == name


def test_find_emergency_by_real_alias_when_available() -> None:
    contacts = load_emergency_contacts()

    contact = next(
        (
            item
            for item in contacts
            if isinstance(item.get("aliases"), list)
            and any(
                str(alias).strip()
                for alias in item.get("aliases", [])
            )
        ),
        None,
    )

    if contact is None:
        return

    alias = next(
        str(alias).strip()
        for alias in contact.get("aliases", [])
        if str(alias).strip()
    )

    result = find_emergency(
        f"I need the {alias} number."
    )

    assert result is not None
    assert result.get("name") == contact.get("name")


def test_compatibility_wrapper_matches_iitj_implementation() -> None:
    contacts = load_emergency_contacts()

    contact = next(
        (
            item
            for item in contacts
            if str(item.get("name", "")).strip()
        ),
        None,
    )

    assert contact is not None

    query = f"Emergency contact for {contact['name']}"

    direct_result = find_emergency(query)
    wrapper_result = compatibility_find_emergency(query)

    assert wrapper_result == direct_result


def test_unknown_emergency_query_returns_none() -> None:
    result = find_emergency(
        "completely nonexistent emergency department xyz"
    )

    assert result is None


if __name__ == "__main__":
    print(
        "IITJ emergency navigation regression tests passed."
    )
