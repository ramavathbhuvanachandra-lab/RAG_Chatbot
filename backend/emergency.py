import json
from pathlib import Path


# Path to data/emergency_contacts.json
DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "emergency_contacts.json"
)


def load_emergency_contacts():
    """
    Load all emergency contacts from JSON.
    """
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Emergency] Error loading contacts: {e}")
        return []


def find_emergency(query: str):
    """
    Find an emergency contact by name or alias.
    Supports both exact match and natural language questions.
    """

    query = query.strip().lower()

    contacts = load_emergency_contacts()

    for contact in contacts:

        # Match contact name
        name = contact.get("name", "").lower()
        if name and name in query:
            return contact

        # Match aliases
        for alias in contact.get("aliases", []):
            alias = alias.lower()
            if alias in query:
                return contact

    return None