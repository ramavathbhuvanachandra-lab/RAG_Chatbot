import json
from pathlib import Path
import re


# Path to data/campus_locations.json
DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "campus_locations.json"
)


def load_locations():
    """Load all campus locations from JSON."""
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Campus Navigation] Error loading locations: {e}")
        return []

def find_location(query: str):
    """
    Find a campus location by name or alias.
    Supports both exact match and questions like:
    - Where is the library?
    - How do I reach the Central Mess?
    """

    query = query.strip().lower()

    locations = load_locations()

    for location in locations:

        # Match location name as a whole word/phrase
        name = location.get("name", "").lower().strip()
        if name:
            pattern = r"\b" + re.escape(name) + r"\b"
            if re.search(pattern, query):
                return location

        # Match aliases as whole words/phrases
        for alias in location.get("aliases", []):
            alias = alias.lower().strip()

            pattern = r"\b" + re.escape(alias) + r"\b"

            if re.search(pattern, query):
                return location

    return None