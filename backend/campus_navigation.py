"""
Compatibility wrapper for IIT Jodhpur campus navigation.

The actual IIT Jodhpur implementation lives under:

    backend.institutions.iitj.campus_navigation

Existing application imports can continue using:

    from backend.campus_navigation import find_location
"""

from backend.institutions.iitj.campus_navigation import (
    find_location,
    load_locations,
)

__all__ = [
    "find_location",
    "load_locations",
]