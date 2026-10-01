"""
Compatibility wrapper for IIT Jodhpur emergency contacts.

The actual IIT Jodhpur implementation lives under:

    backend.institutions.iitj.emergency

Existing application imports can continue using:

    from backend.emergency import find_emergency
"""

from backend.institutions.iitj.emergency import (
    find_emergency,
    load_emergency_contacts,
)

__all__ = [
    "find_emergency",
    "load_emergency_contacts",
]
