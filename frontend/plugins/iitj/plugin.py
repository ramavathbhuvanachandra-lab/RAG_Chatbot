"""IIT Jodhpur UI Presentation Plugin.

Implements the institution UI plugin contract for IIT Jodhpur using verified
campus location data, emergency contacts, and customized branding.
"""

from __future__ import annotations

import json
from pathlib import Path

from frontend.core.models import (
    CampusLocationItem,
    EmergencyContactItem,
    InstitutionUIConfig,
    QuickActionItem,
)
from frontend.core.plugin_registry import InstitutionUIPlugin

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_iitj_campus_locations() -> tuple[CampusLocationItem, ...]:
    """Load IITJ campus location records from data/campus_locations.json."""
    loc_file = _REPO_ROOT / "data" / "campus_locations.json"
    if not loc_file.is_file():
        return ()

    try:
        with open(loc_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        items = []
        for d in data:
            if not isinstance(d, dict):
                continue
            items.append(
                CampusLocationItem(
                    id=str(d.get("id", "")),
                    name=str(d.get("name", "")),
                    category=str(d.get("category", "General")),
                    description=str(d.get("description", "")),
                    timings=str(d.get("timings", "")),
                    google_maps=str(d.get("google_maps", "")),
                    directions=str(d.get("directions", "")),
                    show_in_map=bool(d.get("show_in_map", False)),
                )
            )
        return tuple(items)
    except Exception:
        return ()


def _load_iitj_emergency_contacts() -> tuple[EmergencyContactItem, ...]:
    """Load IITJ emergency contacts from data/emergency_contacts.json."""
    contact_file = _REPO_ROOT / "data" / "emergency_contacts.json"
    if not contact_file.is_file():
        return ()

    try:
        with open(contact_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        items = []
        for d in data:
            if not isinstance(d, dict):
                continue
            items.append(
                EmergencyContactItem(
                    id=str(d.get("id", "")),
                    name=str(d.get("name", "")),
                    phone=str(d.get("phone", "")),
                    description=str(d.get("description", "")),
                    help_text=str(d.get("help_text", "")),
                )
            )
        return tuple(items)
    except Exception:
        return ()


class IITJUIPlugin(InstitutionUIPlugin):
    """IIT Jodhpur presentation plugin."""

    @property
    def institution_id(self) -> str:
        return "iitj"

    def get_config(self) -> InstitutionUIConfig:
        locations = _load_iitj_campus_locations()
        emergency_contacts = _load_iitj_emergency_contacts()

        quick_actions = (
            QuickActionItem(
                id="qa_departments",
                label="Departments",
                prompt="What departments does IIT Jodhpur have?",
                icon="🏛️",
            ),
            QuickActionItem(
                id="qa_btech_programmes",
                label="B.Tech Programmes",
                prompt="What B.Tech programmes does IIT Jodhpur offer?",
                icon="🎓",
            ),
            QuickActionItem(
                id="qa_mtech_eligibility",
                label="M.Tech Eligibility",
                prompt="What are the M.Tech admission eligibility requirements?",
                icon="📋",
            ),
            QuickActionItem(
                id="qa_research_areas",
                label="Research Areas",
                prompt="What are the main research areas at IIT Jodhpur?",
                icon="🔬",
            ),
            QuickActionItem(
                id="qa_tuition_fees",
                label="Tuition Fees",
                prompt="What is the tuition fee per semester at IIT Jodhpur?",
                icon="💳",
            ),
            QuickActionItem(
                id="qa_curriculum",
                label="Curriculum",
                prompt="Where can I find the B.Tech curriculum?",
                icon="📚",
            ),
        )

        supported_topics = (
            "Admissions & Reporting Formalities",
            "B.Tech & M.Tech Academic Guidelines",
            "Ph.D. Leave & Progress Monitoring",
            "Fee Structures & Financial Rules",
            "Hostel Allocation & Dining Rules",
            "Campus Facilities & IT Services",
            "Interactive Campus Navigation",
            "24/7 Security & Emergency Contacts",
        )

        welcome_message = (
            "The IIT Jodhpur AI Assistant helps you explore official institute information, "
            "academics, admissions, degree programmes, campus facilities, fees, research areas, "
            "and other available institutional information."
        )

        return InstitutionUIConfig(
            institution_id="iitj",
            display_name="IIT Jodhpur",
            tagline="Official Academic, Admissions & Campus AI Assistant",
            logo_icon="🎓",
            primary_color="#0F294A",
            accent_color="#2563EB",
            welcome_title="IIT Jodhpur AI Assistant",
            welcome_message=welcome_message,
            supported_topics=supported_topics,
            quick_actions=quick_actions,
            has_campus_map=True,
            campus_map_locations=locations,
            has_emergency_cards=True,
            emergency_contacts=emergency_contacts,
            footer_text="IIT Jodhpur Academic Core • Powered by Canonical AI Platform",
        )


__all__ = ["IITJUIPlugin"]
