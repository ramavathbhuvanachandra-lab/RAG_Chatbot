"""Template Institution UI Plugin.

This module serves as a documented template for onboarding a new institution
to the multi-college Streamlit frontend platform without modifying core code.
"""

from __future__ import annotations

from frontend.core.models import (
    InstitutionUIConfig,
    QuickActionItem,
)
from frontend.core.plugin_registry import InstitutionUIPlugin


class TemplateUIPlugin(InstitutionUIPlugin):
    """Minimal reference UI plugin for onboarding a future college."""

    @property
    def institution_id(self) -> str:
        return "template"

    def get_config(self) -> InstitutionUIConfig:
        quick_actions = (
            QuickActionItem(
                id="qa_admissions",
                label="Admissions & Eligibility",
                prompt="What are the admission requirements and deadlines?",
                icon="🎓",
            ),
            QuickActionItem(
                id="qa_academics",
                label="Academics & Courses",
                prompt="Provide information on programs and course registration.",
                icon="📚",
            ),
        )

        supported_topics = (
            "Undergraduate & Postgraduate Admissions",
            "Academic Regulations & Grading",
            "General Campus Facilities",
        )

        return InstitutionUIConfig(
            institution_id="template",
            display_name="Example Institute of Technology",
            tagline="Academic & Campus AI Assistant",
            logo_icon="🏛️",
            primary_color="#1E3A8A",
            accent_color="#3B82F6",
            welcome_title="Example Institute Assistant",
            welcome_message=(
                "👋 Welcome to the **Example Institute of Technology** AI Assistant.\n\n"
                "Ask questions about our academic regulations, admissions, and campus services."
            ),
            supported_topics=supported_topics,
            quick_actions=quick_actions,
            has_campus_map=False,  # Illustrates optional map omission
            has_emergency_cards=False,  # Illustrates optional emergency omission
            footer_text="Example Institute • Powered by Canonical AI Platform",
        )


__all__ = ["TemplateUIPlugin"]
