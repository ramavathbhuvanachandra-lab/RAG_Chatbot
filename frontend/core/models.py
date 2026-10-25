"""Frontend Core Models.

Defines validated data models and typed configurations for the modular
Streamlit frontend, institution UI plugins, and temporary conversation sessions.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence


# =============================================================================
# Institution UI Plugin Configuration Models
# =============================================================================

@dataclass(frozen=True)
class QuickActionItem:
    """A preconfigured suggested question or action button."""

    id: str
    label: str
    prompt: str
    icon: str = ""
    action_type: str = "prompt"  # "prompt" or "campus_map"


@dataclass(frozen=True)
class CampusLocationItem:
    """A campus location for structured navigation and map rendering."""

    id: str
    name: str
    category: str = "General"
    description: str = ""
    timings: str = ""
    google_maps: str = ""
    directions: str = ""
    show_in_map: bool = True


@dataclass(frozen=True)
class EmergencyContactItem:
    """An emergency contact record for rapid-response display."""

    id: str
    name: str
    phone: str
    description: str = ""
    help_text: str = ""


@dataclass(frozen=True)
class InstitutionUIConfig:
    """Validated branding and presentation configuration for an institution."""

    institution_id: str
    display_name: str
    tagline: str = "AI Academic & Campus Assistant"
    logo_icon: str = "🎓"
    primary_color: str = "#1E3A8A"
    accent_color: str = "#3B82F6"
    welcome_title: str = ""
    welcome_message: str = ""
    supported_topics: tuple[str, ...] = field(default_factory=tuple)
    quick_actions: tuple[QuickActionItem, ...] = field(default_factory=tuple)
    has_campus_map: bool = False
    campus_map_locations: tuple[CampusLocationItem, ...] = field(default_factory=tuple)
    has_emergency_cards: bool = False
    emergency_contacts: tuple[EmergencyContactItem, ...] = field(default_factory=tuple)
    footer_text: str = "Powered by Canonical AI Platform"

    def validate(self) -> None:
        """Validate configuration values against format constraints."""
        if not self.institution_id or not isinstance(self.institution_id, str):
            raise ValueError("institution_id must be a non-empty string.")

        if not re.match(r"^[a-z0-9_\-]+$", self.institution_id):
            raise ValueError(
                f"institution_id '{self.institution_id}' must consist only of lowercase letters, "
                "digits, hyphens, and underscores."
            )

        if not self.display_name or not self.display_name.strip():
            raise ValueError("display_name must be a non-empty string.")

        if self.has_campus_map and not self.campus_map_locations:
            raise ValueError(
                f"Institution '{self.institution_id}' declared has_campus_map=True but provided no campus_map_locations."
            )


# =============================================================================
# Chat and Conversation State Models
# =============================================================================

@dataclass
class ChatMessage:
    """A single message turn in the active conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    response_time: float | None = None
    structured_blocks: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    is_safe_refusal: bool = False
    evidence_status: str = "supported"
    suggested_questions: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "response_time": self.response_time,
            "structured_blocks": self.structured_blocks,
            "sources": self.sources,
            "is_safe_refusal": self.is_safe_refusal,
            "evidence_status": self.evidence_status,
            "diagnostics": self.diagnostics,
        }


@dataclass
class Conversation:
    """An independent conversation thread with a distinct session ID."""

    id: str
    title: str
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_message(self, message: ChatMessage) -> None:
        """Append a message and update timestamp."""
        self.messages.append(message)
        self.updated_at = time.time()


__all__ = [
    "QuickActionItem",
    "CampusLocationItem",
    "EmergencyContactItem",
    "InstitutionUIConfig",
    "ChatMessage",
    "Conversation",
]
