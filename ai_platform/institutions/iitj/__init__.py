"""IIT Jodhpur institution adapter for the reusable AI platform.

Only IITJ-specific configuration and terminology belong here. Retrieval,
ranking, verification, evidence handling, grounding, and generation remain
owned by :mod:`ai_platform.core`.
"""

from .profile import PROFILE
from .semantic_registry import IITJ_SEMANTIC_REGISTRY, SEMANTIC_REGISTRY
from .lexical import get_lexical_profile
from .source_policy import SOURCE_POLICY
from .scope_policy import SCOPE_POLICY
from .navigation import NAVIGATION_MODEL
from .campus_navigation import find_location
from .emergency import find_emergency

STRUCTURED_HANDLERS = {
    "navigation": find_location,
    "emergency": find_emergency,
}

__all__ = [
    "PROFILE",
    "SEMANTIC_REGISTRY",
    "IITJ_SEMANTIC_REGISTRY",
    "get_lexical_profile",
    "SOURCE_POLICY",
    "SCOPE_POLICY",
    "NAVIGATION_MODEL",
    "find_location",
    "find_emergency",
    "STRUCTURED_HANDLERS",
]
