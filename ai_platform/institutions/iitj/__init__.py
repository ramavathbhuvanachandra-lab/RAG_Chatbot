"""IIT Jodhpur institution adapter.

This package contains IITJ-specific configuration and terminology only.
Reusable retrieval, ranking, verification, evidence, and generation remain
owned by the shared backend/core and backend/runtime layers.
"""

from .profile import PROFILE
from .semantic_registry import SEMANTIC_REGISTRY, IITJ_SEMANTIC_REGISTRY
from .lexical import get_lexical_profile
from .source_policy import SOURCE_POLICY
from .scope_policy import SCOPE_POLICY
from .navigation import NAVIGATION_MODEL

__all__ = [
    "PROFILE",
    "SEMANTIC_REGISTRY",
    "IITJ_SEMANTIC_REGISTRY",
    "get_lexical_profile",
    "SOURCE_POLICY",
    "SCOPE_POLICY",
    "NAVIGATION_MODEL",
]