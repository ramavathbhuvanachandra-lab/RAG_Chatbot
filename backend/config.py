"""Backward-compatible configuration bridge.

The new runtime layer owns active deployment configuration.
This module remains temporarily so legacy backend modules can continue to
import the names they already use during the migration.

New code should import from ai_platform.runtime.config instead.
"""

from __future__ import annotations

from ai_platform.runtime.config import PROJECT_ROOT, RUNTIME


# ---------------------------------------------------------------------
# Backward-compatible exports used by the legacy application
# ---------------------------------------------------------------------

INSTITUTION = RUNTIME.institution

DATA_PATH = INSTITUTION.data_path

CHROMA_DB_PATH = INSTITUTION.vectorstore_path

INSTITUTION_ID = INSTITUTION.institution_id

INSTITUTION_NAME = INSTITUTION.display_name

SUPPORTED_LANGUAGES = INSTITUTION.supported_languages

DEFAULT_LANGUAGE = INSTITUTION.default_language

# Additional deployment settings already present in the institution profile.
VECTORSTORE_COLLECTION = INSTITUTION.vectorstore_collection

FALLBACK_MESSAGE = INSTITUTION.fallback_message

ENVIRONMENT = INSTITUTION.environment


__all__ = [
    "PROJECT_ROOT",
    "RUNTIME",
    "INSTITUTION",
    "DATA_PATH",
    "CHROMA_DB_PATH",
    "INSTITUTION_ID",
    "INSTITUTION_NAME",
    "SUPPORTED_LANGUAGES",
    "DEFAULT_LANGUAGE",
    "VECTORSTORE_COLLECTION",
    "FALLBACK_MESSAGE",
    "ENVIRONMENT",
]
