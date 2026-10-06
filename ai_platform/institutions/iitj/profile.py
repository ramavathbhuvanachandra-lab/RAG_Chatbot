"""IIT Jodhpur deployment profile.

This module contains deployment identity and institution-specific runtime
metadata only. Factual answers must come from the IITJ knowledge corpus.
"""

from __future__ import annotations

from pathlib import Path

from ai_platform.core.institution import InstitutionProfile


INSTITUTION_ROOT = Path(__file__).resolve().parent
INSTITUTIONS_ROOT = INSTITUTION_ROOT.parent
PLATFORM_ROOT = INSTITUTIONS_ROOT.parent
PROJECT_ROOT = PLATFORM_ROOT.parent

# Small, versioned institution configuration lives beside the adapter code.
DATA_ROOT = INSTITUTION_ROOT / "data"
LEXICAL_DATA_ROOT = DATA_ROOT

# Deployment identifiers.
INSTITUTION_ID = "iitj"
DISPLAY_NAME = "IIT Jodhpur"

# Existing Chroma collection created by the current IITJ ingestion pipeline.
VECTORSTORE_COLLECTION = "iitj_v1"

# The factual corpus is external to this package. These paths identify the
# deployment's corpus and vector-store roots; they do not contain answer text.
INSTITUTION_DATA_ROOT = PROJECT_ROOT / "data" / "data_iitj"
VECTORSTORE_ROOT = PROJECT_ROOT / "chroma_db"

DEFAULT_LANGUAGE = "English"
SUPPORTED_LANGUAGES = ("English", "Hindi", "Hinglish")

PROFILE = InstitutionProfile(
    institution_id=INSTITUTION_ID,
    display_name=DISPLAY_NAME,
    data_path=INSTITUTION_DATA_ROOT,
    vectorstore_path=VECTORSTORE_ROOT,
    vectorstore_collection=VECTORSTORE_COLLECTION,
    default_language=DEFAULT_LANGUAGE,
    supported_languages=SUPPORTED_LANGUAGES,
    answer_tone="student_friendly",
    answer_verbosity="moderate",
)

PROFILE.validate()

__all__ = [
    "INSTITUTION_ROOT",
    "INSTITUTIONS_ROOT",
    "PLATFORM_ROOT",
    "PROJECT_ROOT",
    "DATA_ROOT",
    "LEXICAL_DATA_ROOT",
    "INSTITUTION_ID",
    "DISPLAY_NAME",
    "VECTORSTORE_COLLECTION",
    "INSTITUTION_DATA_ROOT",
    "VECTORSTORE_ROOT",
    "DEFAULT_LANGUAGE",
    "SUPPORTED_LANGUAGES",
    "PROFILE",
]
