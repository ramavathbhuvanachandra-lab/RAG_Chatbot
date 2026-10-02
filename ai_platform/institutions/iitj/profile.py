"""IIT Jodhpur deployment profile.

Only deployment identity and institution-specific runtime metadata belong here.
Factual answers must still come from the institution corpus.
"""

from __future__ import annotations

from pathlib import Path

from ai_platform.core.institution import InstitutionProfile


INSTITUTION_ROOT = Path(__file__).resolve().parent
INSTITUTIONS_ROOT = INSTITUTION_ROOT.parent
BACKEND_ROOT = INSTITUTIONS_ROOT.parent
PROJECT_ROOT = BACKEND_ROOT.parent

DATA_ROOT = INSTITUTION_ROOT / "data"

INSTITUTION_ID = "iitj"
DISPLAY_NAME = "IIT Jodhpur"

# Keep this aligned with the collection created by your current ingestion.
VECTORSTORE_COLLECTION = "iitj_v1"

# These paths are deployment configuration, not answer knowledge.
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
    "INSTITUTION_ID",
    "DISPLAY_NAME",
    "VECTORSTORE_COLLECTION",
    "INSTITUTION_ROOT",
    "INSTITUTIONS_ROOT",
    "BACKEND_ROOT",
    "PROJECT_ROOT",
    "DATA_ROOT",
    "INSTITUTION_DATA_ROOT",
    "VECTORSTORE_ROOT",
    "DEFAULT_LANGUAGE",
    "SUPPORTED_LANGUAGES",
    "PROFILE",
]