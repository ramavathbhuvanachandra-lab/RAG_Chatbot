"""Standalone runtime configuration for the reusable AI platform.

The runtime is the composition/infrastructure boundary for a deployment.
It selects the active institution and exposes its deployment settings to
platform modules. It does not import the legacy ``backend`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

from ai_platform.core.institution.profile import InstitutionProfile
from ai_platform.institutions.loader import load_institution_profile


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTITUTION = "iitj"


def _env(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return str(value).strip()


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Validated runtime configuration for one institutional deployment."""

    project_root: Path
    institution: InstitutionProfile
    environment: str = "development"

    def validate(self) -> None:
        if not self.project_root:
            raise ValueError("project_root cannot be empty")
        self.institution.validate()
        if not self.environment.strip():
            raise ValueError("environment cannot be empty")


INSTITUTION_ID = (
    _env("INSTITUTION_ID", DEFAULT_INSTITUTION).casefold()
    or DEFAULT_INSTITUTION
)

RUNTIME = RuntimeConfig(
    project_root=PROJECT_ROOT,
    institution=load_institution_profile(INSTITUTION_ID),
    environment=_env("AI_PLATFORM_ENV", "development") or "development",
)
RUNTIME.validate()

# Platform-only convenience exports. These are intentionally owned by the
# new runtime package and do not proxy through the legacy application.
INSTITUTION = RUNTIME.institution
DATA_PATH = INSTITUTION.data_path
CHROMA_DB_PATH = INSTITUTION.vectorstore_path
INSTITUTION_NAME = INSTITUTION.display_name
SUPPORTED_LANGUAGES = INSTITUTION.supported_languages
DEFAULT_LANGUAGE = INSTITUTION.default_language
VECTORSTORE_COLLECTION = INSTITUTION.vectorstore_collection
FALLBACK_MESSAGE = INSTITUTION.fallback_message
ENVIRONMENT = RUNTIME.environment


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_INSTITUTION",
    "INSTITUTION_ID",
    "RuntimeConfig",
    "RUNTIME",
    "INSTITUTION",
    "DATA_PATH",
    "CHROMA_DB_PATH",
    "INSTITUTION_NAME",
    "SUPPORTED_LANGUAGES",
    "DEFAULT_LANGUAGE",
    "VECTORSTORE_COLLECTION",
    "FALLBACK_MESSAGE",
    "ENVIRONMENT",
]
