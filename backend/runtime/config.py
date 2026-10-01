"""Reusable runtime configuration for the institutional RAG application.

This module contains application-runtime settings only.

Institution-specific identity and data locations come from the active
InstitutionProfile. LLM, vector-store, database, and UI settings will be
split into their own layers later; this file deliberately stays small.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from backend.core.institution import InstitutionProfile
from backend.institutions.loader import load_institution_profile


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable safely."""

    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Runtime state shared by the application."""

    project_root: Path
    institution: InstitutionProfile
    environment: str = "development"
    debug: bool = False

    def validate(self) -> None:
        """Validate the runtime contract."""

        if not self.project_root:
            raise ValueError("project_root cannot be empty")

        self.institution.validate()

        if not self.environment.strip():
            raise ValueError("environment cannot be empty")

    @property
    def institution_id(self) -> str:
        """Active institution identifier."""

        return self.institution.institution_id

    @property
    def data_path(self) -> Path:
        """Active institution knowledge/data root."""

        return self.institution.data_path

    @property
    def vectorstore_path(self) -> Path:
        """Active institution vector-store location."""

        return self.institution.vectorstore_path

    @property
    def structured_data_path(self) -> Path | None:
        """Optional active institution structured-data location."""

        return self.institution.structured_data_path


def load_runtime_config(
    institution_id: str | None = None,
) -> RuntimeConfig:
    """Build the runtime configuration from environment + institution profile."""

    institution = load_institution_profile(institution_id)

    config = RuntimeConfig(
        project_root=PROJECT_ROOT,
        institution=institution,
        environment=os.getenv("APP_ENV", "development").strip() or "development",
        debug=_env_flag("DEBUG", default=False),
    )

    config.validate()

    return config


RUNTIME = load_runtime_config()


__all__ = [
    "PROJECT_ROOT",
    "RuntimeConfig",
    "load_runtime_config",
    "RUNTIME",
]