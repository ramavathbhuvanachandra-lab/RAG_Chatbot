"""Load the active institution profile."""

from __future__ import annotations

import importlib
import os

from ai_platform.core.institution.profile import InstitutionProfile


DEFAULT_INSTITUTION = "iitj"


def load_institution_profile(
    institution_id: str | None = None,
) -> InstitutionProfile:
    """Load an institution profile by ID."""

    selected = (
        institution_id
        or os.getenv("INSTITUTION_ID")
        or DEFAULT_INSTITUTION
    ).strip().lower()

    if not selected:
        selected = DEFAULT_INSTITUTION

    module = importlib.import_module(
        f"ai_platform.institutions.{selected}.profile"
    )

    profile = module.PROFILE

    if not isinstance(profile, InstitutionProfile):
        raise TypeError(
            f"ai_platform.institutions.{selected}.profile.PROFILE must be "
            "an InstitutionProfile"
        )

    profile.validate()

    return profile
