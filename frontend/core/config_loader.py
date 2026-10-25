"""Institution Configuration Loader.

Loads, validates, and cross-checks the selected institution UI configuration
against registered AI-platform profiles.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from ai_platform.institutions.loader import (
    DEFAULT_INSTITUTION,
    load_institution_profile,
)
from frontend.core.plugin_registry import get_ui_plugin

if TYPE_CHECKING:
    from frontend.core.models import InstitutionUIConfig

logger = logging.getLogger(__name__)


def get_active_tenant_id() -> str:
    """Resolve the active institution identifier from environment with safe default."""
    tenant = os.getenv("INSTITUTION_ID", DEFAULT_INSTITUTION)
    return (tenant or DEFAULT_INSTITUTION).strip().lower()


def load_institution_ui_config(
    institution_id: str | None = None,
    *,
    verify_ai_profile: bool = True,
) -> InstitutionUIConfig:
    """Load and validate the presentation configuration for the given institution.

    Parameters:
        institution_id: The institution to load (defaults to active tenant).
        verify_ai_profile: Whether to cross-validate that the AI-platform
                           institutional profile exists and matches identity.

    Returns:
        The validated `InstitutionUIConfig`.

    Raises:
        KeyError: If no UI plugin is registered for the institution ID.
        ValueError: If config validation fails or if there is a tenant mismatch.
    """
    target_id = (institution_id or get_active_tenant_id()).strip().lower()

    plugin = get_ui_plugin(target_id)
    if plugin is None:
        raise KeyError(
            f"No UI plugin registered for institution '{target_id}'. "
            "Ensure the institution UI plugin is created and registered."
        )

    config = plugin.get_config()
    config.validate()

    if config.institution_id.lower() != target_id:
        raise ValueError(
            f"UI plugin configuration identity mismatch: requested '{target_id}' "
            f"but plugin returned '{config.institution_id}'."
        )

    if verify_ai_profile:
        try:
            profile = load_institution_profile(target_id)
            if profile.institution_id.lower() != config.institution_id.lower():
                raise ValueError(
                    f"Tenant mismatch between UI plugin ('{config.institution_id}') "
                    f"and AI platform profile ('{profile.institution_id}')."
                )
        except Exception as exc:
            logger.warning(
                "Could not cross-validate AI profile for institution '%s': %s",
                target_id,
                exc,
            )

    return config


__all__ = [
    "get_active_tenant_id",
    "load_institution_ui_config",
]
