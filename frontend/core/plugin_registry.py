"""Institution UI Plugin Registry.

Registers and resolves institution-specific presentation plugins for the shared
Streamlit frontend application.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from frontend.core.models import InstitutionUIConfig

logger = logging.getLogger(__name__)


class InstitutionUIPlugin(ABC):
    """Abstract base contract for an institution UI presentation plugin."""

    @property
    @abstractmethod
    def institution_id(self) -> str:
        """The canonical institution identifier (e.g. 'iitj')."""
        raise NotImplementedError

    @abstractmethod
    def get_config(self) -> InstitutionUIConfig:
        """Return the validated configuration for this institution."""
        raise NotImplementedError


_UI_PLUGIN_REGISTRY: dict[str, InstitutionUIPlugin] = {}
_PLUGINS_DISCOVERED = False


def register_ui_plugin(plugin: InstitutionUIPlugin) -> None:
    """Register an institution UI plugin."""
    norm_id = plugin.institution_id.strip().lower()
    _UI_PLUGIN_REGISTRY[norm_id] = plugin
    logger.debug("Registered UI plugin for institution: %s", norm_id)


def unregister_ui_plugin(institution_id: str) -> None:
    """Unregister an institution UI plugin."""
    norm_id = institution_id.strip().lower()
    _UI_PLUGIN_REGISTRY.pop(norm_id, None)


def _ensure_default_plugins_registered() -> None:
    """Automatically discover and register built-in plugins."""
    global _PLUGINS_DISCOVERED
    if _PLUGINS_DISCOVERED:
        return

    # 1. Register IIT Jodhpur plugin
    try:
        from frontend.plugins.iitj.plugin import IITJUIPlugin
        register_ui_plugin(IITJUIPlugin())
    except Exception as exc:
        logger.warning("Could not auto-register IITJ UI plugin: %s", exc)

    # 2. Register template plugin
    try:
        from frontend.plugins.template.plugin import TemplateUIPlugin
        register_ui_plugin(TemplateUIPlugin())
    except Exception as exc:
        logger.warning("Could not auto-register Template UI plugin: %s", exc)

    _PLUGINS_DISCOVERED = True


def get_ui_plugin(institution_id: str | None) -> InstitutionUIPlugin | None:
    """Resolve a registered UI plugin for the specified institution ID."""
    _ensure_default_plugins_registered()
    if not institution_id:
        return None
    return _UI_PLUGIN_REGISTRY.get(institution_id.strip().lower())


def list_registered_plugins() -> list[str]:
    """List all currently registered institution plugin identifiers."""
    _ensure_default_plugins_registered()
    return sorted(list(_UI_PLUGIN_REGISTRY.keys()))


__all__ = [
    "InstitutionUIPlugin",
    "register_ui_plugin",
    "unregister_ui_plugin",
    "get_ui_plugin",
    "list_registered_plugins",
]
