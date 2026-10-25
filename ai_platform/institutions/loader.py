"""Load the active institution profile."""

from __future__ import annotations

import importlib
import os

from ai_platform.core.institution.profile import InstitutionProfile


from typing import Any, Callable

DEFAULT_INSTITUTION = "iitj"

_INSTITUTION_REGISTRY: dict[str, InstitutionProfile] = {}
_STRUCTURED_HANDLERS_REGISTRY: dict[str, dict[str, Callable[[str], Any]]] = {}


def register_institution_profile(profile: InstitutionProfile) -> None:
    """Register an institution profile in-memory for testing or dynamic tenancies."""
    profile.validate()
    _INSTITUTION_REGISTRY[profile.institution_id.strip().lower()] = profile


def unregister_institution_profile(institution_id: str) -> None:
    """Remove a registered institution profile."""
    _INSTITUTION_REGISTRY.pop(institution_id.strip().lower(), None)


def register_institution_handler(
    institution_id: str,
    handler_type: str,
    handler: Callable[[str], Any],
) -> None:
    """Register a structured capability handler for an institution in-memory."""
    norm_id = institution_id.strip().lower()
    if norm_id not in _STRUCTURED_HANDLERS_REGISTRY:
        _STRUCTURED_HANDLERS_REGISTRY[norm_id] = {}
    _STRUCTURED_HANDLERS_REGISTRY[norm_id][handler_type.strip().lower()] = handler


def unregister_institution_handler(
    institution_id: str,
    handler_type: str | None = None,
) -> None:
    """Remove a registered structured handler for an institution."""
    norm_id = institution_id.strip().lower()
    if norm_id in _STRUCTURED_HANDLERS_REGISTRY:
        if handler_type:
            _STRUCTURED_HANDLERS_REGISTRY[norm_id].pop(handler_type.strip().lower(), None)
        else:
            _STRUCTURED_HANDLERS_REGISTRY.pop(norm_id, None)


def get_institution_handler(
    institution_id: str | None,
    handler_type: str,
) -> Callable[[str], Any] | None:
    """Resolve an optional structured handler for the given institution.

    Order of resolution:
    1. In-memory registered handlers (for dynamic or test tenancies).
    2. Static exports in the institution package (e.g. STRUCTURED_HANDLERS in ai_platform.institutions.<id>).
    3. Direct convention module lookup (e.g. ai_platform.institutions.<id>.campus_navigation.find_location).
    """
    selected = (
        institution_id
        or os.getenv("INSTITUTION_ID")
        or DEFAULT_INSTITUTION
    ).strip().lower()
    if not selected:
        selected = DEFAULT_INSTITUTION

    norm_type = handler_type.strip().lower()

    # 1. In-memory registry check
    if selected in _STRUCTURED_HANDLERS_REGISTRY:
        handler = _STRUCTURED_HANDLERS_REGISTRY[selected].get(norm_type)
        if handler is not None:
            return handler

    # 2. Check institution package for STRUCTURED_HANDLERS map
    try:
        pkg = importlib.import_module(f"ai_platform.institutions.{selected}")
        structured_map = getattr(pkg, "STRUCTURED_HANDLERS", None)
        if isinstance(structured_map, dict) and norm_type in structured_map:
            return structured_map[norm_type]
    except Exception:
        pass

    # 3. Known convention module and function lookup
    type_to_module_and_fn = {
        "navigation": ("campus_navigation", "find_location"),
        "emergency": ("emergency", "find_emergency"),
    }
    if norm_type in type_to_module_and_fn:
        mod_name, fn_name = type_to_module_and_fn[norm_type]
        try:
            mod = importlib.import_module(f"ai_platform.institutions.{selected}.{mod_name}")
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn
        except Exception:
            pass

    return None


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

    if selected in _INSTITUTION_REGISTRY:
        return _INSTITUTION_REGISTRY[selected]

    module = importlib.import_module(
        f"ai_platform.institutions.{selected}.profile"
    )

    profile = module.PROFILE

    if not isinstance(profile, InstitutionProfile):
        raise TypeError(
            f"ai_platform.institutions.{selected}.profile.PROFILE must be "
            "an InstitutionProfile"
        )

    # Check active release configuration for dynamically configured release/collection
    try:
        from ai_platform.packages.registry import get_active_release_config
        active_cfg = get_active_release_config(selected)
        if active_cfg and (
            active_cfg.get("status") in {"active", "rolled_back", "staged"}
            or os.getenv("VECTORSTORE_COLLECTION")
            or os.getenv("RAG_USE_STAGING", "").strip().casefold() in {"1", "true", "yes"}
        ):
            from dataclasses import replace
            from pathlib import Path
            coll = active_cfg.get("vectorstore_collection") or profile.vectorstore_collection
            v_path = Path(active_cfg.get("vectorstore_path") or profile.vectorstore_path)
            d_path = Path(active_cfg.get("package_root") or profile.data_path)
            rel_name = str(active_cfg.get("release_id") or active_cfg.get("active_release") or "default_release")
            meta = tuple(profile.metadata) + (
                ("active_release", rel_name),
            )
            profile = replace(
                profile,
                vectorstore_collection=coll,
                vectorstore_path=v_path,
                data_path=d_path,
                metadata=meta,
            )
    except Exception:
        pass

    profile.validate()

    return profile


__all__ = [
    "DEFAULT_INSTITUTION",
    "register_institution_profile",
    "unregister_institution_profile",
    "load_institution_profile",
    "register_institution_handler",
    "unregister_institution_handler",
    "get_institution_handler",
]
